import astropy.time as atime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

############################## SETTINGS ##############################

"""
This script fits a simple sinusoidal to a determined RV-curve. 
It is not recommended to use this script for RV-curves with large variations in step sizes,
or RV-curves with little datapoints (<20 good datapoints), as results will be very inaccurate.
"""

FILE_PREFIX = "spec-0312-51689-0031"  # File prefix of the RV-Curve to be plotted
PERIOD_UNIT = "auto"
"""
PERIOD_UNIT: Unit the Period is outputted as, may be any of 
"auto" - to automatically select the best unit
"m"- minutes
"h"- hours
"d"- days
"""
MANUAL_INITIAL_GUESS = [150, 24 / 5.56, 0, 0]
"""
MANUAL_INITIAL_GUESS: Manually defined initial guess for the sinusoidal Fit.
Available parameters are, in this order with required units:
Amplitude [km/s]
Frequency [1/days]
RV Offset [km/s]
Period shift phi [days]
"""
GAIA_ID = "3376639486380091008"


############################## FUNCTIONS ##############################


class Sinusfit:
    def __init__(self, A, b, h, shift, errs):
        self.A = A
        self.b = b
        self.h = h
        self.shift = shift
        self.errs = errs

    def evaluate(self, x):
        return self.h + self.A * np.sin(self.b * 2 * np.pi * (x - self.shift))

    def get_params(self):
        return [self.A, self.b, self.h, self.shift]


class Cosinefit:
    def __init__(self, A, b, h, shift, errs):
        self.A = A
        self.b = b
        self.h = h
        self.shift = shift
        self.errs = errs

    def evaluate(self, x):
        return self.h + self.A * np.cos(self.b * 2 * np.pi * (x - self.shift))

    def get_params(self):
        return [self.A, self.b, self.h, self.shift]


def sinusoid(x, A, b, h, shift):
    return h + A * np.sin(b * 2 * np.pi * (x - shift))


def cosinusoid(x, A, b, h, shift):
    return h + A * np.cos(b * 2 * np.pi * (x - shift))


def fit_rv_curve(showplot=True, manguess=MANUAL_INITIAL_GUESS, gaia_id=GAIA_ID, outdir="output"):
    data = pd.read_csv(rf"{outdir}\{gaia_id}\RV_variation.csv")

    stime = np.array([atime.Time(t, format="mjd").to_value(format="mjd") for t in data["mjd"]])
    stime -= np.amin(stime)
    RV = data["culum_fit_RV"]
    RV_err = data["u_culum_fit_RV"]

    if manguess is None:
        # Compute guess parameters via Lomb-Scrargle periodogram (https://doi.org/10.1007/BF00648343)
        frequency, power = LombScargle(stime, RV, RV_err, nterms=100).autopower()

        plt.plot(frequency, power)
        plt.show()

        freq = abs(frequency[np.argmax(power)])
        amp = np.ptp(RV) / 2
        offset = np.mean(RV)
        p0 = [amp, freq, offset, 0]

    else:
        p0 = manguess

    # plt.scatter(stime, RV)
    # timespace = np.linspace(np.amin(stime), np.amax(stime), 1000)
    # plt.plot(timespace, sinusoid(timespace, *p0))
    # plt.show()

    try:
        params, errs = curve_fit(sinusoid,
                                 stime,
                                 RV,
                                 p0=p0,
                                 sigma=RV_err,
                                 bounds=[
                                     [0, 0, -np.inf, -np.inf],
                                     [np.inf, np.inf, np.inf, np.inf]
                                 ],
                                 maxfev=100000)
    except RuntimeError:
        return None

    errs = np.sqrt(np.diag(errs))

    fit = Sinusfit(*params, errs)

    from main import plot_rvcurve_brokenaxis

    plot_rvcurve_brokenaxis(RV, RV_err, stime, gaia_id, fit=fit, custom_saveloc=f"images/{gaia_id}.pdf")

    if showplot:
        plt.show()
    else:
        plt.close()
        plt.clf()
        plt.cla()
    return fit


def sinusoid_fold_wrapper(x, amp, offset, shift):
    return sinusoid(x, amp, 1, offset, shift)


def phasefold(vels, verrs, times, period, gaia_id, p0=None, predetermined=True, custom_saveloc=None):
    """
    :param vels: Radial velocities [km/s]
    :param verrs: Radial velocity errors [km/s]
    :param times: Epochs of the observations
    :param period: period to fold with
    :param gaia_id: GAIA ID of the star
    :param params: optional fit parameters
    :return: phase-folded times
    """

    times = np.array(times)
    times_normalized = (times - np.min(times)) % period / period

    plt.close()
    fig, ax1 = plt.subplots(figsize=(4.8 * 16 / 9, 4.8))
    fig.suptitle(f"Phase-folded Radial Velocity\n Gaia EDR3 {gaia_id}", fontsize=17)
    ax1.set_ylabel("Radial Velocity [km/s]", fontsize=14)
    ax1.set_xlabel("Phase", fontsize=14)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if not p0:
        params, errs = curve_fit(sinusoid_fold_wrapper,
                                 times_normalized,
                                 vels,
                                 p0=[np.ptp(vels) / 2, np.mean(vels), 0],
                                 sigma=verrs,
                                 bounds=[
                                     [0, -np.inf, -1],
                                     [np.inf, np.inf, 1]
                                 ],
                                 maxfev=100000)
    else:
        if predetermined:
            params = p0
        else:
            params, errs = curve_fit(sinusoid_fold_wrapper,
                                     times_normalized,
                                     vels,
                                     p0=p0,
                                     sigma=verrs,
                                     bounds=[
                                         [0, -np.inf, -1],
                                         [np.inf, np.inf, 1]
                                     ],
                                     maxfev=100000)

    if params[-1] < 0:
        params[-1] = 1+params[-1]

    times_normalized -= params[-1]
    times_normalized[times_normalized < 0] += 1

    times_normalized = np.concatenate([times_normalized-1, times_normalized])
    vels = np.concatenate([vels, vels])
    verrs = np.concatenate([verrs, verrs])

    ax1.scatter(times_normalized, vels, zorder=5, color=colors[0], label="Radial Velocities")
    ax1.errorbar(times_normalized, vels, yerr=verrs, capsize=3, linestyle='', zorder=4, color=colors[1])

    fit = Sinusfit(params[0], 1, params[1], 0, verrs)

    timespace = np.linspace(-1, 1, 1000)
    ax1.plot(timespace,
             fit.evaluate(timespace),
             color="darkred",
             zorder=3,
             label="fit")

    culumvs_range = verrs.min()

    if not pd.isnull(culumvs_range):
        ax1.set_ylim((vels.min() - 2 * culumvs_range, vels.max() + 2 * culumvs_range))

    ax1.set_xlim((-1, 1))

    if period < 1:
        annotation_text = f"K = {params[0]:.2f} km/s\nP = {period*24:.2f}h\nφ = {params[2]:.2f}\nOffset = {params[1]:.2f} km/s"
    else:
        annotation_text = f"K = {params[0]:.2f} km/s\nP = {period:.2f}d\nφ = {params[2]:.2f}\nOffset = {params[1]:.2f} km/s"


    plt.annotate(annotation_text, xy=(0.05, 0.1), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange'))

    plt.grid(True, color="gray", linestyle="--", linewidth=1, zorder=1)
    plt.legend(loc='upper right')

    plt.tight_layout()

    if not custom_saveloc:
        plt.savefig(f"images/{gaia_id}_phfold.pdf")
    else:
        plt.savefig(custom_saveloc, dpi=300)

    plt.show()


############################## EXECUTION ##############################


if __name__ == "__main__":
    fit = fit_rv_curve()

    A, b, h, shift = fit.get_params()
    u_A, u_b, _, _ = fit.errs

    print(f"Half-Amplitude K = [{np.abs(A)}+-{u_A}] km/s")
    punits = ["m", "h", "d", "y"]
    punit = punits[np.argmin([
        np.abs(1 - (60 * 24 / b)),
        np.abs(1 - (24 / b)),
        np.abs(1 - 1 / b),
        np.abs(1 - 1 / (365 * b))]
    )]

    if punit == "m":
        print(f"Period p = [{60 * 24 / b}+-{60 * 24 * u_b / b ** 2}] m")
    elif punit == "h":
        print(f"Period p = [{24 / b}+-{24 * u_b / b ** 2}] h")
    elif punit == "d":
        print(f"Period p = [{1 / b}+-{u_b / b ** 2}] d")
    elif punit == "y":
        print(f"Period p = [{1 / (365 * b)}+-{u_b / (365 * b ** 2)}] y")
