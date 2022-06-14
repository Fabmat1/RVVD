import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import pandas as pd
import astropy.time as atime
import numpy as np

############################## SETTINGS ##############################

"""
This script fits a simple sinusoidal to a determined RV-curve. 
It is not recommended to use this script for RV-curves with large variations in step sizes,
or RV-curves with little datapoints (<20 good datapoints), as results will be very inaccurate.
"""


FILE_PREFIX = "spec-2682-54401-0569"  # File prefix of the RV-Curve to be plotted
PERIOD_UNIT = "auto"
"""
PERIOD_UNIT: Unit the Period is outputted as, may be any of 
"auto" - to automatically select the best unit
"m"- minutes
"h"- hours
"d"- days
"y"- years
"""
MANUAL_INITIAL_GUESS = [50, 1, 0, 0]
"""
MANUAL_INITIAL_GUESS: Manually defined initial guess for the sinusoidal Fit.
Available parameters are, in this order with required units:
Amplitude [km/s]
Frequency [1/day]
RV Offset [km/s]
Period shift phi [1/day]
"""

############################## FUNCTIONS ##############################


def sinusoid(x, A, b, h, shift):
    return h + A * np.sin(b * 2 * np.pi * (x - shift))


def compute_nfft(sample_instants, sample_values):
    x = np.fft.fftfreq(len(sample_instants), (sample_instants[1] - sample_instants[0]))  # assume uniform spacing
    y = abs(np.fft.fft(sample_values))
    return x, y


def fit_rv_curve(fpre=FILE_PREFIX, punit=PERIOD_UNIT, showplot=True, verbose=True):
    data = pd.read_csv(rf"output\{fpre}\RV_variation.csv")

    stime = np.array([atime.Time(t, format="mjd").to_value(format="mjd") for t in data["mjd"]])
    stime -= np.amin(stime)
    RV = data["culum_fit_RV"]
    RV_err = data["u_culum_fit_RV"]

    # Compute guess parameters via Lomb-Scrargle periodogram (https://doi.org/10.1007/BF00648343)
    frequency, power = LombScargle(stime, RV).autopower()

    freq = abs(frequency[np.argmax(power)+1 if np.argmax(power)+1 != len(power) else np.argmax(power)])
    amp = np.std(RV) * 2. ** 0.5
    offset = np.mean(RV)
    p0 = [amp, freq, offset, 0]

    try:
        params, errs = curve_fit(sinusoid,
                                 stime,
                                 RV,
                                 p0=p0,
                                 sigma=RV_err,
                                 bounds=[
                                     [0, 0, -np.inf, -np.inf],
                                     [0.75*(np.abs(np.amax(RV)-np.amin(RV))), np.inf, np.inf, np.inf]
                                 ],
                                 maxfev=10000)
    except RuntimeError:
        return ["--", "--"], ["--", "--"]


    A, b, h, shift = params
    errs = np.sqrt(np.diag(errs))
    u_A, u_b, u_h, u_shift = errs

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    p = 1/b
    u_p = u_b / b ** 2

    if verbose:
        print(f"Half-Amplitude K = [{np.abs(A)}+-{u_A}] km/s")
        punits = ["m", "h", "d", "y"]
        if punit not in punits:
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

    mjd_linspace = np.linspace(np.amin(stime), np.amax(stime), 1000)

    plt.title(f"Radial Velocity over Time\n Gaia EDR3 944390774983674496")
    plt.ylabel("Radial Velocity [km/s]")
    plt.xlabel("Date")
    plt.scatter(stime, RV, zorder=5, color=colors[0])
    plt.plot(mjd_linspace, sinusoid(mjd_linspace, *params), zorder=2, color=colors[2])
    plt.errorbar(stime, RV, yerr=RV_err, capsize=3, linestyle='', zorder=3, color=colors[1], label='_nolegend_')
    plt.legend(["Measured Values", "Best Fit"], loc="upper left")
    plt.savefig(rf"output\{fpre}\RV_var_plusfit.png", dpi=300)
    if showplot:
        plt.show()
    else:
        plt.close()
        plt.clf()
        plt.cla()
    return [A, p], [u_A, u_p]


############################## EXECUTION ##############################


if __name__ == "__main__":
    fit_rv_curve()
