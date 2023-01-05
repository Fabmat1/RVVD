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
"y"- years
"""
MANUAL_INITIAL_GUESS = [280, 13.05, -200, 3968.0697424]
"""
MANUAL_INITIAL_GUESS: Manually defined initial guess for the sinusoidal Fit.
Available parameters are, in this order with required units:
Amplitude [km/s]
Frequency [1/days]
RV Offset [km/s]
Period shift phi [days]
"""
GAIA_ID = "4415762082969078400"


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
