# TODO: Fit single Peak by doing continuum fit and gauss-Lorentz fit
# TODO: Error Estimation via monte-Carlo method
import glob
import os
from datetime import datetime
from pprint import pprint

import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.special import erf
from scipy.constants import c
from astropy.io import fits
import astropy.time as atime

FILE_LOC = "data/sdss_spec.fits"
DAT_TYPE = "numeric"  # dict, numeric
OUTLIER_MAX_SIGMA = 2
PLOTOVERVIEW = False

lines = {
    "H_alpha": 6562.79,
    "H_beta": 4861.35,
    "H_gamma": 4340.472,
    "H_delta": 4101.734,
    "He_I_4472": 4471.4802,  # ?
    "He_I_4922": 4921.9313,
    "He_II_4541": 4541,  # ???
    "He_II_4686": 4685.70,
    "He_II_5412": 5411.52
}


class NoiseWarning(UserWarning):
    pass


class InaccurateDateWarning(UserWarning):
    pass


class FitUnsuccessfulWarning(UserWarning):
    pass


def slicearr(arr, lower, upper):
    """
    :param arr: array to be sliced into parts
    :param lower: lower bound-value in array
    :param upper: upper bound-value in array
    :return: sliced array, indices of subarray in old array
    """
    try:
        assert lower < upper
        newarr = arr[arr > lower]
        newarr = newarr[newarr < upper]
    except AssertionError as e:
        print("Lower bound larger than upper bound!")
        exit()
    if len(newarr) == 0:
        loind = np.where(arr == arr[arr > lower][0])[0][0]
        upind = loind + 1
    else:
        loind, upind = np.where(arr == newarr[0])[0][0], np.where(arr == newarr[-1])[0][0] + 1
    return newarr, loind, upind


def faddeeva(z):
    return np.exp(-z ** 2) * (1 - erf(-1.j * z))


def lorentzian(x, gamma, x_0):
    return 1 / np.pi * (gamma / 2) / ((x - x_0) ** 2 + (gamma / 2) ** 2)


def gaussian(x, gamma, x_0):
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp((-(x - x_0) ** 2) / (2 * sigma ** 2))


def v_from_doppler(f_o, f_s):
    """
    :param f_o: Observed Frequency
    :param f_s: Source Frequency
    :return: Velocity Shift calculated from relativistic doppler effect
    """
    return c * (f_o ** 2 - f_s ** 2) / (f_o ** 2 + f_s ** 2)


def v_from_doppler_err(f_o, f_s, u_f_o, u_f_s):
    """
    :param f_o: Observed Frequency
    :param f_s: Source Frequency
    :param u_f_o: Uncertainty of Observed Frequency
    :param u_f_s: Uncertainty of Source Frequency
    :return: Uncertainty for Velocity Shift calculated from relativistic doppler effect
    """
    return c * ((4 * f_o ** 2 * f_s) / ((f_o ** 2 + f_s ** 2) ** 2) * u_f_s + (4 * f_o * f_s ** 2) / (
            (f_o ** 2 + f_s ** 2) ** 2) * u_f_o)


def to_sigma(gamma):
    return gamma/(2*np.sqrt(2*np.log(2)))


def height_err(eta, sigma, gamma, scaling, u_eta, u_sigma, u_gamma, u_scaling):
    return u_scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (gamma * np.pi)) + \
           u_sigma * eta * scaling / (sigma ** 2 * np.sqrt(2 * np.pi)) + \
           u_eta * scaling* ( 1/ (sigma * np.sqrt(2 * np.pi)) - 2 / (gamma * np.pi)) + \
           u_gamma * scaling * (1 - eta) * 2 / (gamma**2 * np.pi)


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    # z = (x + 1.j * gamma) / (sigma * np.sqrt(2))
    # return -scaling * (np.real(faddeeva(z)) / (sigma * np.sqrt(2 * np.pi))) + slope * (x - shift) + height
    return -scaling * (eta * gaussian(x, gamma, shift) + (1 - eta) * lorentzian(x, gamma, shift)) + slope * x + height


def load_spectrum(filename):
    """
    :param filename: Spectrum File location
    :return: Spectrum Wavelengths, Corresponding flux
    """
    if filename.endswith(".csv"):
        data = pd.read_csv(filename)
        data = data.to_numpy()

        wavelength = data[:, 0]
        flux = data[:, 1]
    elif filename.endswith(".fits"):
        hdul = fits.open(filename)
        data = hdul[1].data
        # pprint(hdul[0].header)
        try:
            tai = hdul[0].header["TAI"]
            time = atime.Time(tai+atime.Time(datetime.strptime("17/11/1858", '%d/%m/%Y')).to_value(format="unix_tai"), format="unix_tai")
        except KeyError:
            warnings.warn("Could not get TAI timestamp, trying MJD...", NoiseWarning)
            mjd = hdul[0].header["MJD"]
            time = atime.Time(mjd, format="mjd")
        flux = data["flux"]
        wavelength = 10 ** data["loglam"]
    else:
        raise FileNotFoundError
    if PLOTOVERVIEW:
        plt.title("Full Spectrum Overview")
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.plot(wavelength, flux)
        plt.show()
    return wavelength, flux, time


def verify_peak(wavelength, sigma):
    """
    :param wavelength: Wavelength array
    :param sigma: Standard deviation of Fit Function
    :return: Returns False if the Peak width is in the order of one step in the Wavelength array
    """
    d_wl = wavelength[1] - wavelength[0]
    if sigma < d_wl:
        return False
    else:
        return True


def calc_SNR(params, flux, wavelength, margin):
    """
    :param params: parameters of Fit
    :param flux: Flux array
    :param wavelength: Wavelength array
    :param margin: index width of plotted area (arbitrary)
    :return:    Mean squared displacement(MSD) of signal area,
                MSD of noise background,
                Logarithmic Signal-to-Noise ratio
    """
    scaling, gamma, shift, slope, height, eta = params
    wlmargin = np.abs(wavelength[0] - wavelength[margin])
    flux = flux - slope * wavelength - height
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))

    slicedwl, loind, upind = slicearr(wavelength, shift - 2 * sigma, shift + 2 * sigma)
    signalstrength = np.mean(np.square(flux[loind:upind]))

    if 2 * sigma < wlmargin:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - wlmargin, shift - 2 * sigma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + 2 * sigma, shift + wlmargin)
    else:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - wlmargin, shift - sigma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + sigma, shift + wlmargin)
        warnings.warn("Sigma very large, Fit seems improbable!", NoiseWarning)
        plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    color="red")

    noisestrength = np.mean(np.square(np.array(flux[lloind:lupind].tolist() + flux[uloind:uupind].tolist())))

    SNR = 10 * np.log10(signalstrength / noisestrength)

    return signalstrength, noisestrength, SNR


def plot_peak_region(wavelength, flux, center, margin, fit, time):
    slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)
    fig = plt.plot(slicedwl, flux[loind:upind])
    sucess = True

    if fit:
        initial_h = np.mean(flux[loind:upind])
        initial_s = np.abs((initial_h - np.min(flux[
                                               round(loind + abs(upind - loind) / 2 - abs(upind - loind) / 20):round(
                                                   loind + abs(upind - loind) / 2 + abs(
                                                       upind - loind) / 20)])) / 0.71725216658522349)
        if initial_s == np.nan:
            initial_s = 1
        try:
            params, errs = curve_fit(pseudo_voigt,
                                     slicedwl,
                                     flux[loind:upind],
                                     [initial_s, 5, center, 0, initial_h, 0.5],
                                     # scaling, gamma, shift, slope, height, eta
                                     bounds=(
                                         [0, 0, center - 15, -np.inf, 0, 0],
                                         [np.inf, np.sqrt(2 * np.log(2)) * margin / 4, center + 15, np.inf, np.inf, 1]
                                     ))
            errs = np.sqrt(np.diag(errs))

            # plt.plot(slicedwl, pseudo_voigt(slicedwl, initial_s, 1, 1, center, 0, initial_h, 0.5))
            if not verify_peak(wavelength, params[1] / (2 * np.sqrt(2 * np.log(2)))):
                if sucess:
                    plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                color="red")
                warnings.warn(f"WL {center}Å: Peak seems to be fitted to Noise, SNR and peak may not be accurate.",
                              NoiseWarning)
                sucess = False
            sstr, nstr, SNR = calc_SNR(params, flux, wavelength, margin)
            plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}dB ", (10, 10), xycoords="figure pixels")
            if SNR < 2:
                if sucess:
                    plt.figtext(0.3, 0.95, f"BAD SIGNAL!",
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                color="red")
                warnings.warn(f"WL {center}Å: Signal-to-Noise ratio out of bounds!",
                              NoiseWarning)
                sucess = False
            plt.plot(slicedwl,
                     pseudo_voigt(slicedwl, params[0], params[1], params[2], params[3], params[4], params[5]), zorder=5)
        except RuntimeError:
            sucess = False
            warnings.warn("Could not find a good Fit!", FitUnsuccessfulWarning)
            plt.figtext(0.3, 0.95, f"FIT FAILED!",
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color="red")
            plt.plot(slicedwl, pseudo_voigt(slicedwl, initial_s, 1, center, 0, initial_h, 0.5), zorder=5)
        except ValueError as e:
            print("No peak found:", e)
            plt.figtext(0.3, 0.95, f"FIT FAILED!",
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color="red")
            sucess = False
            plt.plot(slicedwl, pseudo_voigt(slicedwl, initial_s, 1, center, 0, initial_h, 0.5), zorder=5)
    plt.axvline(center, linewidth=0.5, color='lightgrey', linestyle='dashed', zorder=1)
    plt.legend(["Flux", "Best Fit"])
    outtimestr = time.to_datetime().strftime("%m_%d_%Y__%H_%M_%S")
    if not os.path.isdir(f'output/spec_{outtimestr}/'):
        os.mkdir(f"output/spec_{outtimestr}/")
    plt.savefig(f"output/spec_{outtimestr}/plot_{round(center)}Å", dpi=500)
    plt.show()
    if sucess:
        return sucess, errs, params
    else:
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False]


def print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc):
    if sucess:
        sigma = to_sigma(gamma)
        sigma_err = to_sigma(errs[1])
        print("######################## FIT RESULTS ########################")
        print(f"Result for line {lstr} @ {round(loc)}Å:")
        print(f"\nPeak Height I={scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))}±{height_err(eta,sigma,gamma,scaling,errs[5],sigma_err,errs[1],errs[0])}")
        print(f"Standard deviation σ={sigma}±{sigma_err}")
        print(f"Peak location x_0={shift}±{errs[2]}")
        print("#############################################################\n\n")
    else:
        print("######################## FIT RESULTS ########################")
        print(f"Result for line {lstr} @ {round(loc)}Å:")
        print(f"FIT FAILED!")
        print("#############################################################\n\n")


def print_single_spec_results(complete_v_shift, v_std, filename):
    print("\n\n##################### SPECTRUM RESULTS ######################")
    print(f"Result for Spectrum {os.path.basename(filename)}:")
    print(f"Velocity: [{round(complete_v_shift/1000, 2)}±{round(v_std/1000, 2)}]km/s")
    print("#############################################################\n\n")


def check_for_outliers(array):
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    outlierloc = distance_from_mean < OUTLIER_MAX_SIGMA * standard_deviation
    print(distance_from_mean, OUTLIER_MAX_SIGMA * standard_deviation)
    return np.array(outlierloc)


def single_spec_shift(filename):
    wl, flx, time = load_spectrum(filename)
    velocities = []
    verrs = []
    for lstr, loc in lines.items():
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.title(f"Fit for Line {lstr} @ {round(loc)}Å")
        sucess, errs, [scaling, gamma, shift, slope, height, eta] = plot_peak_region(wl, flx, loc, 100, True, time)
        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
            velocities.append(v_from_doppler(shift, loc))
            verrs.append(v_from_doppler_err(shift, loc, errs[2], 0))
    velocities = np.array(velocities)
    verrs = np.array(verrs)
    outloc = check_for_outliers(velocities)
    if np.invert(outloc).sum() != 0:
        print(f"! DETECTED OUTLIER CANDIDATE (DEVIATION > {OUTLIER_MAX_SIGMA}σ), REMOVE OUTLIER? [Y/N]")
        print(f"IN ARRAY: {velocities}")
        del_outlier = input()
        if del_outlier.lower() == "y":
            velocities = velocities[outloc]
            verrs = verrs[outloc]

    v_std = np.sqrt(np.std(velocities) ** 2 + np.sum(verrs ** 2))  # ?????
    # v_std = np.sqrt(np.sum(verrs ** 2)) ?????
    complete_v_shift = np.mean(velocities)
    print_single_spec_results(complete_v_shift, v_std, filename)
    return complete_v_shift, v_std, time


def open_spec_files(loc, ftype):
    flist = []
    if ftype == "dict":
        for file in os.listdir(loc):
            if file.endswith(".csv") or file.endswith(".fits"):
                flist.append(os.path.join(loc, file))
    elif ftype == "numeric":
        pstart, pend = loc.split(".")
        flist = glob.glob(pstart+"*"+pend)
    return flist


if __name__ == "__main__":
    files = open_spec_files(FILE_LOC, DAT_TYPE)
    spectimes = []
    specvels = []
    specverrs = []
    for file in files:
        complete_v_shift, v_std, time = single_spec_shift(file)
        spectimes.append(time.to_datetime())
        specvels.append(complete_v_shift)
        specverrs.append(v_std)
    specvels = np.array(specvels)/1000
    specverrs = np.array(specverrs)/1000
    plt.title("Radial Velocity over Time")
    plt.ylabel("Radial Velocity [km/s]")
    plt.xlabel("Date")
    plt.plot_date(spectimes, specvels, xdate=True)
    plt.errorbar(spectimes, specvels, yerr=specverrs)
    plt.show()
