# TODO: First fit all lines individually, then together
# TODO: Print pdf of individual RVs
# TODO: Input list of 100-200 brightest sdV and sdOB Stars

import glob
import os
import re
from datetime import datetime
from pprint import pprint

import logging
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.constants import c
from astropy.io import fits
import astropy.time as atime

EXTENSION = ".txt"
CATALOGUE = "selected_objects.csv"
FILE_LOC = "spectra/"
DATA_TYPE = "numeric"  # dict, numeric
OUTLIER_MAX_SIGMA = 2
CUT_MARGIN = 20
SHOW_PLOTS = True
PLOTOVERVIEW = False
AUTO_REMOVE_OUTLIERS = True

lines = {
    "H_alpha": 6562.79,
    "H_beta": 4861.35,
    "H_gamma": 4340.472,
    "H_delta": 4101.734,
    # "H_epsilon": 3970.075,
    # "H_zeta": 3888.052,
    # "H_eta": 3835.387,
    "He_I_4026": 4026.19,
    "He_I_4472": 4471.4802,
    "He_I_4922": 4921.9313,
    "He_I_5016": 5015.678,
    "He_I_5876": 5875.6,
    "He_I_6678": 6678.15,
    # "He_II_4541": 4541.59,
    "He_II_4686": 4685.70,
    # "He_II_5412": 5411.52
}

disturbing_lines = {
    "H_alpha": 6562.79,
    "H_beta": 4861.35,
    "H_gamma": 4340.472,
    "H_delta": 4101.734,
    "H_epsilon": 3970.075,
    "H_zeta": 3888.052,
    "H_eta": 3835.387,
    "He_I_4026": 4026.19,
    "He_I_4472": 4471.4802,
    "He_I_4922": 4921.9313,
    "He_I_5016": 5015.678,
    "He_I_5876": 5875.6,
    "He_I_6678": 6678.15,
    "He_II_4541": 4541.59,
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
    return gamma / (2 * np.sqrt(2 * np.log(2)))


def height_err(eta, sigma, gamma, scaling, u_eta, u_sigma, u_gamma, u_scaling):
    return u_scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (gamma * np.pi)) + \
           u_sigma * eta * scaling / (sigma ** 2 * np.sqrt(2 * np.pi)) + \
           u_eta * scaling * (1 / (sigma * np.sqrt(2 * np.pi)) - 2 / (gamma * np.pi)) + \
           u_gamma * scaling * (1 - eta) * 2 / (gamma ** 2 * np.pi)


def voigt(x, scaling, gamma, shift, slope, height):
    sigma = to_sigma(gamma)
    z = (x + 1.j * gamma) / (sigma * np.sqrt(2))
    return -scaling * (np.real(faddeeva(z)) / (sigma * np.sqrt(2 * np.pi))) + slope * (x - shift) + height


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    return -scaling * (eta * gaussian(x, gamma, shift) + (1 - eta) * lorentzian(x, gamma, shift)) + slope * x + height


def load_spectrum(filename, filetype="noncoadded_txt"):
    """
    :param filename: Spectrum File location
    :param filetype: Type of spectrum file
                "simple_csv" : Simple .csv file with wavelength in the first, and flux in the second column, seperated by commas
                "coadded_fits" : .fits spectrum file as one would get via Vizier query
                "noncoadded_txt" : Simple .txt file with wavelength in the first, flux in the second and flux error in the third column, separated by spaces

    :return: Spectrum Wavelengths, Corresponding flux, time of observation, flux error (if available)
    """
    if filetype == "noncoadded_txt":
        wavelength = []
        flux = []
        flux_std = []
        with open(filename) as dsource:
            for row in dsource:
                if "#" not in row:
                    wl, flx, flx_std = row.split(" ")
                    wavelength.append(float(wl))
                    flux.append(float(flx))
                    flux_std.append(float(flx_std))
        wavelength = np.array(wavelength)
        flux = np.array(flux)
        flux_std = np.array(flux_std)
        filename_prefix, nspec = filename.split("_")
        nspec = nspec.replace(".txt", "")
        nspec = int(nspec)
        with open(filename.split("_")[0] + "_mjd.txt") as dateinfo:
            n = 0
            for d in dateinfo:
                if "#" not in d:
                    n += 1
                    if nspec == n:
                        mjd = float(d)
                        t = atime.Time(mjd, format="mjd")

    elif filetype == "simple_csv":
        data = pd.read_csv(filename)
        data = data.to_numpy()

        wavelength = data[:, 0]
        flux = data[:, 1]
    elif filetype == "coadded_fits":
        hdul = fits.open(filename)
        data = hdul[1].data
        # pprint(hdul[0].header)
        try:
            tai = hdul[0].header["TAI"]
            t = atime.Time(tai + atime.Time(datetime.strptime("17/11/1858", '%d/%m/%Y')).to_value(format="unix_tai"),
                           format="unix_tai")
        except KeyError:
            warnings.warn("Could not get TAI timestamp, trying MJD...", NoiseWarning)
            mjd = hdul[0].header["MJD"]
            t = atime.Time(mjd, format="mjd")
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
    if "flux_std" not in vars():
        flux_std = np.zeros(np.shape(flux))
    return wavelength, flux, t, flux_std


def verify_peak(wavelength, sigma, params, errs):
    """
    :param wavelength: Wavelength array
    :param sigma: Standard deviation of Fit Function
    :param params: Fit parameters
    :param errs: Fit errors
    :return: Returns False if the Peak width is in the order of one step in the Wavelength array
    """
    d_wl = wavelength[1] - wavelength[0]
    if sigma < d_wl:
        return False
    else:
        errs = np.array(errs)
        errs[errs == 0] = 1e-10
        p_over_err = np.abs(np.array(params) / errs)[:-1]
        if np.sum(p_over_err < 2) > 0:
            return False
        return True


def calc_SNR(params, flux, wavelength, margin, sanitized=False):
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

    if not sanitized:
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
    else:
        noisestrength = np.mean(np.square(np.array(flux[:loind].tolist() + flux[upind:].tolist())))

    SNR = 10 * np.log10(signalstrength / noisestrength)

    return signalstrength, noisestrength, SNR


def sanitise_flux(flux, wavelength_pov, wls):
    mask = np.full(np.shape(wls), True)
    for line in disturbing_lines.values():
        if line != wavelength_pov:
            _, loind, upind = slicearr(wls, line - CUT_MARGIN, line + CUT_MARGIN)
            mask[loind:upind] = False
    clean_flux = np.ma.MaskedArray(flux, ~mask)
    cut_flux = np.ma.MaskedArray(flux, mask)
    return clean_flux, cut_flux, mask


def cosmic_ray(slicedwl, flux, params, errs):
    if np.sum(np.abs(np.array((params[3], params[4]))/np.array((errs[3], errs[4]))) > 2) == 2:
        scaling, gamma, shift, slope, height, eta = params
        flux = flux - slope * slicedwl - height
        m = np.mean(flux)
        std = np.std(flux)
        sigma = to_sigma(gamma)
        h = scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))
    else:
        return False, []
    diffarr = np.diff(flux, prepend=flux[0])
    m_diffarr = np.mean(diffarr)
    std_diffarr = np.std(diffarr)
    if h > 2*std:
        return True, np.where(np.logical_and(flux > m + 2 * h, diffarr > m_diffarr + 2 * std_diffarr))
    else:
        return True, np.where(np.logical_and(flux > m + 2 * std, diffarr > m_diffarr + 2 * std_diffarr))


def plot_peak_region(wavelength, flux, flux_std, center, margin, fit, time, sanitize=False):
    for key, val in lines.items():
        if round(val) == round(center):
            lstr = key
    if "lstr" not in locals():
        lstr = "unknown"
    plt.title(f"Fit for Line {lstr} @ {round(center)}Å")
    if sanitize:
        flux, cut_flux, mask = sanitise_flux(flux, center, wavelength)
    slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)
    plt.plot(slicedwl, flux[loind:upind])
    sucess = True

    if sanitize:
        plt.plot(slicedwl, cut_flux[loind:upind], color="lightgrey", label='_nolegend_')
        wavelength = wavelength[mask]
        flux = flux.compressed()
        flux_std = flux_std[mask]
        slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

    flux_std[flux_std == 0] = np.mean(flux_std)

    if fit:
        initial_h = np.mean(flux[loind:upind])
        initial_s = np.abs((initial_h - np.min(flux[
                                               round(loind + abs(upind - loind) / 2 - abs(upind - loind) / 20):round(
                                                   loind + abs(upind - loind) / 2 + abs(
                                                       upind - loind) / 20)])) / 0.71725216658522349)
        if initial_s == np.nan:
            initial_s = 1

        initial_params = [initial_s, 5, center, 0, initial_h, 0.5]

        try:
            params, errs = curve_fit(pseudo_voigt,
                                     slicedwl,
                                     flux[loind:upind],
                                     initial_params,
                                     # scaling, gamma, shift, slope, height, eta
                                     bounds=(
                                         [0, 0, center - 15, -np.inf, 0, 0],
                                         [np.inf, np.sqrt(2 * np.log(2)) * margin / 4, center + 15, np.inf, np.inf, 1]
                                     ),
                                     sigma=flux_std[loind:upind]
                                     )

            errs = np.sqrt(np.diag(errs))

            cr, cr_ind = cosmic_ray(slicedwl, flux[loind:upind], params, errs)
            if cr and np.sum(cr_ind) > 0:
                cr_ind += loind
                plt.cla()
                for i in cr_ind[0]:
                    plt.plot(wavelength[i-1:i+1], flux[i-1:i+1], color="lightgray", label='_nolegend_')
                return plot_peak_region(np.delete(wavelength, cr_ind), np.delete(flux, cr_ind),
                                        np.delete(flux_std, cr_ind), center, margin, fit, time)


            # plt.plot(slicedwl, pseudo_voigt(slicedwl, initial_s, 1, 1, center, 0, initial_h, 0.5))
            if not verify_peak(wavelength, params[1] / (2 * np.sqrt(2 * np.log(2))), params, errs):
                if sucess:
                    warn_text = plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                                            horizontalalignment='right',
                                            verticalalignment='bottom',
                                            color="red")
                    if not sanitize:
                        warn_text.set_visible(False)
                        plt.cla()
                        return plot_peak_region(wavelength, flux, flux_std, center, margin, fit, time, sanitize=True)
                warnings.warn(f"WL {center}Å: Peak seems to be fitted to Noise, SNR and peak may not be accurate.",
                              NoiseWarning)
                sucess = False
            sstr, nstr, SNR = calc_SNR(params, flux, wavelength, margin, sanitize)
            plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}dB ", (10, 10), xycoords="figure pixels")
            if SNR < 2:
                if sucess:
                    warn_text = plt.figtext(0.3, 0.95, f"BAD SIGNAL!",
                                            horizontalalignment='right',
                                            verticalalignment='bottom',
                                            color="red")
                    if not sanitize:
                        warn_text.set_visible(False)
                        plt.cla()
                        return plot_peak_region(wavelength, flux, flux_std, center, margin, fit, time, sanitize=True)
                warnings.warn(f"WL {center}Å: Signal-to-Noise ratio out of bounds!",
                              NoiseWarning)
                sucess = False

            plt.plot(slicedwl, pseudo_voigt(slicedwl, *params), zorder=5)
        except RuntimeError:
            sucess = False
            warnings.warn("Could not find a good Fit!", FitUnsuccessfulWarning)
            warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    color="red")
            if not sanitize:
                warn_text.set_visible(False)
                plt.cla()
                return plot_peak_region(wavelength, flux, flux_std, center, margin, fit, time, sanitize=True)
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
        except ValueError as e:
            print("No peak found:", e)
            warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    color="red")
            if not sanitize:
                warn_text.set_visible(False)
                plt.cla()
                return plot_peak_region(wavelength, flux, flux_std, center, margin, fit, time, sanitize=True)
            sucess = False
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
    plt.axvline(center, linewidth=0.5, color='lightgrey', linestyle='dashed', zorder=1)
    plt.legend(["Flux", "Best Fit"])
    outtimestr = time.to_datetime().strftime("%m_%d_%Y__%H_%M_%S")
    if not os.path.isdir(f'output/spec_{outtimestr}/'):
        os.mkdir(f"output/spec_{outtimestr}/")
    plt.savefig(f"output/spec_{outtimestr}/plot_{round(center)}Å", dpi=500)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.cla()
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
        print(f"\nPeak Height I={scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))}"
              f"±{height_err(eta, sigma, gamma, scaling, errs[5], sigma_err, errs[1], errs[0])}")
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
    print(f"Velocity: [{round(complete_v_shift / 1000, 2)}±{round(v_std / 1000, 2)}]km/s")
    print("#############################################################\n\n")


def check_for_outliers(array):
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    outlierloc = distance_from_mean < OUTLIER_MAX_SIGMA * standard_deviation
    print(distance_from_mean, OUTLIER_MAX_SIGMA * standard_deviation)
    return np.array(outlierloc)


def single_spec_shift(filename):
    wl, flx, time, flx_std = load_spectrum(filename)
    velocities = []
    verrs = []
    for lstr, loc in lines.items():
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        # plt.title(f"Fit for Line {lstr} @ {round(loc)}Å")
        sucess, errs, [scaling, gamma, shift, slope, height, eta] = plot_peak_region(wl, flx, flx_std, loc, 100, True,
                                                                                     time)
        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
            velocities.append(v_from_doppler(shift, loc))
            verrs.append(v_from_doppler_err(shift, loc, errs[2], 0))
    velocities = np.array(velocities)
    verrs = np.array(verrs)
    outloc = check_for_outliers(velocities)
    if np.invert(outloc).sum() != 0:
        if not AUTO_REMOVE_OUTLIERS:
            print(f"! DETECTED OUTLIER CANDIDATE (DEVIATION > {OUTLIER_MAX_SIGMA}σ), REMOVE OUTLIER? [Y/N]")
            print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = input()
        else:
            print(f"! DETECTED OUTLIER (DEVIATION > {OUTLIER_MAX_SIGMA}σ)")
            print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = "y"

        if del_outlier.lower() == "y":
            velocities = velocities[outloc]
            verrs = verrs[outloc]

    v_std = np.sqrt(np.sum(verrs ** 2))
    complete_v_shift = np.mean(velocities)
    print_single_spec_results(complete_v_shift, v_std, filename)
    return complete_v_shift, v_std, time


def open_spec_files(loc, fpre, end=".txt"):
    """
    :param loc: Directory of spectrum files
    :param fpre: file prefix
    :param end: file extension
    :return: list of spectrumfiles
    """
    flist = []
    if "." not in end:
        end = "." + end
    for fname in os.listdir(loc):
        if re.match(fpre + r"_[0-9]+" + end, fname):
            flist.append(loc + fname)
    return flist


def files_from_catalogue(cat):
    catalogue = pd.read_csv(cat)
    return [a.split(".")[0] for a in catalogue["file"]]


if __name__ == "__main__":
    for file_prefix in files_from_catalogue(CATALOGUE):
        fileset = open_spec_files(FILE_LOC, file_prefix, end=EXTENSION)
        spectimes = []
        specvels = []
        specverrs = []
        for file in fileset:
            complete_v_shift, v_std, time = single_spec_shift(file)
            spectimes.append(time.to_datetime())
            specvels.append(complete_v_shift)
            specverrs.append(v_std)
        specvels = np.array(specvels) / 1000
        specverrs = np.array(specverrs) / 1000
        if not SHOW_PLOTS:
            plt.cla()
        plt.title("Radial Velocity over Time")
        plt.ylabel("Radial Velocity [km/s]")
        plt.xlabel("Date")
        plt.plot_date(spectimes, specvels, xdate=True)
        plt.errorbar(spectimes, specvels, yerr=specverrs, capsize=3)
        plt.show()
