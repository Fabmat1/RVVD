# TODO: First fit all lines individually, then together
# TODO: Print pdf of individual RVs
# TODO: Input list of 100-200 brightest sdV and sdOB Stars

import glob
import os
import re
from datetime import datetime
from pprint import pprint
from lmfit import Parameters, minimize, report_fit
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
MARGIN = 100
SHOW_PLOTS = False
PLOTOVERVIEW = False
AUTO_REMOVE_OUTLIERS = True
SAVE_SINGLE_IMGS = True
SAVE_COMPOSITE_IMG = True
COSMIC_RAY_DETECTION_LIM = (2, 1)   # minimum times peak height/flux std required to detect cr, minimum times diff
# std required to detect cr

output_table_cols = pd.DataFrame({
    "subspectrum": [],
    "line_name": [],
    "line_loc": [],
    "height": [],
    "u_heigth": [],
    "reduction_factor": [],
    "u_reduction_factor": [],
    "lambda_0": [],
    "u_lambda_0": [],
    "eta": [],
    "u_eta": [],
    "sigma": [],
    "u_sigma": [],
    "gamma": [],
    "u_gamma": [],
    "scaling": [],
    "u_scaling": [],
    "flux_0": [],
    "u_flux_0": [],
    "slope": [],
    "u_slope": [],
    "RV": [],
    "u_RV": [],
    "signal_strength": [],
    "noise_strength": [],
    "SNR": [],
    "sanitized": [],
    "cr_ind": [],
}, dtype=object)

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
        loind = np.where(arr == arr[arr > lower][0])[0][0]
        upind = loind + 1
        newarr = np.empty([1])
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
    return np.sqrt((u_scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (gamma * np.pi))) ** 2 + \
                   (u_sigma * eta * scaling / (sigma ** 2 * np.sqrt(2 * np.pi))) ** 2 + \
                   (u_eta * scaling * (1 / (sigma * np.sqrt(2 * np.pi)) - 2 / (gamma * np.pi))) ** 2 + \
                   (u_gamma * scaling * (1 - eta) * 2 / (gamma ** 2 * np.pi)) ** 2)


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


def expand_mask(mask):
    """
    :param mask: Boolean mask of an array
    :return: Boolean mask, with
    """
    nmask = np.logical_and(np.logical_and(np.roll(mask, -1), np.roll(mask, 1)), mask)
    return nmask


def sanitise_flux(flux, wavelength_pov, wls):
    mask = np.full(np.shape(wls), True)
    for line in disturbing_lines.values():
        if line != wavelength_pov:
            _, loind, upind = slicearr(wls, line - CUT_MARGIN, line + CUT_MARGIN)
            mask[loind:upind] = False
    clean_flux = np.ma.MaskedArray(flux, ~mask)
    cut_flux = np.ma.MaskedArray(flux, expand_mask(mask))
    return clean_flux, cut_flux, mask


def cosmic_ray(slicedwl, flux, params, errs):
    if np.sum(np.abs(np.array((params[3], params[4])) / np.array((errs[3], errs[4]))) > 2) == 2:
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
    if h > 2 * std:
        return True, np.where(np.logical_and(flux > m + COSMIC_RAY_DETECTION_LIM[0] * h, diffarr > m_diffarr + COSMIC_RAY_DETECTION_LIM[1] * std_diffarr))
    else:
        return True, np.where(np.logical_and(flux > m + COSMIC_RAY_DETECTION_LIM[0] * std, diffarr > m_diffarr + COSMIC_RAY_DETECTION_LIM[1] * std_diffarr))


def plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=False, used_cr_inds=None):
    if used_cr_inds is None:
        used_cr_inds = []

    f_pre, subspec_ind = file_prefix.split("_")

    for key, val in lines.items():
        if round(val) == round(center):
            lstr = key
    if "lstr" not in locals():
        lstr = "unknown"
    plt.title(f"Fit for Line {lstr} @ {round(center)}Å")
    if sanitize:
        flux, cut_flux, mask = sanitise_flux(flux, center, wavelength)
    slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)
    plt.plot(slicedwl, flux[loind:upind], zorder=5)

    fluxmin = flux[loind:upind].min()
    fluxmax = flux[loind:upind].max()
    pltrange = fluxmax - fluxmin
    plt.ylim((fluxmin - 0.05 * pltrange, fluxmax + 0.05 * pltrange))

    sucess = True

    if sanitize:
        plt.plot(slicedwl, cut_flux[loind:upind], color="lightgrey", label='_nolegend_', zorder=1)
        wavelength = wavelength[mask]
        flux = flux.compressed()
        flux_std = flux_std[mask]
        slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

    flux_std[flux_std == 0] = np.mean(flux_std)

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
                plt.plot(wavelength[i - 1:i + 2], flux[i - 1:i + 2], color="lightgray", label='_nolegend_')
            return plot_peak_region(np.delete(wavelength, cr_ind), np.delete(flux, cr_ind),
                                    np.delete(flux_std, cr_ind), center, margin, file_prefix, used_cr_inds=cr_ind)

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
                    return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix,
                                            sanitize=True)
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
                    return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix,
                                            sanitize=True)
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
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=True)
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
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=True)
        sucess = False
        plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
    plt.axvline(center, linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
    plt.legend(["Flux", "Best Fit"])

    if not os.path.isdir(f'output/{f_pre}/'):
        os.mkdir(f"output/{f_pre}/")
    if not os.path.isdir(f'output/{f_pre}/{subspec_ind}'):
        os.mkdir(f"output/{f_pre}/{subspec_ind}")
    if SAVE_SINGLE_IMGS:
        plt.savefig(f"output/{f_pre}/{subspec_ind}/{round(center)}Å", dpi=500)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.cla()
        plt.clf()
    if sucess:
        return sucess, errs, params, [sstr, nstr, SNR], sanitize, used_cr_inds
    else:
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False],\
               [False, False, False], False, False


def pseudo_voigt_height(errs, scaling, eta, gamma, sigma=False, sigma_err=False):
    if not sigma:
        sigma = to_sigma(gamma)
    if not sigma_err:
        sigma_err = to_sigma(errs[1])
    heigth = scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))
    err = height_err(eta, sigma, gamma, scaling, errs[5], sigma_err, errs[1], errs[0])
    return heigth, err


def print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc):
    if sucess:
        sigma = to_sigma(gamma)
        sigma_err = to_sigma(errs[1])
        h, u_h = pseudo_voigt_height(errs, scaling, eta, gamma, sigma, sigma_err)
        print("######################## FIT RESULTS ########################")
        print(f"Result for line {lstr} @ {round(loc)}Å:")
        print(f"\nPeak Height I={h}±{u_h}")
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
    file_prefix = filename.split("/")[-1].split(".")[0]
    velocities = []
    verrs = []
    output_table = output_table_cols.copy()
    subspec = int(filename.split("/")[-1].split("_")[1].split(".")[0])

    for lstr, loc in lines.items():
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        # plt.title(f"Fit for Line {lstr} @ {round(loc)}Å")
        sucess, errs, [scaling, gamma, shift, slope, height, eta], [sstr, nstr, SNR], sanitized, cr_ind = plot_peak_region(wl, flx, flx_std, loc, MARGIN,
                                                                                     file_prefix)
        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
            rv = v_from_doppler(shift, loc)
            u_rv = v_from_doppler_err(shift, loc, errs[2], 0)
            velocities.append(rv)
            verrs.append(u_rv)

            h, u_h = pseudo_voigt_height(errs, scaling, eta, gamma)
            u_scaling, u_gamma, u_shift, u_slope, u_height, u_eta = errs

            output_table_row = pd.DataFrame({
                "subspectrum": [subspec],
                "line_name": [lstr],
                "line_loc": [loc],
                "height": [h],
                "u_heigth": [u_h],
                "reduction_factor": [shift/loc],
                "u_reduction_factor": [u_shift/loc],
                "lambda_0": [shift],
                "u_lambda_0": [u_shift],
                "eta": [eta],
                "u_eta": [u_eta],
                "sigma": [to_sigma(gamma)],
                "u_sigma": [to_sigma(u_gamma)],
                "gamma": [gamma],
                "u_gamma": [u_gamma],
                "scaling": [scaling],
                "u_scaling": [u_scaling],
                "flux_0": [height],
                "u_flux_0": [u_height],
                "slope": [slope],
                "u_slope": [u_slope],
                "RV": [rv],
                "u_RV": [u_rv],
                "signal_strength": [sstr],
                "noise_strength": [nstr],
                "SNR": [SNR],
                "sanitized": [sanitized],
                "cr_ind": [cr_ind],
            })

            output_table = pd.concat([output_table, output_table_row], axis=0)

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

    return complete_v_shift, v_std, time, file_prefix, output_table


def make_dataset(params, ind, wl, lines, curvetypes):
    scaling = params[f'scaling_{ind}']
    gamma = params[f'gamma_{ind}']
    shift = params[f'r_factor_{ind}']*list(lines)[ind]
    slope = params[f'slope_{ind}']
    height = params[f'height_{ind}']
    if curvetypes[ind] == "voigt":
        eta = params[f'eta_{ind}']
        return pseudo_voigt(wl, scaling, gamma, shift, slope, height, eta)
    elif curvetypes[ind] == "gauss":
        return height+slope*wl-scaling*gaussian(wl, gamma, shift)
    elif curvetypes[ind] == "lorentz":
        return height+slope*wl-scaling*lorentzian(wl, gamma, shift)


def outer_fit(params, wl_dataset, flux_dataset, lines, curvetypes):
    ndata = len(wl_dataset)
    resids = []

    for i in range(ndata):
        resids.append(flux_dataset[i] - make_dataset(params, i, wl_dataset[i], lines, curvetypes))

    return np.concatenate(resids)


def v_from_doppler_rel(r_factor):
    return c*(r_factor**2-1)/(r_factor**2+1)


def v_from_doppler_rel_err(r_factor):
    return 4*c*r_factor/(np.power((r_factor**2+1), 2))


def culumative_shift(output_table_spec, file):
    wl, flx, time, flx_std = load_spectrum(file)
    lines = output_table_spec["line_loc"]
    flux_sanitized = output_table_spec["sanitized"]
    cr_ind = output_table_spec["cr_ind"]
    wl_dataset = []
    flux_dataset = []
    flux_std_dataset = []
    output_table = output_table_cols.copy()

    for i, line in enumerate(lines):
        if not list(flux_sanitized)[i]:
            line_cr_inds = list(cr_ind)[i]
            if len(line_cr_inds) > 0:
                wldata = np.delete(wl, line_cr_inds)
                flx = np.delete(flx, line_cr_inds)
                flx_std = np.delete(flx_std, line_cr_inds)
            wldata, loind, upind = slicearr(wl, line - MARGIN, line + MARGIN)

            wl_dataset.append(wldata)
            flux_dataset.append(flx[loind:upind])
            flux_std_dataset.append(flx_std[loind:upind])

        else:
            flx, cut_flux, mask = sanitise_flux(flx, line, wl)
            wl = wl[mask]
            flx = flx.compressed()
            flx_std = flx_std[mask]
            wldata, loind, upind = slicearr(wl, line - MARGIN, line + MARGIN)
            wl_dataset.append(wldata)
            flux_dataset.append(flx[loind:upind])
            flux_std_dataset.append(flx_std[loind:upind])

    fit_params = Parameters()
    wl_inds = []
    curvetypes = []
    for iwl, wl in enumerate(wl_dataset):
        wl_inds.append(iwl)
        pred_scaling = list(output_table_spec["scaling"])[iwl]
        pred_gamma = list(output_table_spec["gamma"])[iwl]
        pred_r_factor = list(output_table_spec["reduction_factor"])[iwl]
        pred_slope = list(output_table_spec["slope"])[iwl]
        pred_height = list(output_table_spec["flux_0"])[iwl]
        pred_eta = list(output_table_spec["eta"])[iwl]
        # if .01 < pred_eta < .99:
        fit_params.add(f'eta_{iwl}', value=pred_eta, min=0, max=1)
        curvetypes.append("voigt")
        # elif pred_eta > .99:
        #     curvetypes.append("gauss")
        # elif pred_eta < .01:
        #     curvetypes.append("lorentz")
        fit_params.add(f'scaling_{iwl}', value=pred_scaling, min=0, max=np.inf)
        fit_params.add(f'gamma_{iwl}', value=pred_gamma, min=0, max=np.inf)
        fit_params.add(f'r_factor_{iwl}', value=pred_r_factor, min=0, max=np.inf)
        fit_params.add(f'slope_{iwl}', value=pred_slope, min=-np.inf, max=np.inf)
        fit_params.add(f'height_{iwl}', value=pred_height, min=0, max=np.inf)

    for i in range(len(wl_inds)-1):
        i += 1
        fit_params[f'r_factor_{i}'].expr = 'r_factor_0'

    # try:
    out = minimize(outer_fit, fit_params, args=(wl_dataset, flux_dataset, lines, curvetypes))
    report_fit(out.params)

    file_prefix, subspec_ind = file.split("/")[1].split(".")[0].split("_")

    culumv = v_from_doppler_rel(out.params["r_factor_0"])
    r_err_ind = out.var_names.index("r_factor_0")
    errs = np.sqrt(np.diag(out.covar))
    culumv_errs = v_from_doppler_rel_err(errs[r_err_ind])
    # except AttributeError:
    #     print("Fit failed, retrying...")
    #     retry_table = output_table_spec.copy()
    #     retry_table.reset_index(inplace=True, drop=True)
    #     for i, row in retry_table.iterrows():
    #         try:
    #             retry_table.loc[i, "eta"] = out.params[f"eta_{i}"].value
    #         except KeyError:
    #             pass
    #         retry_table.loc[i, "scaling"] = out.params[f"scaling_{i}"].value+0.1*out.params[f"scaling_{i}"].value
    #         retry_table.loc[i, "gamma"] = out.params[f"gamma_{i}"].value+0.1*out.params[f"gamma_{i}"].value
    #         retry_table.loc[i, "reduction_factor"] = out.params[f"r_factor_{i}"].value+0.1*out.params[f"r_factor_{i}"].value
    #         retry_table.loc[i, "slope"] = out.params[f"slope_{i}"].value+0.1*out.params[f"slope_{i}"].value
    #         retry_table.loc[i, "flux_0"] = out.params[f"height_{i}"].value+0.1*out.params[f"height_{i}"].value
    #     return culumative_shift(retry_table, file)

    for i in range(len(wl_inds)-1):
        lname = list(output_table_spec['line_name'])[i]
        lines = list(lines)
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.title(f"Culumative Fit for Line {lname} @ {round(lines[i])}Å")
        plt.axvline(lines[i], linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
        plt.plot(wl_dataset[i], flux_dataset[i])
        plt.plot(wl_dataset[i], make_dataset(out.params, i, wl_dataset[i], lines, curvetypes))
        if SAVE_SINGLE_IMGS:
            plt.savefig(f"output/{file_prefix}/{subspec_ind}/culum_{round(lines[i])}Å.png", dpi=200)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.clf()
            plt.cla()

        try:
            eta = out.params[f"eta_{i}"].value
            u_eta = [errs[out.var_names.index(f"eta_{i}")]]
        except KeyError:
            eta = "--"
            u_eta = "--"

        output_table_row = pd.DataFrame({
            "subspectrum": [list(output_table_spec["subspectrum"])[0]],
            "line_name": [lname],
            "line_loc": [lines[i]],
            "height": ["--"],
            "u_heigth": ["--"],
            "reduction_factor": [out.params[f"r_factor_0"].value],
            "u_reduction_factor": [errs[out.var_names.index(f"r_factor_0")]],
            "lambda_0": [lines[i]*out.params[f"r_factor_{i}"].value],
            "u_lambda_0": [lines[i]*errs[out.var_names.index(f"r_factor_0")]],
            "eta": [eta],
            "u_eta": [u_eta],
            "sigma": [to_sigma(out.params[f"gamma_{i}"].value)],
            "u_sigma": [to_sigma(errs[out.var_names.index(f"gamma_{i}")])],
            "gamma": [out.params[f"gamma_{i}"].value],
            "u_gamma": [errs[out.var_names.index(f"gamma_{i}")]],
            "scaling": [out.params[f"scaling_{i}"].value],
            "u_scaling": [errs[out.var_names.index(f"scaling_{i}")]],
            "flux_0": [out.params[f"height_{i}"].value],
            "u_flux_0": [errs[out.var_names.index(f"height_{i}")]],
            "slope": [out.params[f"slope_{i}"].value],
            "u_slope": [errs[out.var_names.index(f"slope_{i}")]],
            "RV": [culumv],
            "u_RV": [culumv_errs],
            "signal_strength": ["--"],
            "noise_strength": ["--"],
            "SNR": ["--"],
            "sanitized": ["--"],
            "cr_ind": ["--"]
        })

        output_table = pd.concat([output_table, output_table_row], axis=0)

    return culumv, culumv_errs, output_table


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
    global catalogue
    catalogue = pd.read_csv(cat)
    return [a.split(".")[0] for a in catalogue["file"]]


if __name__ == "__main__":
    file_prefixes = files_from_catalogue(CATALOGUE)
    for file_prefix in file_prefixes:
        fileset = open_spec_files(FILE_LOC, file_prefix, end=EXTENSION)
        spectimes = []
        spectimes_mjd = []
        specvels = []
        specverrs = []
        culumvs = []
        culumvs_errs =[]
        single_output_table = output_table_cols.copy()
        culumative_output_table = output_table_cols.copy()
        for file in fileset:
            complete_v_shift, v_std, time, file_prefix, output_table_spec = single_spec_shift(file)
            single_output_table = pd.concat([single_output_table, output_table_spec], axis=0)
            culumv, culumv_errs, output_table_spec = culumative_shift(output_table_spec, file)
            culumative_output_table = pd.concat([culumative_output_table, output_table_spec], axis=0)
            culumvs.append(culumv)
            culumvs_errs.append(culumv_errs)
            spectimes.append(time.to_datetime())
            spectimes_mjd.append(time.mjd)
            specvels.append(complete_v_shift)
            specverrs.append(v_std)

        if "catalogue" in vars():
            gaia_id = catalogue["source_id"][file_prefixes.index(file_prefix.split('_')[0])]
        else:
            gaia_id = file_prefix

        single_output_table.drop("sanitized", axis=1)
        single_output_table.to_csv(f"output/{file_prefix.split('_')[0]}/single_spec_vals.csv", index=False)
        culumative_output_table.drop("sanitized", axis=1)
        culumative_output_table.to_csv(f"output/{file_prefix.split('_')[0]}/culum_spec_vals.csv", index=False)

        specvels = np.array(specvels) / 1000
        specverrs = np.array(specverrs) / 1000
        culumvs = np.array(culumvs) / 1000
        culumvs_errs = np.array(culumvs_errs) / 1000
        rvtable = pd.DataFrame({
            "culum_fit_RV": culumvs,
            "u_culum_fit_RV": culumvs_errs,
             "single_fit_RV": specvels,
            "u_single_fit_RV": specverrs,
            "mjd": spectimes_mjd,
            "readable_time": [ts.strftime("%m/%d/%Y %H:%M:%S:%f") for ts in spectimes]
        })
        rvtable.to_csv(f"output/{file_prefix.split('_')[0]}/RV_variation.csv", index=False)
        if not SHOW_PLOTS:
            plt.cla()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        plt.title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}")
        plt.ylabel("Radial Velocity [km/s]")
        plt.xlabel("Date")
        plt.gcf().autofmt_xdate()
        plt.plot_date(spectimes, specvels, xdate=True, zorder=5)
        plt.plot_date(spectimes, culumvs, xdate=True, zorder=5)
        plt.errorbar(spectimes, specvels, yerr=specverrs, capsize=3, linestyle='', zorder=1)
        plt.errorbar(spectimes, culumvs, yerr=culumvs_errs, capsize=3, linestyle='', zorder=1)
        plt.tight_layout()
        plt.savefig(f"output/{file_prefix.split('_')[0]}/RV_variation.png", dpi=300)
        plt.show()
