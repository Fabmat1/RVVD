import glob
import os
import re
import warnings
from datetime import datetime
from multiprocessing import Pool

import astropy.time as atime
import img2pdf
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.constants import c, pi
from scipy.optimize import curve_fit
from scipy.special import erf

from analyse_results import result_analysis

############################## SETTINGS ##############################


### GENERAL EXECUTION SETTINGS

EXTENSION = ".txt"  # extension of the ASCII spectra files
SPECTRUM_FILE_SEPARATOR = " "  # Separator between columns in the ASCII file
USE_CATALOGUE = True  # whether only a subset of stars defined by a catalogue should be used
CATALOGUE = "all_objects.csv"  # the location of the catalogue
FILE_LOC = "spectra/"  # directory that holds the spectrum files
VERBOSE = False  # enable/disable verbose output
CHECK_FOR_DOUBLES = True  # check if there are any stars for which multiple spectra files exist (needs catalogue)
NO_NEGATIVE_FLUX = (True, 0.1)
"""
NO_NEGATIVE_FLUX: check for negative flux values (bool value at index 0) and filter spectra files with significant portions (Max allowed negative percentage at index 1)
"""
SUBDWARF_SPECIFIC_ADJUSTMENTS = True  # Apply some tweaks for the script to be optimized to hot subdwarfs

### FIT SETTINGS

OUTLIER_MAX_SIGMA = 3
"""
OUTLIER_MAX_SIGMA: Sigma value above which a line from the individual gets rejected as a fit to a wrong line.
Outliers do not get used in the cumulative fit.
"""
ALLOW_SINGLE_DATAPOINT_PEAKS = True  # Whether to accept lines that are made up by only one datapoint.
MAX_ERR = 100000  # Maximum allowed error above which a RV gets rejected as bad [m/s]
CUT_MARGIN = 20  # Margin used for cutting out disturbing lines, if their standard deviation was not yet determined [Å]
MARGIN = 100  # Window margin around lines used in determining fits [Å]
AUTO_REMOVE_OUTLIERS = True  # Whether an input from the user is required to remove outliers from being used in the cumulative fit 
MIN_ALLOWED_SNR = 5  # Minimum allowed SNR to include a line in the cumulative fit
SNR_PEAK_RANGE = 1.5  # Width of the peak that is considered the "signal" [Multiples of the FWHM]
COSMIC_RAY_DETECTION_LIM = 3  # minimum times peak height/flux std required to detect cr, minimum times diff
# std required to detect cr
USE_LINE_AVERAGES = False  # Determine guessed FWHM by examining previously fitted lines, not recommended when using multiprocessing!

### PLOT AND STORAGE SETTINGS

mpl.rcParams['figure.dpi'] = 300  # DPI value of plots that are created, if they are not pdf files
PLOT_FMT = ".pdf"  # File format of plots (.pdf is recommended due to smaller file sizes)
SHOW_PLOTS = False  # Show matplotlib plotting window for each plot
PLOTOVERVIEW = False  # Plot overview of entire subspectrum
SAVE_SINGLE_IMGS = False  # Save individual plots of fits as images in the respective folders !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!
REDO_IMAGES = False  # Redo images already present in folders
SAVE_COMPOSITE_IMG = True  # Save RV-Curve plot
REDO_STARS = False  # Whether to redo stars for which RVs have already be determined
PLOT_LABELS_FONT_SIZE = 14  # Label font size
PLOT_TITLE_FONT_SIZE = 17  # Title font size
CREATE_PDF = True  # Group all RV-plots into one big .pdf at the end of the calculations !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!
CREATE_RESULTTABLE = True  # Create pdf with result parameters (requires pdflatex)

# Lines that can potentially be used in fitting:
lines_to_fit = {
    "H_alpha": 6562.79,
    "H_beta": 4861.35,
    "H_gamma": 4340.472,
    "H_delta": 4101.734,
    # "H_epsilon": 3970.075,
    # "H_zeta": 3888.052,
    # "H_eta": 3835.387,
    # "He_I_4100": 4100.0,
    # "He_I_4339": 4338.7,
    # "He_I_4859": 4859.35,
    # "He_I_6560": 6560.15,
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

# Lines that need to potentially be cut out (also include all lines from above)
disturbing_lines = {
    "H_alpha": 6562.79,
    "H_beta": 4861.35,
    "H_gamma": 4340.472,
    "H_delta": 4101.734,
    "H_epsilon": 3970.075,
    "H_zeta": 3889.064,
    "H_eta": 3835.397,
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

line_FWHM_guesses = {'H_alpha': 13.893084511019556,  # FWHM guess for fitting for each line (only relevant if USE_LINE_AVERAGES is set to false)
                     'H_beta': 14.60607925873273,
                     'H_gamma': 13.98306118364695,
                     'H_delta': 14.16665277785858,
                     'He_I_4026': 5.823862835075121,
                     'He_I_4472': 5.1500543269762975,
                     'He_I_4922': 5.573968470249642,
                     'He_I_5016': 3.6450408054430654,
                     'He_I_5876': 4.513226333567151,
                     'He_I_6678': 5.4447656874775,
                     'He_II_4686': 6.364193877260614,
                     "He_II_4541": 5,
                     "He_II_5412": 5}

############################## FUNCTIONS ##############################

c = c
pi = pi
avg_line_fwhm = dict.fromkeys(lines_to_fit)
revlines = dict((v, k) for k, v in lines_to_fit.items())

output_table_cols = pd.DataFrame({
    "subspectrum": [],
    "line_name": [],
    "line_loc": [],
    "height": [],
    "u_height": [],
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

wl_splitinds, flux_splitinds, flux_std_splitinds = [0, 0, 0]
linelist = []

log_two = np.log(2)


class NoiseWarning(UserWarning):
    pass


class InaccurateDateWarning(UserWarning):
    pass


class FitUnsuccessfulWarning(UserWarning):
    pass


def splitname(name):
    allsplit = name.split("_")
    if len(allsplit) < 3:
        return allsplit
    return "_".join(allsplit[:-1]), allsplit[-1]


def slicearr(arr, lower, upper):
    """
    :param arr: array to be sliced into parts
    :param lower: lower bound-value in array
    :param upper: upper bound-value in array
    :return: sliced array, indices of subarray in old array
    """
    if lower > upper:
        loind = np.where(arr == arr[arr > lower][0])[0][0]
        upind = loind + 1
        newarr = np.array([])
        return newarr, loind, upind
    else:
        newarr = arr[np.logical_and(arr > lower, arr < upper)]
    if len(newarr) == 0:
        if len(arr[arr > lower]) != 0:
            loind = np.where(arr == arr[arr > lower][0])[0][0]
            upind = loind + 1
        else:
            loind = 0
            upind = 0
    else:
        loind, upind = np.where(arr == newarr[0])[0][0], np.where(arr == newarr[-1])[0][0] + 1
    return newarr, loind, upind


def faddeeva(z):
    return np.exp(-z ** 2) * (1 - erf(-1.j * z))


def lorentzian(x, gamma, x_0):
    return 1 / pi * (gamma / 2) / ((x - x_0) ** 2 + (gamma / 2) ** 2)


def gaussian(x, gamma, x_0):
    sigma = gamma / (2 * np.sqrt(2 * log_two))
    return 1 / (sigma * np.sqrt(2 * pi)) * np.exp((-(x - x_0) ** 2) / (2 * sigma ** 2))


def v_from_doppler(lambda_o, lambda_s):
    """
    :param lambda_o: Observed Wavelength
    :param lambda_s: Source Wavelength
    :return: Radial Velocity calculated from relativistic doppler effect
    """
    return c * (lambda_o ** 2 - lambda_s ** 2) / (lambda_o ** 2 + lambda_s ** 2)


def v_from_doppler_err(lambda_o, lambda_s, u_lambda_o):
    """
    :param lambda_o: Observed Wavelength
    :param lambda_s: Source Wavelength
    :param u_lambda_o: Uncertainty of Observed Wavelength
    :return: Uncertainty for Radial Velocity calculated from relativistic doppler effect
    """
    return c * ((4 * lambda_o * lambda_s ** 2) / ((lambda_o ** 2 + lambda_s ** 2) ** 2) * u_lambda_o)


def v_from_doppler_rel(r_factor):
    """
    :param r_factor: Wavelength reduction factor lambda_o/lambda_s
    :return: Radial Velocity calculated from relativistic doppler effect
    """
    return c * (r_factor ** 2 - 1) / (1 + r_factor ** 2)


def v_from_doppler_rel_err(r_factor, u_r_factor):
    """
    :param r_factor: Wavelength reduction factor lambda_o/lambda_s
    :param u_r_factor: Uncertainty for wavelength reduction factor
    :return: Uncertainty for Radial Velocity calculated from relativistic doppler effect
    """
    return 4 * c * r_factor / ((r_factor ** 2 + 1) ** 2) * u_r_factor


def to_sigma(gamma):
    return gamma / (2 * np.sqrt(2 * log_two))


def height_err(eta, gamma, scaling, u_eta, u_gamma, u_scaling):
    return np.sqrt((u_scaling * 2 / (np.pi * gamma) * (eta * (np.sqrt(log_two * np.pi) - 1) + 1)) ** 2 + \
                   (u_gamma * 2 * scaling / (np.pi * gamma ** 2) * (eta * (np.sqrt(log_two * np.pi) - 1) + 1)) ** 2 + \
                   (u_eta * 2 * scaling / (np.pi * gamma) * (np.sqrt(log_two * np.pi) - 1)) ** 2)


def voigt(x, scaling, gamma, shift, slope, height):
    sigma = to_sigma(gamma)
    z = (x + 1.j * gamma) / (sigma * np.sqrt(2))
    return -scaling * (np.real(faddeeva(z)) / (sigma * np.sqrt(2 * np.pi))) + slope * (x - shift) + height


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    g = gaussian(x, gamma, shift)
    l = lorentzian(x, gamma, shift)
    return -scaling * (eta * g + (1 - eta) * l) + slope * x + height


def load_spectrum(filename, filetype="noncoadded_txt", preserve_below_zero=False):
    """
    :param filename: Spectrum File location
    :param filetype: Type of spectrum file
                "simple_csv" : Simple .csv file with wavelength in the first, and flux in the second column, seperated by commas
                "coadded_fits" : .fits spectrum file as one would get via Vizier query
                "noncoadded_txt" : Simple .txt file with wavelength in the first, flux in the second and flux error in the third column, separated by spaces

    :return: Spectrum Wavelengths, Corresponding flux, time of observation, flux error (if available)
    """
    # Modify this to account for your specific needs!
    if filetype == "noncoadded_txt":
        data = np.loadtxt(filename, comments="#", delimiter=SPECTRUM_FILE_SEPARATOR)
        wavelength = data[:, 0]
        flux = data[:, 1]
        flux_std = data[:, 2]
        filename_prefix, nspec = splitname(filename)
        if NO_NEGATIVE_FLUX[0] and not preserve_below_zero:
            mask = flux > 0
            wavelength = wavelength[mask]
            flux_std = flux_std[mask]
            flux = flux[mask]
        nspec = nspec.replace(".txt", "")
        nspec = int(nspec)
        try:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#", delimiter=SPECTRUM_FILE_SEPARATOR)[nspec - 1], format="mjd")
        except IndexError:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#", delimiter=SPECTRUM_FILE_SEPARATOR), format="mjd")

    elif filetype == "simple_csv":
        data = pd.read_csv(filename)
        data = data.to_numpy()

        wavelength = data[:, 0]
        flux = data[:, 1]
    elif filetype == "coadded_fits":
        hdul = fits.open(filename)
        data = hdul[1].data
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
        plt.plot(wavelength, flux, color="navy")
        if SHOW_PLOTS:
            plt.show()
    if "flux_std" not in vars():
        flux_std = np.zeros(np.shape(flux))
    return wavelength, flux, t, flux_std


def calc_SNR(params, flux, wavelength, margin):
    """
    :param params: parameters of Fit
    :param flux: Flux array
    :param wavelength: Wavelength array
    :param margin: index width of plotted area (arbitrary)
    :return:    Mean squared displacement(MSD) of signal area,
                MSD of noise background,
                Signal-to-Noise ratio
    """
    scaling, gamma, shift, slope, height, eta = params
    flux = flux - slope * wavelength - height

    slicedwl, loind, upind = slicearr(wavelength, shift - SNR_PEAK_RANGE * gamma, shift + SNR_PEAK_RANGE * gamma)
    signalstrength = np.mean(np.square(flux[loind:upind]))

    if upind == loind + 1 and not ALLOW_SINGLE_DATAPOINT_PEAKS:
        warnings.warn("Peak is only a single datapoint, Fit rejected.", FitUnsuccessfulWarning)
        return 0, 1, 0
    elif upind == loind + 1 and ALLOW_SINGLE_DATAPOINT_PEAKS:
        # Maybe resample here
        pass

    if 2 * SNR_PEAK_RANGE * gamma < margin:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - margin, shift - SNR_PEAK_RANGE * gamma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + SNR_PEAK_RANGE * gamma, shift + margin)
    else:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - margin, shift - gamma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + gamma, shift + margin)
        warnings.warn("Sigma very large, Fit seems improbable!", NoiseWarning)
        if SAVE_SINGLE_IMGS:
            plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color="red")
    noisestrength = np.mean(np.square(np.array(flux[lloind:lupind].tolist() + flux[uloind:uupind].tolist())))

    SNR = signalstrength / noisestrength

    if np.isnan(SNR):
        signalstrength = 0
        noisestrength = 1
        SNR = 0

    return signalstrength, noisestrength, SNR


def expand_mask(mask):
    """
    :param mask: Boolean mask of an array
    :return: Boolean mask, with False boolean values "expanded" by one
    """
    nmask = np.logical_and(np.logical_and(np.roll(mask, -1), np.roll(mask, 1)), mask)
    return nmask


def sanitize_flux(flux, wavelength_pov, wls):
    """
    :param flux: flux array
    :param wavelength_pov: Wavelength of line that is being fitted
    :param wls: Wavelengths of all Lines that can be in the Spectrum
    :return: [masked array, masked array, array] Masked array of flux without
    disturbing lines, Masked array of flux that was cut out, mask of bools that was used
    """
    mask = np.full(np.shape(wls), True)
    for line in disturbing_lines.values():
        roundedline = round(line)
        try:
            if f"{roundedline}" in linewidths.keys():
                margin = linewidths[f"{roundedline}"]
            else:
                margin = CUT_MARGIN
        except NameError:
            margin = CUT_MARGIN
        if not roundedline - 2 < round(wavelength_pov) < roundedline + 2:
            lowerbound = line - margin
            upperbound = line + margin
            if round(wavelength_pov) - 25 < upperbound < round(wavelength_pov) + 25:
                upperbound = round(wavelength_pov) - 25
            if round(wavelength_pov) - 25 < lowerbound < round(wavelength_pov) + 25:
                lowerbound = round(wavelength_pov) + 25
            _, loind, upind = slicearr(wls, lowerbound, upperbound)
            mask[loind:upind] = False
    clean_flux = np.ma.MaskedArray(flux, ~mask)
    cut_flux = np.ma.MaskedArray(flux, expand_mask(mask))
    return clean_flux, cut_flux, mask


def cosmic_ray(slicedwl, flux, params, wl_pov, predetermined_crs=np.array([])):
    """
    :param slicedwl: Wavelenght array for which cosmic ray locations should be found
    :param flux: Corresponding flux
    :param params: Fit parameters
    :param wl_pov: Wavelength of the line for which closeby cosmics are determined
    :param predetermined_crs: Cosmic rays that were detemined in a previous iteration of the function
    :return: [bool, array] Whether any cosmic rays could be found, Their index locations in modified_slicedwl
    """

    if len(predetermined_crs) != 0:
        mask_array = np.zeros(np.shape(flux), dtype=bool)
        mask_array[predetermined_crs] = True
        normalized_flux = np.ma.MaskedArray(flux, mask_array)
        modified_slicedwl = np.ma.MaskedArray(slicedwl, mask_array)
    else:
        normalized_flux = flux
        modified_slicedwl = slicedwl

    scaling, gamma, shift, slope, height, eta = params
    sigma = to_sigma(gamma)
    normalized_flux = normalized_flux - slope * modified_slicedwl - height
    _, _, mask = sanitize_flux(flux, wl_pov, slicedwl)
    normalized_flux[~mask] = np.ma.masked

    lwl_for_std, lloind, lupind = slicearr(modified_slicedwl, shift - MARGIN, shift - 2 * sigma)
    uwl_for_std, uloind, uupind = slicearr(modified_slicedwl, shift + 2 * sigma, shift + MARGIN)

    if type(normalized_flux) == np.ma.core.MaskedArray:
        for_std = np.concatenate(
            [normalized_flux[lloind:lupind].compressed(), normalized_flux[uloind:uupind].compressed()])
    else:
        for_std = np.concatenate([normalized_flux[lloind:lupind], normalized_flux[uloind:uupind]])
    std = np.std(for_std)

    if len(predetermined_crs) == 0:
        initial_crs = np.where(normalized_flux > COSMIC_RAY_DETECTION_LIM * std)[0]
    else:
        allowed_inds = np.concatenate((predetermined_crs + 1, predetermined_crs - 1))
        initial_crs = np.where(normalized_flux > COSMIC_RAY_DETECTION_LIM * std)[0]
        mask = np.isin(initial_crs, allowed_inds)
        initial_crs = initial_crs[mask]

    if len(initial_crs) > 0:
        if len(predetermined_crs) == 0:
            return cosmic_ray(slicedwl, flux, params, wl_pov, predetermined_crs=initial_crs)
        else:
            return True, np.sort(np.concatenate([initial_crs, predetermined_crs]))
    else:
        if len(predetermined_crs) == 0:
            return False, []
        else:
            return True, predetermined_crs


def peak_amp_from_height(h, gamma, eta):
    return h / (np.pi * gamma / (2 * (1 + (np.sqrt(np.pi * log_two) - 1) * eta)))


def plot_peak_region(wavelengthdata, fluxdata, flux_stddata, center, margin, file_prefix, sanitize=False,
                     used_cr_inds=[], reset_initial_param=False):
    wavelength = np.copy(wavelengthdata)
    flux = np.copy(fluxdata)
    flux_std = np.copy(flux_stddata)

    coverage = [len(np.where(np.logical_and(wavelength > center - 2, wavelength < center + 2))[0]) == 0, len(np.where(np.logical_and(wavelength > center - margin - 2, wavelength < center - margin + 2))[0]) == 0,
                len(np.where(np.logical_and(wavelength > center + margin - 2, wavelength < center + margin + 2))[0]) == 0]

    if sum(coverage) != 0:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False

    for i in disturbing_lines.values():
        if i != center:
            if i - CUT_MARGIN < center + MARGIN or i + CUT_MARGIN > center - MARGIN:
                sanitize = True

    if used_cr_inds is None:
        used_cr_inds = []

    f_pre, subspec_ind = splitname(file_prefix)

    if sanitize:
        flux, cut_flux, mask = sanitize_flux(flux, center, wavelength)
    slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)
    try:
        assert len(slicedwl) > 0
    except AssertionError:
        warnings.warn("Seems like the line(s) you want to look at are not in the spectrum " + file_prefix + " Line @ " + str(center) + " Check your files!", FitUnsuccessfulWarning)
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False

    for key, val in lines_to_fit.items():
        if round(val) == round(center):
            lstr = key
    if "lstr" not in locals():
        lstr = "unknown"

    if SAVE_SINGLE_IMGS:
        plt.title(f"Fit for Line {lstr} @ {round(center)}Å")
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.plot(slicedwl, flux[loind:upind], zorder=5)

    sucess = True

    if sanitize:
        if SAVE_SINGLE_IMGS:
            plt.plot(slicedwl, cut_flux[loind:upind], color="lightgrey", label='_nolegend_', zorder=1)
        wavelength = wavelength[mask]
        flux = flux.compressed()
        flux_std = flux_std[mask]
        slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

    flux_std[flux_std == 0] = np.mean(flux_std)

    try:
        initial_slope = (np.mean(flux[loind:upind][-round(len(flux[loind:upind]) / 5):]) - np.mean(flux[loind:upind][:round(len(flux[loind:upind]) / 5)])) / (slicedwl[-1] - slicedwl[0])
        initial_h = np.mean(flux[loind:upind][:round(len(flux[loind:upind]) / 5)]) - np.mean(slicedwl[:round(len(flux[loind:upind]) / 5)]) * initial_slope
    except IndexError:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False

    try:
        flx_for_initial = flux[loind:upind] - slicedwl * initial_slope + initial_h
    except ValueError:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False

    if lstr != "unknown" and not reset_initial_param:
        if USE_LINE_AVERAGES:
            fwhmavg = avg_line_fwhm[lstr]
        else:
            fwhmavg = (line_FWHM_guesses[lstr], 100)
        if fwhmavg is not None:
            if fwhmavg[1] >= 10:
                initial_g = fwhmavg[0]
            else:
                initial_g = 5
        else:
            initial_g = 5
    else:
        initial_g = 5

    initial_s = np.ptp(flx_for_initial) * np.pi * initial_g / ((np.sqrt(np.log(2 * np.pi)) - 1) + 2)

    if initial_s == np.nan:
        initial_s = 1

    # scaling, gamma, shift, slope, height, eta
    initial_params = [initial_s, initial_g, center, initial_slope, initial_h, 0.5]

    bounds = (
        [0, 0, center - MARGIN * 0.25, -np.inf, -np.inf, 0],
        [np.inf, margin / 2, center + MARGIN * 0.25, np.inf, np.inf, 1]
    )

    try:
        params, errs = curve_fit(pseudo_voigt,
                                 slicedwl,
                                 flux[loind:upind],
                                 initial_params,
                                 # scaling, gamma, shift, slope, height, eta
                                 bounds=bounds,
                                 sigma=flux_std[loind:upind]
                                 )

        errs = np.sqrt(np.diag(errs))

        if len(used_cr_inds) == 0:
            cr, cr_ind = cosmic_ray(slicedwl, flux[loind:upind], params, center)
            if cr and np.sum(cr_ind) > 0:
                cr_ind += loind
                cr_true_inds = wavelengthdata.searchsorted(wavelength[cr_ind])
                plt.close()
                if SAVE_SINGLE_IMGS:
                    for i in cr_ind:
                        plt.plot(wavelength[i - 1:i + 2], flux[i - 1:i + 2], color="lightgray", label='_nolegend_')
                return plot_peak_region(np.delete(wavelength, cr_ind), np.delete(flux, cr_ind),
                                        np.delete(flux_std, cr_ind), center, margin, file_prefix,
                                        used_cr_inds=cr_true_inds)
        sstr, nstr, SNR = calc_SNR(params, flux, wavelength, margin)
        if SAVE_SINGLE_IMGS:
            plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}", (10, 10), xycoords="figure pixels")
        if SNR < MIN_ALLOWED_SNR:
            if sucess:
                if SAVE_SINGLE_IMGS:
                    warn_text = plt.figtext(0.3, 0.95, f"BAD SIGNAL!",
                                            horizontalalignment='right',
                                            verticalalignment='bottom',
                                            color="red")
                if not sanitize:
                    if SAVE_SINGLE_IMGS:
                        warn_text.set_visible(False)
                        plt.close()
                    return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix,
                                            sanitize=True)
            warnings.warn(f"WL {center}Å: Signal-to-Noise ratio out of bounds!",
                          NoiseWarning)
            sucess = False

        # Maybe do resampling here someday
        height, u_height = pseudo_voigt_height(errs, params[0], params[5], params[1])
        if height > 1.5 * np.ptp(flux[loind:upind]) and sucess:
            plt.close()
            sucess = False
        if height < u_height and sucess:
            plt.close()
            sucess = False
        #     return plot_peak_region(wavelengthdata, fluxdata, flux_stddata, center, margin, file_prefix, sanitize, used_cr_inds)

        if SAVE_SINGLE_IMGS:
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *params), zorder=5)
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=6, color="black")

    except RuntimeError:
        sucess = False
        warnings.warn("Could not find a good Fit!", FitUnsuccessfulWarning)

        if SAVE_SINGLE_IMGS:
            warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    color="red")
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
        if not sanitize:
            if SAVE_SINGLE_IMGS:
                warn_text.set_visible(False)
                plt.close()
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=True)
    except ValueError as e:
        print(initial_params)
        print(bounds)
        print("No peak found:", e)
        warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                color="red")
        if not sanitize:
            warn_text.set_visible(False)
            plt.close()
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=True)
        sucess = False
        if SAVE_SINGLE_IMGS:
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
    if SAVE_SINGLE_IMGS:
        plt.axvline(center, linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
        plt.legend(["Flux", "Best Fit"])

    if not os.path.isdir(f'output/{f_pre}/'):
        os.mkdir(f"output/{f_pre}/")
    if not os.path.isdir(f'output/{f_pre}/{subspec_ind}'):
        os.mkdir(f"output/{f_pre}/{subspec_ind}")
    if SAVE_SINGLE_IMGS:
        plt.savefig(f"output/{f_pre}/{subspec_ind}/{round(center)}Å{PLOT_FMT}")
        if SHOW_PLOTS:
            plt.show()
    plt.close()
    if sucess:
        return sucess, errs, params, [sstr, nstr, SNR], sanitize, used_cr_inds
    else:
        if not reset_initial_param:
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, sanitize=sanitize, reset_initial_param=True)
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False


def pseudo_voigt_height(errs, scaling, eta, gamma):
    height = 2 * scaling / (np.pi * gamma) * (1 + (np.sqrt(np.pi * log_two) - 1) * eta)
    if errs is not None:
        err = height_err(eta, gamma, scaling, errs[5], errs[1], errs[0])
        return height, err
    else:
        return height


def print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc):
    if VERBOSE:
        if sucess:
            sigma = to_sigma(gamma)
            sigma_err = to_sigma(errs[1])
            h, u_h = pseudo_voigt_height(errs, scaling, eta, gamma)
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
    if VERBOSE:
        print("\n\n##################### SPECTRUM RESULTS ######################")
        print(f"Result for Spectrum {os.path.basename(filename)}:")
        print(f"Velocity: [{round(complete_v_shift / 1000, 2)}±{round(v_std / 1000, 2)}]km/s")
        print("#############################################################\n\n")


def check_for_outliers(array):
    med = np.median(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - med)
    outlierloc = distance_from_mean < OUTLIER_MAX_SIGMA * standard_deviation
    return np.array(outlierloc)


def single_spec_shift(filepath, wl, flx, flx_std):
    fname = filepath.split("/")[-1].split(".")[0]
    velocities = []
    verrs = []
    output_table = output_table_cols.copy()
    file_prefix, subspec = splitname(fname)
    subspec = int(subspec)
    global linewidths
    linewidths = {}

    for lstr, loc in lines_to_fit.items():
        sucess, errs, [scaling, gamma, shift, slope, height, eta], [sstr, nstr,
                                                                    SNR], sanitized, cr_ind = plot_peak_region(wl, flx,
                                                                                                               flx_std,
                                                                                                               loc,
                                                                                                               MARGIN,
                                                                                                               fname)

        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
            linewidths[str(round(loc))] = 1.5 * gamma
            rv = v_from_doppler(shift, loc)
            u_rv = v_from_doppler_err(shift, loc, errs[2])
            velocities.append(rv)
            verrs.append(u_rv)

            h, u_h = pseudo_voigt_height(errs, scaling, eta, gamma)
            u_scaling, u_gamma, u_shift, u_slope, u_height, u_eta = errs

            output_table_row = pd.DataFrame({
                "subspectrum": [subspec],
                "line_name": [lstr],
                "line_loc": [loc],
                "height": [h],
                "u_height": [u_h],
                "reduction_factor": [shift / loc],
                "u_reduction_factor": [u_shift / loc],
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
    if np.invert(outloc).sum() != 0 and len(velocities) > 4:
        if not AUTO_REMOVE_OUTLIERS:
            print(f"! DETECTED OUTLIER CANDIDATE (DEVIATION > {OUTLIER_MAX_SIGMA}σ), REMOVE OUTLIER? [Y/N]")
            print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = input()
        else:
            if VERBOSE:
                print(f"! DETECTED OUTLIER (DEVIATION > {OUTLIER_MAX_SIGMA}σ)")
                print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = "y"

        if del_outlier.lower() == "y":
            velocities = velocities[outloc]
            verrs = verrs[outloc]
            output_table = output_table.loc[outloc, :]

    v_std = np.sqrt(np.sum(verrs ** 2))
    complete_v_shift = np.mean(velocities)
    print_single_spec_results(complete_v_shift, v_std, filepath)

    return complete_v_shift, v_std, file_prefix, output_table


def outer_fit(params, wl_dataset, flux_dataset, lines, curvetypes):
    ndata = len(wl_dataset)
    resids = []

    for i in range(ndata):
        resids.append(flux_dataset[i] - make_dataset(params, i, wl_dataset[i], lines, curvetypes))

    return np.concatenate(resids)


def make_dataset(wl, rfac, i, params):
    shift = rfac * linelist[i]
    scaling, gamma, slope, height, eta = params
    return pseudo_voigt(wl, scaling, gamma, shift, slope, height, eta)


def culum_fit_funciton(wl, r_factor, *args):
    wl_dataset = np.split(wl, wl_splitinds)
    resids = []

    params = []
    for a in args:
        params.append(a)

    params = np.array(params)
    params = np.split(params, len(params) / 5)

    for i, paramset in enumerate(params):
        resids.append(make_dataset(wl_dataset[i], r_factor, i, paramset))

    return np.concatenate(resids)


def cumulative_shift(output_table_spec, file, file_prefix, exclude_lines=None):
    if exclude_lines is None:
        exclude_lines = []
    global linelist
    wl, flx, time, flx_std = load_spectrum(file)
    linelist = output_table_spec["line_loc"]
    linelist = list(linelist)
    flux_sanitized = output_table_spec["sanitized"]
    cr_ind = output_table_spec["cr_ind"]
    subspec_ind = list(output_table_spec["subspectrum"])[0]
    lname = None

    linelist = [l for l in linelist if l not in exclude_lines]

    wl_dataset = []
    flux_dataset = []
    flux_std_dataset = []
    output_table = output_table_cols.copy()

    cr_inds = cr_ind.to_numpy()
    new_cr_inds = []
    for i in cr_inds:
        if type(i) != list:
            new_cr_inds.append(i)
    cr_inds = new_cr_inds
    if len(cr_inds) > 0:
        cr_inds = np.concatenate(cr_inds)
        wl = np.delete(wl, cr_inds)
        flx = np.delete(flx, cr_inds)
        flx_std = np.delete(flx_std, cr_inds)

    for i, line in enumerate(linelist):
        if not list(flux_sanitized)[i]:
            wldata, loind, upind = slicearr(wl, line - MARGIN, line + MARGIN)
            if len(wldata) > 1 and len(flx[loind:upind]) > 1:
                wl_dataset.append(wldata)
                flux_dataset.append(flx[loind:upind])
                flux_std_dataset.append(flx_std[loind:upind])
            else:
                linelist.remove(line)
        else:
            sanflx, cut_flux, mask = sanitize_flux(flx, line, wl)
            sanwl = wl[mask]
            sanflx = sanflx.compressed()
            sanflx_std = flx_std[mask]
            wldata, loind, upind = slicearr(sanwl, line - MARGIN, line + MARGIN)
            if len(wldata) > 1 and len(flx[loind:upind]) > 1:
                wl_dataset.append(wldata)
                flux_dataset.append(sanflx[loind:upind])
                flux_std_dataset.append(sanflx_std[loind:upind])
            else:
                linelist.remove(line)

    if len(wl_dataset) == 0:
        print("Sanitization failed!")
        return None, None, None, False

    global wl_splitinds, flux_splitinds, flux_std_splitinds
    wl_splitinds = [len(wldata) for wldata in wl_dataset]
    flux_splitinds = [len(flxdata) for flxdata in flux_dataset]
    flux_std_splitinds = [len(flxstddata) for flxstddata in flux_std_dataset]
    wl_splitinds = [sum(wl_splitinds[:ind + 1]) for ind, wl in enumerate(wl_splitinds)]
    flux_splitinds = [sum(flux_splitinds[:ind + 1]) for ind, flx in enumerate(flux_splitinds)]
    flux_std_splitinds = [sum(flux_std_splitinds[:ind + 1]) for ind, flxstd in enumerate(flux_std_splitinds)]

    rmean = output_table_spec["reduction_factor"].mean()

    p0 = [rmean]
    bounds = [
        [0],
        [np.inf]
    ]
    # scaling, gamma, slope, height, eta
    for i in range(len(wl_dataset)):
        specparamrow = output_table_spec.loc[output_table_spec["line_loc"] == list(linelist)[i]]
        try:
            p0 += [
                specparamrow["scaling"][0],
                specparamrow["gamma"][0],
                specparamrow["slope"][0],
                specparamrow["flux_0"][0],
                specparamrow["eta"][0],
            ]
            bounds[0] += [specparamrow["scaling"][0] / 2, specparamrow["gamma"][0] / 2, -np.inf, -np.inf, 0]
            bounds[1] += [specparamrow["scaling"][0] * 1.5, specparamrow["gamma"][0] * 1.5, np.inf, np.inf, 1]
        except KeyError:
            p0 += [
                specparamrow["scaling"],
                specparamrow["gamma"],
                specparamrow["slope"],
                specparamrow["flux_0"],
                specparamrow["eta"],
            ]
            bounds[0] += [specparamrow["scaling"] / 2, specparamrow["gamma"] / 2, -np.inf, -np.inf, 0]
            bounds[1] += [specparamrow["scaling"] * 1.5, specparamrow["gamma"] * 1.5, np.inf, np.inf, 1]

    wl_dataset = np.concatenate(wl_dataset, axis=0)
    flux_dataset = np.concatenate(flux_dataset, axis=0)
    flux_std_dataset = np.concatenate(flux_std_dataset, axis=0)

    flux_std_dataset[flux_std_dataset == 0] = np.mean(
        flux_std_dataset)  # Zeros in the std-dataset will raise exceptions (And are scientifically nonsensical)

    try:
        params, errs = curve_fit(
            culum_fit_funciton,
            wl_dataset,
            flux_dataset,
            p0=p0,
            bounds=bounds,
            sigma=flux_std_dataset,
            max_nfev=100000
        )
    except RuntimeError:
        print("Cumulative fit failed!")
        return None, None, None, False

    errs = np.sqrt(np.diag(errs))

    r_factor = params[0]
    r_factor_err = errs[0]

    culumv = v_from_doppler_rel(r_factor)
    culumv_errs = v_from_doppler_rel_err(r_factor, r_factor_err)

    errs = np.split(np.array(errs)[1:], len(np.array(errs)[1:]) / 5)
    params = np.split(np.array(params)[1:], len(np.array(params)[1:]) / 5)

    if np.abs(culumv_errs) > MAX_ERR or culumv_errs == 0.0:
        if len(linelist) > 1:
            u_heights = np.array([height_err(p[4], p[1], p[0], e[4], e[1], e[0]) for p, e in zip(params, errs)])
            exclude_lines.append(linelist[np.argmax(u_heights)])
            return cumulative_shift(output_table_spec, file, file_prefix, exclude_lines)
        else:
            print("Cumulative fit spurious!")
            return None, None, None, False

    wl_dataset = np.split(wl_dataset, wl_splitinds)
    flux_dataset = np.split(flux_dataset, flux_splitinds)

    for i, paramset in enumerate(params):
        lname = list(output_table_spec['line_name'])[i]
        lines = list(linelist)

        scaling, gamma, slope, height, eta = paramset
        shift = r_factor * lines[i]
        scalingerr, gammaerr, slopeerr, heighterr, etaerr = errs[i]
        shifterr = r_factor_err * lines[i]

        p_height = pseudo_voigt_height(None, scaling, eta, gamma)
        p_height_err = height_err(eta, gamma, scaling, etaerr, gammaerr, scalingerr)

        if SAVE_SINGLE_IMGS:
            plt.ylabel("Flux [ergs/s/cm^2/Å]")
            plt.xlabel("Wavelength [Å]")
            plt.title(f"cumulative Fit for Line {lname} @ {round(lines[i])}Å")
            plt.axvline(lines[i], linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
            plt.plot(wl_dataset[i], flux_dataset[i])
            wllinspace = np.linspace(wl_dataset[i][0], wl_dataset[i][-1], 1000)
            plt.plot(wllinspace, make_dataset(wllinspace, r_factor, i, paramset))
        if p_height < p_height_err or p_height > 1.5 * np.ptp(flux_dataset[i]):
            if SAVE_SINGLE_IMGS:
                plt.figtext(0.3, 0.95, f"FIT REJECTED!",
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            color="red")
                subspec_ind = str(subspec_ind) if len(str(subspec_ind)) != 1 else "0" + str(subspec_ind)
                plt.savefig(f"output/{file_prefix}/{subspec_ind}/culum_{round(lines[i])}Å{PLOT_FMT}")
            if len(linelist) > 1:
                exclude_lines.append(lines[i])
                return cumulative_shift(output_table_spec, file, file_prefix, exclude_lines)
            else:
                print("Cumulative shift rejected!")
                return None, None, None, False
        if SAVE_SINGLE_IMGS:
            subspec_ind = str(subspec_ind) if len(str(subspec_ind)) != 1 else "0" + str(subspec_ind)
            plt.savefig(f"output/{file_prefix}/{subspec_ind}/culum_{round(lines[i])}Å{PLOT_FMT}")
            if SHOW_PLOTS:
                plt.show()
        plt.close()

        if USE_LINE_AVERAGES:
            if avg_line_fwhm[lname] is not None:
                avg_line_fwhm[lname] = ((avg_line_fwhm[lname][0] * avg_line_fwhm[lname][1] + gamma) / (avg_line_fwhm[lname][1] + 1), avg_line_fwhm[lname][1] + 1)
            else:
                avg_line_fwhm[lname] = (gamma, 1)

        output_table_row = pd.DataFrame({
            "subspectrum": [list(output_table_spec["subspectrum"])[0]],
            "line_name": [lname],
            "line_loc": [lines[i]],
            "height": [p_height],
            "u_height": [p_height_err],
            "reduction_factor": [r_factor],
            "u_reduction_factor": [r_factor_err],
            "lambda_0": [shift],
            "u_lambda_0": [shifterr],
            "eta": [eta],
            "u_eta": [etaerr],
            "sigma": [to_sigma(gamma)],
            "u_sigma": [to_sigma(gammaerr)],
            "gamma": [gamma],
            "u_gamma": [gammaerr],
            "scaling": [scaling],
            "u_scaling": [scalingerr],
            "flux_0": [height],
            "u_flux_0": [heighterr],
            "slope": [slope],
            "u_slope": [slopeerr],
            "RV": [culumv],
            "u_RV": [culumv_errs],
            "signal_strength": ["--"],
            "noise_strength": ["--"],
            "SNR": ["--"],
            "sanitized": ["--"],
            "cr_ind": [cr_inds]
        })

        output_table = pd.concat([output_table, output_table_row], axis=0)

    return culumv, culumv_errs, output_table, True


def open_spec_files(loc, fpre, end=".txt"):
    """
    :param loc: list of files in output
    :param fpre: file prefix
    :param end: file extension
    :return: list of spectrumfiles
    """
    flist = []
    if "." not in end:
        end = "." + end
    for fname in loc:
        if fpre in fname:
            if re.match(fpre + r"_[0-9]+" + end, fname):
                flist.append(FILE_LOC + fname)
    return flist


def files_from_catalogue(cat):
    if USE_CATALOGUE:
        catalogue = pd.read_csv(cat)
        return [a.replace(".fits", "") for a in catalogue["file"]], catalogue
    else:
        filelist = glob.glob("spectra/*.txt")
        filenamelist = [filepath.split("\\")[-1] for filepath in filelist]
        fileprefixes = [splitname(filename)[0] for filename in filenamelist if "_mjd" in filename]
        return fileprefixes, None


def print_status(file, fileset, catalogue, file_prefix):
    if USE_CATALOGUE:
        gaia_id = catalogue["source_id"][file_prefixes.index(file_prefix)]
        print(
            f"Doing Fits for System GAIA EDR3 {gaia_id} ({file_prefix}) [{(fileset.index(file) + 1)}/{(len(fileset))}]")
    else:
        print(f"Doing Fits for System {file_prefix} [{(fileset.index(file) + 1)}/{(len(fileset))}]")


def gap_detection(arr):
    gaps = np.diff(arr)
    gapsmean = np.mean(gaps)
    gapsmed = np.median(gaps)

    if len(gaps) > 4:
        lo = np.median(gaps[np.where(gaps < gapsmed)[0]])
        hi = np.median(gaps[np.where(gaps > gapsmed)[0]])

        iqr = hi - lo

        outlier_bound = hi + 1.5 * iqr
        if not np.logical_and(gapsmed - 0.25 * gapsmed < gaps, gaps < gapsmed + 0.25 * gapsmed).all():
            return gaps, list(np.where(np.logical_or(gaps > outlier_bound, gaps > 10 * np.amin(gaps)))[0] + 1)
        else:
            return gaps, []
    else:
        if not np.logical_and(gapsmed - 0.25 * gapsmed < gaps, gaps < gapsmed + 0.25 * gapsmed).all():
            return gaps, list(np.where(np.logical_or(gaps > 2 * gapsmed, gaps > 10 * np.amin(gaps)))[0] + 1)
        else:
            return gaps, []


def gap_inds(array):
    if len(array) <= 2:
        return np.array([])
    gaps, gaps_inds = gap_detection(array)

    if len(gaps_inds) == 0:
        return np.array(gaps_inds)

    allgaps = np.array([])
    for i, subarr in enumerate(np.split(array, np.array(gaps_inds))):
        subgaps = gap_inds(subarr)
        allgaps = np.concatenate([allgaps, subgaps + gaps_inds[i - 1] if i != 0 else subgaps])

    gaps_inds = np.concatenate([gaps_inds, allgaps])
    gaps_inds, counts = np.unique(gaps_inds.astype(int), return_counts=True)

    return np.sort(gaps_inds)


def plot_rvcurve_brokenaxis(vels, verrs, times, fprefix, gaia_id, merged=False, extravels=None, extraverrs=None, extratimes=None, fit=None, custom_saveloc=None):
    plt.close()

    if len(times) > 2:
        gaps_inds = gap_inds(times)

        widths = [arr[-1] - arr[0] for arr in np.split(np.array(times), gaps_inds)]

        widths = np.array(widths)
        toothin = np.where(widths < 0.05 * np.amax(widths))[0]
        widths[toothin] += 0.05 * np.amax(widths)
    else:
        gaps_inds = []
        widths = np.array([1])
    fig, axs = plt.subplots(1, len(widths), sharey="all", facecolor="w", gridspec_kw={'width_ratios': widths}, figsize=(4.8 * 16 / 9, 4.8))
    plt.subplots_adjust(wspace=.075)

    if type(axs) == np.ndarray:
        axs[0].set_ylabel("Radial Velocity [km/s]", fontsize=PLOT_LABELS_FONT_SIZE)
    else:
        axs.set_ylabel("Radial Velocity [km/s]", fontsize=PLOT_LABELS_FONT_SIZE)

    invis_ax = fig.add_subplot(111)

    invis_ax.spines['top'].set_color('none')
    invis_ax.spines['bottom'].set_color('none')
    invis_ax.spines['left'].set_color('none')
    invis_ax.spines['right'].set_color('none')
    invis_ax.patch.set_alpha(0)
    invis_ax.tick_params(labelcolor='w',
                         top=False,
                         bottom=False,
                         left=False,
                         right=False,
                         labeltop=False,
                         labelbottom=False,
                         labelleft=False,
                         labelright=False)

    if len(times) == 0:
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        p = plt.Rectangle((left, bottom), width, height, fill=False)
        p.set_transform(invis_ax.transAxes)
        p.set_clip_on(False)
        invis_ax.add_patch(p)
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=PLOT_LABELS_FONT_SIZE, labelpad=45)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}", fontsize=PLOT_TITLE_FONT_SIZE, pad=10)
        invis_ax.text(0.5 * (left + right), 0.5 * (bottom + top), 'NO GOOD SUBSPECTRA FOUND!',
                      horizontalalignment='center',
                      verticalalignment='center',
                      transform=invis_ax.transAxes,
                      color="red")
        if custom_saveloc is None:
            if not merged:
                fig.savefig(f"output/{fprefix}/RV_variation_broken_axis{PLOT_FMT}", dpi=300)
            else:
                fig.savefig(f"output/{fprefix}_merged/RV_variation_broken_axis{PLOT_FMT}", dpi=300)
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close()
            return
        else:
            fig.savefig(custom_saveloc)

    normwidths = widths / np.linalg.norm(widths)

    if np.any(normwidths < .5) or np.amax(times) > 100:
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=PLOT_LABELS_FONT_SIZE, labelpad=45)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}", fontsize=PLOT_TITLE_FONT_SIZE, pad=10)
        plt.subplots_adjust(top=0.85, bottom=.2, left=.1, right=.95)
    else:
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=PLOT_LABELS_FONT_SIZE, labelpad=20)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}", fontsize=PLOT_TITLE_FONT_SIZE, pad=10)
        plt.subplots_adjust(top=0.85, left=.1, right=.95)

    times = np.array(times)
    times -= np.amin(times)

    splittimes = np.split(times, gaps_inds)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(widths) > 1:
        for ind, ax in enumerate(axs):
            start = splittimes[ind][0]
            end = splittimes[ind][-1]
            span = end - start
            if span == 0:
                span = 0.05 * np.amax(widths)
            start -= span * 0.1
            end += span * 0.1
            ax.set_xlim(start, end)
            mask = np.logical_and(times >= start, times <= end)

            ax.scatter(times[mask],
                       vels[mask],
                       zorder=5,
                       color=colors[0],
                       clip_on=False)
            ax.errorbar(times[mask],
                        vels[mask],
                        yerr=verrs[mask],
                        capsize=3,
                        linestyle='',
                        zorder=1,
                        color=colors[1],
                        clip_on=False)
            if fit is not None:
                timespace = np.linspace(start, end)
                ax.plot(timespace,
                        fit.evaluate(timespace),
                        color="darkred",
                        zorder=0)

            if extravels is not None:
                extramask = np.logical_and(extratimes >= start, extratimes <= end)

                ax.scatter(extratimes[extramask],
                           extravels[extramask],
                           zorder=5,
                           color="C2",
                           clip_on=False)
                ax.errorbar(extratimes[extramask],
                            extravels[extramask],
                            yerr=extraverrs[extramask],
                            capsize=3,
                            linestyle='',
                            zorder=1,
                            color="C3",
                            clip_on=False)

            kwargs = dict(transform=fig.transFigure, color='k', clip_on=False)
            [xmin, ymin], [xmax, ymax] = ax.get_position().get_points()
            d = .0075
            if ind == 0:
                ax.plot((xmax - d, xmax + d), (ymin - 2 * d, ymin + 2 * d), **kwargs)
                ax.plot((xmax - d, xmax + d), (ymax - 2 * d, ymax + 2 * d), **kwargs)
                ax.yaxis.tick_left()
                ax.spines['right'].set_visible(False)
            elif 0 < ind < len(axs) - 1:
                ax.plot((xmax - d, xmax + d), (ymin - 2 * d, ymin + 2 * d), **kwargs)
                ax.plot((xmax - d, xmax + d), (ymax - 2 * d, ymax + 2 * d), **kwargs)
                ax.plot((xmin - d, xmin + d), (ymin - 2 * d, ymin + 2 * d), **kwargs)
                ax.plot((xmin - d, xmin + d), (ymax - 2 * d, ymax + 2 * d), **kwargs)
                ax.tick_params(
                    axis='y',
                    which='both',
                    right=False,
                    left=False,
                    labelleft=False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
            else:
                ax.plot((xmin - d, xmin + d), (ymin - 2 * d, ymin + 2 * d), **kwargs)
                ax.plot((xmin - d, xmin + d), (ymax - 2 * d, ymax + 2 * d), **kwargs)
                ax.yaxis.tick_right()
                ax.spines['left'].set_visible(False)
            if normwidths[ind] < 0.20:
                ax.set_xticks([(start + end) / 2], [round((start + end) / 2, 2)])
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if 0.2 < normwidths[ind] < 0.5:
                N = 2
                if np.amax(np.unique(np.round(np.linspace(start, end - span * 0.15, 2), N), return_counts=True)[1]) > 1:
                    N = 3
                if ind == 0:
                    ax.set_xticks(np.linspace(start, end - span * 0.3, 2).tolist(), np.round(np.linspace(start, end - span * 0.15, 2), N).tolist())
                else:
                    ax.set_xticks(np.linspace(start + span * 0.3, end - span * 0.3, 2).tolist(), np.round(np.linspace(start + span * 0.15, end - span * 0.15, 2), N).tolist())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if normwidths[ind] > 0.5:
                N = 2
                if np.amax(np.unique(np.round(np.linspace(start, end - span * 0.15, 4), N), return_counts=True)[1]) > 1:
                    N = 3
                if end > 100:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                if ind == 0:
                    ax.set_xticks(np.linspace(start, end - span * 0.15, 4).tolist(), np.round(np.linspace(start, end - span * 0.15, 4), N).tolist())
                else:
                    ax.set_xticks(np.linspace(start + span * 0.15, end - span * 0.15, 4).tolist(), np.round(np.linspace(start + span * 0.15, end - span * 0.15, 4), N).tolist())
            # ax.ticklabel_format(useOffset=False)
    else:
        axs.scatter(times, vels, zorder=5, color=colors[0])
        axs.errorbar(times, vels, yerr=verrs, capsize=3, linestyle='', zorder=1, color=colors[1])

    if custom_saveloc is None:
        if not merged:
            if extravels is None:
                fig.savefig(f"output/{fprefix}/RV_variation_broken_axis{PLOT_FMT}")
            else:
                fig.savefig(f"output/{fprefix}/RV_variation_broken_axis_comparison{PLOT_FMT}")
        else:
            if extravels is None:
                fig.savefig(f"output/{fprefix}_merged/RV_variation_broken_axis{PLOT_FMT}")
            else:
                fig.savefig(f"output/{fprefix}_merged/RV_variation_broken_axis_comparison{PLOT_FMT}")
    else:
        fig.savefig(custom_saveloc)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_rvcurve(vels, verrs, times, fprefix, gaia_id, merged=False):
    plt.close()
    fig, ax1 = plt.subplots(figsize=(4.8 * 16 / 9, 4.8))
    fig.suptitle(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}", fontsize=PLOT_TITLE_FONT_SIZE)
    ax1.set_ylabel("Radial Velocity [km/s]", fontsize=PLOT_LABELS_FONT_SIZE)
    ax1.set_xlabel("Time from first datapoint onward [Days]", fontsize=PLOT_LABELS_FONT_SIZE)

    if len(times) == 0:
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        p = plt.Rectangle((left, bottom), width, height, fill=False)
        p.set_transform(ax1.transAxes)
        p.set_clip_on(False)
        ax1.add_patch(p)
        ax1.text(0.5 * (left + right), 0.5 * (bottom + top), 'NO GOOD SUBSPECTRA FOUND!',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax1.transAxes,
                 color="red")
        if not merged:
            fig.savefig(f"output/{fprefix}/RV_variation{PLOT_FMT}", dpi=300)
        else:
            fig.savefig(f"output/{fprefix}_merged/RV_variation{PLOT_FMT}", dpi=300)

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        return

    times = np.array(times)
    times -= np.amin(times)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax1.scatter(times, vels, zorder=5, color=colors[0])
    ax1.errorbar(times, vels, yerr=verrs, capsize=3, linestyle='', zorder=1, color=colors[1])

    culumvs_range = verrs.min()

    if not pd.isnull(culumvs_range):
        ax1.set_ylim((vels.min() - 2 * culumvs_range, vels.max() + 2 * culumvs_range))

    plt.tight_layout()
    if not merged:
        fig.savefig(f"output/{fprefix}/RV_variation{PLOT_FMT}", dpi=300)
    else:
        fig.savefig(f"output/{fprefix}_merged/RV_variation{PLOT_FMT}", dpi=300)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def create_pdf():
    plotpdf_exists = os.path.exists("all_RV_plots.pdf")
    plotpdfbroken_exists = os.path.exists("all_RV_plots_broken_axis.pdf")

    if not plotpdf_exists or not plotpdfbroken_exists:
        dirname = os.path.dirname(__file__)
        dirs = [f.path for f in os.scandir(os.path.join(dirname, "output")) if f.is_dir()]
        files = [os.path.join(d, f"RV_variation{PLOT_FMT}") for d in dirs]
        files_brokenaxis = [os.path.join(d, f"RV_variation_broken_axis{PLOT_FMT}") for d in dirs]

        restable = pd.read_csv("result_parameters.csv", delimiter=",", dtype={"source_id": np.int64,
                                                                              "logp": "float",
                                                                              "associated_files": "string"})

        restable.reset_index(inplace=True, drop=True)

        files.sort(key=lambda f: restable.index[restable["source_id"] == restable[[f.split("\\")[-2].replace("_merged", "") in c for c in list(restable['associated_files'])]]["source_id"].values[0]])
        files_brokenaxis.sort(key=lambda f: restable.index[restable["source_id"] == restable[[f.split("\\")[-2].replace("_merged", "") in c for c in list(restable['associated_files'])]]["source_id"].values[0]])

        if PLOT_FMT.lower() != ".pdf":
            if not plotpdf_exists:
                with open("all_RV_plots.pdf", "wb") as f:
                    f.write(img2pdf.convert(files))
            if not plotpdfbroken_exists:
                with open("all_RV_plots_broken_axis.pdf", "wb") as f:
                    f.write(img2pdf.convert(files_brokenaxis))
        else:
            from PyPDF2 import PdfFileMerger
            # if not plotpdf_exists:
            #     merger = PdfFileMerger()
            #     [merger.append(pdf) for pdf in files]
            #     with open("all_RV_plots.pdf", "wb") as new_file:
            #         merger.write(new_file)
            if not plotpdfbroken_exists:
                merger = PdfFileMerger()
                [merger.append(pdf) for pdf in files_brokenaxis]
                with open("all_RV_plots_broken_axis.pdf", "wb") as new_file:
                    merger.write(new_file)


def initial_variables():
    global dirfiles
    dirfiles = os.listdir(FILE_LOC)
    global catalogue, file_prefixes
    file_prefixes, catalogue = files_from_catalogue(CATALOGUE)
    if not VERBOSE:
        warnings.filterwarnings("ignore")


def main_loop(file_prefix):
    if os.path.isfile(f'output/.{file_prefix}') and not REDO_STARS:
        return
    fileset = open_spec_files(dirfiles, file_prefix, end=EXTENSION)
    if os.path.isdir(f'output/{file_prefix}') and not REDO_STARS:
        # TODO: Fix this mess
        if os.path.isfile(f'output/{file_prefix}/RV_variation_broken_axis{PLOT_FMT}'):
            if not SAVE_SINGLE_IMGS:
                return
            if not REDO_IMAGES:
                return
            nspec = len(fileset)
            if os.listdir(f'output/{file_prefix}/{str(nspec) if len(str(nspec)) != 1 else "0" + str(nspec)}/'):
                return
    spectimes = []
    spectimes_mjd = []
    specvels = []
    specverrs = []
    culumvs = []
    culumvs_errs = []
    single_output_table = output_table_cols.copy()
    cumulative_output_table = output_table_cols.copy()
    spec_class = catalogue["SPEC_CLASS"][file_prefixes.index(file_prefix)]
    for file in fileset:
        print_status(file, fileset, catalogue, file_prefix)
        wl, flx, time, flx_std = load_spectrum(file)
        if np.sum(flx < 0) / len(flx) > NO_NEGATIVE_FLUX[1]:
            continue
        complete_v_shift, v_std, file_prefix, output_table_spec = single_spec_shift(file, wl, flx, flx_std)
        if len(output_table_spec.index) == 0 or pd.isnull(complete_v_shift):
            print("Not a single good line was found in this subspectrum!")
            continue
        elif VERBOSE:
            print(f"Fits for {len(output_table_spec.index)} Lines complete!")
        single_output_table = pd.concat([single_output_table, output_table_spec], axis=0)
        if SUBDWARF_SPECIFIC_ADJUSTMENTS:
            if "He-" in spec_class:
                preexcluded_lines = []
                for key, value in lines_to_fit.items():
                    if "H_" in key:
                        preexcluded_lines.append(value)
                culumv, culumv_errs, output_table_spec, success = cumulative_shift(output_table_spec, file, file_prefix, exclude_lines=preexcluded_lines)
            else:
                culumv, culumv_errs, output_table_spec, success = cumulative_shift(output_table_spec, file, file_prefix)
        else:
            culumv, culumv_errs, output_table_spec, success = cumulative_shift(output_table_spec, file, file_prefix)
        if not success:
            continue
        cumulative_output_table = pd.concat([cumulative_output_table, output_table_spec], axis=0)
        culumvs.append(culumv)
        culumvs_errs.append(culumv_errs)
        spectimes.append(time.to_datetime())
        spectimes_mjd.append(time.mjd)
        specvels.append(complete_v_shift)
        specverrs.append(v_std)

    if USE_CATALOGUE:
        gaia_id = catalogue["source_id"][file_prefixes.index(file_prefix)]
    else:
        gaia_id = file_prefix

    with np.printoptions(linewidth=10000):
        single_output_table.drop("sanitized", axis=1)
        if not os.path.isdir(f"output/{file_prefix}"):
            os.mkdir(f"output/{file_prefix}")
        single_output_table.to_csv(f"output/{file_prefix}/single_spec_vals.csv", index=False)
        cumulative_output_table.drop("sanitized", axis=1)
        cumulative_output_table.to_csv(f"output/{file_prefix}/culum_spec_vals.csv", index=False)

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
    with np.printoptions(linewidth=10000):
        rvtable.to_csv(f"output/{file_prefix}/RV_variation.csv", index=False)
    if SAVE_COMPOSITE_IMG:
        # plot_rvcurve(culumvs, culumvs_errs, spectimes_mjd, file_prefix, gaia_id)
        plot_rvcurve_brokenaxis(culumvs, culumvs_errs, spectimes_mjd, file_prefix, gaia_id)


############################## EXECUTION ##############################


if __name__ == "__main__":
    if not os.path.isdir(f'output'):
        os.mkdir("output")
    if not VERBOSE:
        warnings.filterwarnings("ignore")
    print("Loading spectrum files...")
    file_prefixes, catalogue = files_from_catalogue(CATALOGUE)

    pool = Pool(initializer=initial_variables)
    pool.map(main_loop, file_prefixes)
    print("Fits are completed, analysing results...")
    result_analysis(True, catalogue)
    if CREATE_PDF:
        create_pdf()
    print("All done!")
