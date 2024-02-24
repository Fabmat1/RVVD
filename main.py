import itertools
import os
import shutil
import warnings
from multiprocessing import Pool, current_process

import PyPDF2.errors
import astropy.time as atime
import img2pdf
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import c, pi
from scipy.optimize import curve_fit

from analyse_results import result_analysis

############################## SETTINGS ##############################

### GENERAL EXECUTION SETTINGS
general_config = {
    "SPECTRUM_FILE_SEPARATOR": " ",  # Separator between columns in the ASCII file
    "CATALOGUE": "object_catalogue.csv",  # the location of the catalogue
    "FILE_LOC": "spectra_processed",  # directory that holds the spectrum files
    "OUTPUT_DIR": "output",
    "VERBOSE": False,  # enable/disable verbose output
    "NO_NEGATIVE_FLUX": True,  # check for negative flux values
    "SORT_OUT_NEG_FLX": 0.1,  # filter out spectra files with significant portions of negative flux
    "SUBDWARF_SPECIFIC_ADJUSTMENTS": True,  # Apply some tweaks for the script to be optimized to hot subdwarfs
    "GET_TICS": True,  # Get TIC IDs via query. This will be slow the first time it is run.
    "GET_VISIBILITY": True,  # Whether to get the visibility of the objects for a certain night and location.
    "FOR_DATE": "2024-03-05",  # Date for which to get the visibility
    "TAG_KNOWN": True  # Tag systems where RV variability is known
}

### FIT SETTINGS
fit_config = {
    "OUTLIER_MAX_SIGMA": 3,
    # Sigma value above which a line from the individual gets rejected as a fit to a wrong line. Outliers do not get used in the cumulative fit.
    "ALLOW_SINGLE_DATAPOINT_PEAKS": True,  # Whether to accept lines that are made up by only one datapoint.
    "MAX_ERR": 100000,  # Maximum allowed error above which a RV gets rejected as bad [m/s]
    "CUT_MARGIN": 20,
    # Margin used for cutting out disturbing lines, if their standard deviation was not yet determined [Å]
    "MARGIN": 75,  # Window margin around lines used in determining fits [Å]
    "AUTO_REMOVE_OUTLIERS": True,
    # Whether an input from the user is required to remove outliers from being used in the cumulative fit
    "MIN_ALLOWED_SNR": 5,  # Minimum allowed SNR to include a line in the cumulative fit
    "SNR_PEAK_RANGE": 1.5,  # Width of the peak that is considered the "signal" [Multiples of the FWHM]
    "COSMIC_RAY_DETECTION_LIM": 3,  # minimum times peak height/flux std required to detect cr, minimum times diff
    # std required to detect cr
    "USE_LINE_AVERAGES": False,
    # Determine guessed FWHM by examining previously fitted lines, not recommended when using multiprocessing!
}

### PLOT AND STORAGE SETTINGS

plot_config = {
    "FIG_DPI": 300,  # DPI value of plots that are created, if they are not pdf files
    "PLOT_FMT": ".pdf",  # File format of plots (.pdf is recommended due to smaller file sizes)
    "SHOW_PLOTS": False,  # Show matplotlib plotting window for each plot
    "PLOTOVERVIEW": False,  # Plot overview of entire subspectrum
    "SAVE_SINGLE_IMGS": False,
    # Save individual plots of fits as images in the respective folders !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!
    "REDO_IMAGES": False,  # Redo images already present in folders
    "SAVE_COMPOSITE_IMG": True,  # Save RV-Curve plot
    "REDO_STARS": False,  # Whether to redo stars for which RVs have already be determined
    "PLOT_LABELS_FONT_SIZE": 14,  # Label font size
    "PLOT_TITLE_FONT_SIZE": 17,  # Title font size
    "CREATE_PDF": True,
    # Group all RV-plots into one big .pdf at the end of the calculations !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!
}

mpl.rcParams['figure.dpi'] = plot_config["FIG_DPI"]

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

line_FWHM_guesses = {'H_alpha': 13.893084511019556,
                     # FWHM guess for fitting for each line (only relevant if USE_LINE_AVERAGES is set to false)
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
mpl.rcParams["axes.formatter.useoffset"] = False
general_config["post_progress"] = False

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
}, dtype=object)

wl_splitinds, flux_splitinds, flux_std_splitinds = [0, 0, 0]
linelist = []

log_two = np.log(2)


class WindowWrapper:
    def __init__(self, window=None):
        self.window = window

    def __getattr__(self, name):
        if self.window is not None:
            return getattr(self.window, name)
        else:
            return None


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


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    g = gaussian(x, gamma, shift)
    l = lorentzian(x, gamma, shift)
    return -scaling * (eta * g + (1 - eta) * l) + slope * x + height


def load_spectrum(filename, preserve_below_zero=False, no_meta=False):
    """
    :param filename: Spectrum File location
    :return: Spectrum Wavelengths, Corresponding flux, time of observation, flux error (if available)
    """
    # Modify this to account for your specific needs!
    data = np.loadtxt(filename, comments="#", delimiter=general_config["SPECTRUM_FILE_SEPARATOR"])
    wavelength = data[:, 0]
    flux = data[:, 1]
    flux_std = data[:, 2]

    if general_config["NO_NEGATIVE_FLUX"] and not preserve_below_zero:
        mask = flux > 0
        wavelength = wavelength[mask]
        flux_std = flux_std[mask]
        flux = flux[mask]
    if not no_meta:
        filename_prefix, nspec = splitname(filename)
        nspec = nspec.replace(".txt", "")
        nspec = int(nspec)
        try:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#",
                                      delimiter=general_config["SPECTRUM_FILE_SEPARATOR"])[nspec - 1], format="mjd")
        except IndexError:
            t = atime.Time(np.loadtxt(splitname(filename)[0] + "_mjd.txt", comments="#",
                                      delimiter=general_config["SPECTRUM_FILE_SEPARATOR"]), format="mjd")

        if plot_config['PLOTOVERVIEW']:
            plt.title("Full Spectrum Overview")
            plt.ylabel("Flux [ergs/s/cm^2/Å]")
            plt.xlabel("Wavelength [Å]")
            plt.plot(wavelength, flux, color="navy")
            if plot_config['SHOW_PLOTS']:
                plt.show()
        if "flux_std" not in vars():
            flux_std = np.zeros(np.shape(flux))
        return wavelength, flux, t, flux_std
    else:
        return wavelength, flux, flux_std


def calc_SNR(params, flux, wavelength, margin, cr_inds=None):
    """
    :param params: parameters of Fit
    :param flux: Flux array
    :param wavelength: Wavelength array
    :param margin: index width of plotted area (arbitrary)
    :return:    Mean squared displacement(MSD) of signal area,
                MSD of noise background,
                Signal-to-Noise ratio
    """
    if len(cr_inds) > 0:
        mask = np.ones_like(wavelength, dtype=bool)
        mask[cr_inds] = False
        wavelength = wavelength[mask]
        flux = flux[mask]

    scaling, gamma, shift, slope, height, eta = params
    flux = flux - slope * wavelength - height

    slicedwl, loind, upind = slicearr(wavelength, shift - fit_config['SNR_PEAK_RANGE'] * gamma,
                                      shift + fit_config['SNR_PEAK_RANGE'] * gamma)
    signalstrength = np.mean(np.square(flux[loind:upind]))

    if upind == loind + 1 and not fit_config['ALLOW_SINGLE_DATAPOINT_PEAKS']:
        warnings.warn("Peak is only a single datapoint, Fit rejected.", FitUnsuccessfulWarning)
        return 0
    elif upind == loind + 1 and fit_config['ALLOW_SINGLE_DATAPOINT_PEAKS']:
        # Maybe resample here
        pass

    if 2 * fit_config['SNR_PEAK_RANGE'] * gamma < margin:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - margin, shift - fit_config['SNR_PEAK_RANGE'] * gamma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + fit_config['SNR_PEAK_RANGE'] * gamma, shift + margin)
    else:
        slicedwl, lloind, lupind = slicearr(wavelength, shift - margin, shift - gamma)
        slicedwl, uloind, uupind = slicearr(wavelength, shift + gamma, shift + margin)
        warnings.warn("Sigma very large, Fit seems improbable!", NoiseWarning)
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color="red")
    noisestrength = np.mean(np.square(np.array(flux[lloind:lupind].tolist() + flux[uloind:uupind].tolist())))

    SNR = signalstrength / noisestrength

    if np.isnan(SNR):
        SNR = 0

    return SNR


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
            if line in linewidths.keys():
                margin = linewidths[line]
            else:
                margin = fit_config['CUT_MARGIN']
        except NameError:
            margin = fit_config['CUT_MARGIN']
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

    lwl_for_std, lloind, lupind = slicearr(modified_slicedwl, shift - fit_config['MARGIN'], shift - 2 * sigma)
    uwl_for_std, uloind, uupind = slicearr(modified_slicedwl, shift + 2 * sigma, shift + fit_config['MARGIN'])

    if type(normalized_flux) == np.ma.core.MaskedArray:
        for_std = np.concatenate(
            [normalized_flux[lloind:lupind].compressed(), normalized_flux[uloind:uupind].compressed()])
    else:
        for_std = np.concatenate([normalized_flux[lloind:lupind], normalized_flux[uloind:uupind]])
    std = np.std(for_std)

    if len(predetermined_crs) == 0:
        initial_crs = np.where(normalized_flux > fit_config['COSMIC_RAY_DETECTION_LIM'] * std)[0]
    else:
        allowed_inds = np.concatenate((predetermined_crs + 1, predetermined_crs - 1))
        initial_crs = np.where(normalized_flux > fit_config['COSMIC_RAY_DETECTION_LIM'] * std)[0]
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


def plot_peak_region(wavelengthdata, fluxdata, flux_stddata, center, margin, file_prefix, gaia_id, subspec_ind,
                     sanitize=False,
                     used_cr_inds=None, reset_initial_param=False):
    if used_cr_inds is None:
        used_cr_inds = []
    wavelength = np.copy(wavelengthdata)
    flux = np.copy(fluxdata)
    flux_std = np.copy(flux_stddata)

    coverage = [len(np.where(np.logical_and(wavelength > center - 2, wavelength < center + 2))[0]) == 0, len(
        np.where(np.logical_and(wavelength > center - margin - 2, wavelength < center - margin + 2))[0]) == 0,
                len(np.where(np.logical_and(wavelength > center + margin - 2, wavelength < center + margin + 2))[
                        0]) == 0]

    if sum(coverage) != 0:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
            False, False, False

    for i in disturbing_lines.values():
        if i != center:
            if i - fit_config['CUT_MARGIN'] < center + fit_config['MARGIN'] or i + fit_config['CUT_MARGIN'] > center - \
                    fit_config['MARGIN']:
                sanitize = True

    if used_cr_inds is None:
        used_cr_inds = []

    if sanitize:
        flux, cut_flux, mask = sanitize_flux(flux, center, wavelength)
    slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)
    try:
        assert len(slicedwl) > 0
    except AssertionError:
        warnings.warn(
            "Seems like the line(s) you want to look at are not in the spectrum " + file_prefix + " Line @ " + str(
                center) + " Check your files!", FitUnsuccessfulWarning)
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
            False, False, False

    for key, val in lines_to_fit.items():
        if round(val) == round(center):
            lstr = key
    if "lstr" not in locals():
        lstr = "unknown"

    if plot_config['SAVE_SINGLE_IMGS']:
        plt.title(f"Fit for Line {lstr} @ {round(center)}Å")
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.plot(slicedwl, flux[loind:upind], zorder=5)

    sucess = True

    if sanitize:
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.plot(slicedwl, cut_flux[loind:upind], color="lightgrey", label='_nolegend_', zorder=1)
        wavelength = wavelength[mask]
        flux = flux.compressed()
        flux_std = flux_std[mask]
        slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

    flux_std[flux_std == 0] = np.mean(flux_std)

    try:
        initial_slope = (np.mean(flux[loind:upind][-round(len(flux[loind:upind]) / 5):]) - np.mean(
            flux[loind:upind][:round(len(flux[loind:upind]) / 5)])) / (slicedwl[-1] - slicedwl[0])
        initial_h = np.mean(flux[loind:upind][:round(len(flux[loind:upind]) / 5)]) - np.mean(
            slicedwl[:round(len(flux[loind:upind]) / 5)]) * initial_slope
    except IndexError:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
            False, False, False

    try:
        flx_for_initial = flux[loind:upind] - slicedwl * initial_slope + initial_h
    except ValueError:
        return False, [False, False, False, False, False, False], [False, False, False, False, False, False], \
            False, False, False

    if lstr != "unknown" and not reset_initial_param:
        if fit_config['USE_LINE_AVERAGES']:
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
        [0, 0, center - fit_config['MARGIN'] * 0.25, -np.inf, -np.inf, 0],
        [np.inf, margin / 2, center + fit_config['MARGIN'] * 0.25, np.inf, np.inf, 1]
    )

    try:
        if np.sum(flux_std) != 0:
            params, errs = curve_fit(pseudo_voigt,
                                     slicedwl,
                                     flux[loind:upind],
                                     initial_params,
                                     # scaling, gamma, shift, slope, height, eta
                                     bounds=bounds,
                                     sigma=flux_std[loind:upind]
                                     )
        else:
            params, errs = curve_fit(pseudo_voigt,
                                     slicedwl,
                                     flux[loind:upind],
                                     initial_params,
                                     # scaling, gamma, shift, slope, height, eta
                                     bounds=bounds
                                     )

        errs = np.sqrt(np.diag(errs))

        if len(used_cr_inds) == 0:
            cr, cr_ind = cosmic_ray(slicedwl, flux[loind:upind], params, center)
            if cr and np.sum(cr_ind) > 0:
                cr_ind += loind
                cr_true_inds = wavelengthdata.searchsorted(wavelength[cr_ind])
                plt.close()
                if plot_config['SAVE_SINGLE_IMGS']:
                    for i in cr_ind:
                        plt.plot(wavelength[i - 1:i + 2], flux[i - 1:i + 2], color="lightgray", label='_nolegend_')
                return plot_peak_region(np.delete(wavelength, cr_ind), np.delete(flux, cr_ind),
                                        np.delete(flux_std, cr_ind), center, margin, file_prefix, gaia_id, subspec_ind,
                                        used_cr_inds=cr_true_inds)
        SNR = calc_SNR(params, flux, wavelength, margin)
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}", (10, 10), xycoords="figure pixels")
        if SNR < fit_config['MIN_ALLOWED_SNR']:
            if sucess:
                if plot_config['SAVE_SINGLE_IMGS']:
                    warn_text = plt.figtext(0.3, 0.95, f"BAD SIGNAL!",
                                            horizontalalignment='right',
                                            verticalalignment='bottom',
                                            color="red")
                if not sanitize:
                    if plot_config['SAVE_SINGLE_IMGS']:
                        warn_text.set_visible(False)
                        plt.close()
                    return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, gaia_id,
                                            subspec_ind,
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
        #     return plot_peak_region(wavelengthdata, fluxdata, flux_stddata, center, margin, file_prefix, gaia_id, subspec_ind, sanitize, used_cr_inds)

        if plot_config['SAVE_SINGLE_IMGS']:
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *params), zorder=5)
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=6, color="black")

    except RuntimeError:
        sucess = False
        warnings.warn("Could not find a good Fit!", FitUnsuccessfulWarning)

        if plot_config['SAVE_SINGLE_IMGS']:
            warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    color="red")
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
        if not sanitize:
            if plot_config['SAVE_SINGLE_IMGS']:
                warn_text.set_visible(False)
                plt.close()
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, gaia_id, subspec_ind,
                                    sanitize=True)
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
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, gaia_id, subspec_ind,
                                    sanitize=True)
        sucess = False
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
    if plot_config['SAVE_SINGLE_IMGS']:
        plt.axvline(center, linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
        plt.legend(["Flux", "Best Fit"])

    if not os.path.isdir(f'{general_config["OUTPUT_DIR"]}/{gaia_id}/'):
        os.mkdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}/")
    if not os.path.isdir(f'{general_config["OUTPUT_DIR"]}/{gaia_id}/{subspec_ind}') and plot_config['SAVE_SINGLE_IMGS']:
        os.mkdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}")
        plt.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}/{round(center)}Å{plot_config['PLOT_FMT']}")
        if plot_config['SHOW_PLOTS']:
            plt.show()
    plt.close()
    if sucess:
        return sucess, errs, params, SNR, sanitize, used_cr_inds
    else:
        if not reset_initial_param:
            return plot_peak_region(wavelength, flux, flux_std, center, margin, file_prefix, gaia_id, subspec_ind,
                                    sanitize=sanitize, reset_initial_param=True)
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False], \
            False, False, False


def pseudo_voigt_height(errs, scaling, eta, gamma):
    height = 2 * scaling / (np.pi * gamma) * (1 + (np.sqrt(np.pi * log_two) - 1) * eta)
    if errs is not None:
        err = height_err(eta, gamma, scaling, errs[5], errs[1], errs[0])
        return height, err
    else:
        return height


def print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc):
    if general_config["VERBOSE"]:
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
    if general_config["VERBOSE"]:
        print("\n\n##################### SPECTRUM RESULTS ######################")
        print(f"Result for Spectrum {os.path.basename(filename)}:")
        print(f"Velocity: [{round(complete_v_shift / 1000, 2)}±{round(v_std / 1000, 2)}]km/s")
        print("#############################################################\n\n")


def check_for_outliers(array):
    return np.zeros(np.shape(array)).astype(bool)
    # med = np.median(array)
    # standard_deviation = np.std(array)
    # distance_from_mean = abs(array - med)
    # outlierloc = distance_from_mean < fit_config["OUTLIER_MAX_SIGMA"] * standard_deviation
    # return np.array(outlierloc)


def single_spec_shift(filepath, wl, flx, flx_std, gaia_id, subspec_ind):
    fname = filepath.split("/")[-1].split(".")[0]
    velocities = []
    verrs = []
    output_table = output_table_cols.copy()
    global linewidths
    linewidths = {}

    for lstr, loc in lines_to_fit.items():
        sucess, errs, [scaling, gamma, shift, slope, height, eta], SNR, sanitized, cr_ind = plot_peak_region(wl, flx,
                                                                                                             flx_std,
                                                                                                             loc,
                                                                                                             fit_config[
                                                                                                                 'MARGIN'],
                                                                                                             fname,
                                                                                                             gaia_id,
                                                                                                             subspec_ind)

        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
            linewidths[loc] = 1.5 * gamma
            rv = v_from_doppler(shift, loc)
            u_rv = v_from_doppler_err(shift, loc, errs[2])
            velocities.append(rv)
            verrs.append(u_rv)

            h, u_h = pseudo_voigt_height(errs, scaling, eta, gamma)
            u_scaling, u_gamma, u_shift, u_slope, u_height, u_eta = errs

            output_table_row = pd.DataFrame({
                "subspectrum": [f"{gaia_id}_{subspec_ind}"],
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
                "SNR": [SNR],
                "cr_ind": [cr_ind],
                "sanitized": [sanitized],
            })

            output_table = pd.concat([output_table, output_table_row], axis=0)

    velocities = np.array(velocities)
    verrs = np.array(verrs)
    outloc = check_for_outliers(velocities)
    if np.invert(outloc).sum() != 0 and len(velocities) > 4:
        if not fit_config['AUTO_REMOVE_OUTLIERS']:
            print(
                f"! DETECTED OUTLIER CANDIDATE (DEVIATION > {fit_config['OUTLIER_MAX_SIGMA']}σ), REMOVE OUTLIER? [Y/N]")
            print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = input()
        else:
            if general_config["VERBOSE"]:
                print(f"! DETECTED OUTLIER (DEVIATION > {fit_config['OUTLIER_MAX_SIGMA']}σ)")
                print(f"IN ARRAY: {velocities}; SPECIFICALLY {velocities[~outloc]}")
            del_outlier = "y"

        if del_outlier.lower() == "y":
            velocities = velocities[outloc]
            verrs = verrs[outloc]
            output_table = output_table.loc[outloc, :]

    v_std = np.sqrt(np.sum(verrs ** 2))
    complete_v_shift = np.mean(velocities)
    print_single_spec_results(complete_v_shift, v_std, filepath)

    return complete_v_shift, v_std, output_table


def process_spectrum(file, gaia_id, subspec_ind, exclude_lines=[]):
    wavelengthdata, fluxdata, time, flux_stddata = load_spectrum(file)
    linelist = []
    paramlist = []
    cr_inds = []
    r_facs = []
    sanlist = []
    velocities = []
    verrs = []

    margin = fit_config['MARGIN']

    for lstr, center in lines_to_fit.items():
        if center in exclude_lines:
            continue
        wavelength = np.copy(wavelengthdata)
        flux = np.copy(fluxdata)
        flux_std = np.copy(flux_stddata)

        coverage = [len(np.where(np.logical_and(wavelength > center - 2, wavelength < center + 2))[0]) == 0,
                    len(np.where(np.logical_and(wavelength > center - margin - 2, wavelength < center - margin + 2))[0]) == 0,
                    len(np.where(np.logical_and(wavelength > center + margin - 2, wavelength < center + margin + 2))[0]) == 0]

        if sum(coverage) != 0:
            # print(f"[{gaia_id}] Line {lstr} not covered by spectrum, skipping!")
            continue

        sanitize = False
        for i in disturbing_lines.values():
            if (i != center) and (i - fit_config['CUT_MARGIN'] < center + fit_config['MARGIN'] or i + fit_config['CUT_MARGIN'] > center - fit_config['MARGIN']):
                sanitize = True

        if sanitize:
            flux, cut_flux, mask = sanitize_flux(flux, center, wavelength)
        slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

        try:
            assert len(slicedwl) > 0
        except AssertionError:
            warnings.warn(
                "Seems like the line(s) you want to look at are not in the spectrum " + gaia_id + " Line @ " + str(
                    center) + " Check your files!", FitUnsuccessfulWarning)
            sucess = False
            continue

        if plot_config['SAVE_SINGLE_IMGS']:
            plt.title(f"Fit for Line {lstr} @ {round(center)}Å")
            plt.ylabel("Flux [ergs/s/cm^2/Å]")
            plt.xlabel("Wavelength [Å]")
            plt.plot(slicedwl, flux[loind:upind], zorder=5)

        sucess = True

        if sanitize:
            if plot_config['SAVE_SINGLE_IMGS']:
                plt.plot(slicedwl, cut_flux[loind:upind], color="lightgrey", label='_nolegend_', zorder=1)
            wavelength = wavelength[mask]
            flux = flux.compressed()
            flux_std = flux_std[mask]
            slicedwl, loind, upind = slicearr(wavelength, center - margin, center + margin)

        flux_std[flux_std == 0] = np.mean(flux_std)

        initial_slope = (np.mean(flux[loind:upind][-round((upind - loind) / 5):]) - np.mean(
            flux[loind:upind][:round((upind - loind) / 5)])) / (slicedwl[-1] - slicedwl[0])
        initial_h = np.mean(flux[loind:upind][:round((upind - loind) / 5)]) - np.mean(
            slicedwl[:round((upind - loind) / 5)]) * initial_slope

        try:
            flx_for_initial = flux[loind:upind] - slicedwl * initial_slope + initial_h
        except ValueError:
            sucess = False
            continue

        if lstr != "unknown":
            if fit_config['USE_LINE_AVERAGES']:
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
            [0, 0, center - fit_config['MARGIN'] * 0.25, -np.inf, -np.inf, 0],
            [np.inf, margin / 2, center + fit_config['MARGIN'] * 0.25, np.inf, np.inf, 1]
        )
        try:
            if np.sum(flux_std) != 0:
                params, errs = curve_fit(pseudo_voigt,
                                         slicedwl,
                                         flux[loind:upind],
                                         initial_params,
                                         # scaling, gamma, shift, slope, height, eta
                                         bounds=bounds,
                                         sigma=flux_std[loind:upind]
                                         )
            else:
                params, errs = curve_fit(pseudo_voigt,
                                         slicedwl,
                                         flux[loind:upind],
                                         initial_params,
                                         # scaling, gamma, shift, slope, height, eta
                                         bounds=bounds
                                         )

            errs = np.sqrt(np.diag(errs))

            cr, cr_ind = cosmic_ray(slicedwl, flux[loind:upind], params, center)
            if len(cr_ind) > 0:
                mask = np.ones_like(slicedwl, dtype=bool)
                mask[cr_ind] = False
                slicedwl = slicedwl[mask]
                slicedflux = flux[loind:upind][mask]
                slicedflux_std = flux_std[loind:upind][mask]

                cr_ind += loind
                cr_true_inds = wavelengthdata.searchsorted(wavelength[cr_ind])
                cr_inds = [*cr_inds, *cr_true_inds]

                if np.sum(slicedflux_std) != 0:
                    params, errs = curve_fit(pseudo_voigt,
                                             slicedwl,
                                             slicedflux,
                                             initial_params,
                                             # scaling, gamma, shift, slope, height, eta
                                             bounds=bounds,
                                             sigma=slicedflux_std
                                             )
                else:
                    params, errs = curve_fit(pseudo_voigt,
                                             slicedwl,
                                             slicedflux,
                                             initial_params,
                                             # scaling, gamma, shift, slope, height, eta
                                             bounds=bounds
                                             )

                errs = np.sqrt(np.diag(errs))

            SNR = calc_SNR(params, flux, wavelength, margin, cr_inds=cr_ind)

            if SNR < fit_config['MIN_ALLOWED_SNR']:
                sucess = False

            height, u_height = pseudo_voigt_height(errs, params[0], params[5], params[1])
            if height > 1.5 * np.ptp(flux[loind:upind]) and sucess:
                sucess = False
            if height < u_height and sucess:
                sucess = False

        except RuntimeError:
            sucess = False
            warnings.warn("Could not find a good Fit!", FitUnsuccessfulWarning)

            if plot_config['SAVE_SINGLE_IMGS']:
                warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                        horizontalalignment='right',
                                        verticalalignment='bottom',
                                        color="red")
                plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)

        except ValueError as e:
            print("No peak found:", e)
            warn_text = plt.figtext(0.3, 0.95, f"FIT FAILED!",
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    color="red")
            sucess = False
            if plot_config['SAVE_SINGLE_IMGS']:
                plt.plot(slicedwl, pseudo_voigt(slicedwl, *initial_params), zorder=5)
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.axvline(center, linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
            plt.legend(["Flux", "Best Fit"])

        if not os.path.isdir(f'{general_config["OUTPUT_DIR"]}/{gaia_id}/'):
            os.mkdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}/")
        if not os.path.isdir(f'{general_config["OUTPUT_DIR"]}/{gaia_id}/{subspec_ind}') and plot_config['SAVE_SINGLE_IMGS']:
            os.mkdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}")
            plt.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}/{round(center)}Å{plot_config['PLOT_FMT']}")
            if plot_config['SHOW_PLOTS']:
                plt.show()
        plt.close()

        if sucess:
            rv = v_from_doppler(params[2], center)
            u_rv = v_from_doppler_err(params[2], center, errs[2])
            velocities.append(rv)
            verrs.append(u_rv)
            sanlist.append(sanitize)
            linelist.append(center)
            paramlist.append(params)
            r_facs.append(params[2] / center)

        # print(f"[{gaia_id}] First pass found {len(linelist)} good lines.")
    if len(linelist) == 0:
        print(f"[{gaia_id}] First pass found no good lines, continuing to next spectrum...")
        return None, None, None, False
    mask = np.ones_like(wavelengthdata, dtype=bool)
    cr_inds = np.array(cr_inds, dtype=int)
    mask[cr_inds[cr_inds < len(cr_inds)]] = False
    wavelengthdata = wavelengthdata[mask]
    fluxdata = fluxdata[mask]
    flux_stddata = flux_stddata[mask]

    filedata = wavelengthdata, fluxdata, time, flux_stddata

    return cumulative_shift(filedata, linelist, paramlist, r_facs, exclude_lines, sanlist, gaia_id, subspec_ind)


def cumulative_shift(filedata, linelist, paramlist, r_facs, exclude_lines, sanlist, gaia_id, subspec_ind):
    wl, flx, time, flx_std = filedata

    linelist = [l for l in linelist if l not in exclude_lines]

    wl_dataset = []
    flux_dataset = []
    flux_std_dataset = []
    output_table = output_table_cols.copy()

    for i, line in enumerate(linelist):
        if not sanlist[i]:
            wldata, loind, upind = slicearr(wl, line - fit_config['MARGIN'], line + fit_config['MARGIN'])
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
            wldata, loind, upind = slicearr(sanwl, line - fit_config['MARGIN'], line + fit_config['MARGIN'])
            if len(wldata) > 1 and len(flx[loind:upind]) > 1:
                wl_dataset.append(wldata)
                flux_dataset.append(sanflx[loind:upind])
                flux_std_dataset.append(sanflx_std[loind:upind])
            else:
                linelist.remove(line)

    if len(wl_dataset) == 0:
        print("No good data after sanitization!")
        # print(f"[{gaia_id}] Sanitization failed, aborting...")
        return None, None, None, False

    wl_splitinds = [len(wldata) for wldata in wl_dataset]
    flux_splitinds = [len(flxdata) for flxdata in flux_dataset]
    flux_std_splitinds = [len(flxstddata) for flxstddata in flux_std_dataset]
    wl_splitinds = [sum(wl_splitinds[:ind + 1]) for ind, wl in enumerate(wl_splitinds)]
    flux_splitinds = [sum(flux_splitinds[:ind + 1]) for ind, flx in enumerate(flux_splitinds)]
    flux_std_splitinds = [sum(flux_std_splitinds[:ind + 1]) for ind, flxstd in enumerate(flux_std_splitinds)]

    rmean = np.array(r_facs).mean()

    p0 = [rmean]
    bounds = [
        [0],
        [np.inf]
    ]
    # scaling, gamma, slope, height, eta
    for i in range(len(wl_dataset)):
        theseparams = paramlist[i]
        p0 += [
            theseparams[0],
            theseparams[1],
            theseparams[3],
            theseparams[4],
            theseparams[5],
        ]
        bounds[0] += [theseparams[0] / 2, theseparams[1] / 2, -np.inf, -np.inf, 0]
        bounds[1] += [theseparams[0] * 1.5, theseparams[1] * 1.5, np.inf, np.inf, 1]

    wl_dataset = np.concatenate(wl_dataset, axis=0)
    flux_dataset = np.concatenate(flux_dataset, axis=0)
    flux_std_dataset = np.concatenate(flux_std_dataset, axis=0)

    flux_std_dataset[flux_std_dataset == 0] = np.mean(flux_std_dataset)

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

    try:
        if np.sum(flux_std_dataset) != 0:
            params, errs = curve_fit(
                culum_fit_funciton,
                wl_dataset,
                flux_dataset,
                p0=p0,
                bounds=bounds,
                sigma=flux_std_dataset,
                max_nfev=100000
            )
        else:
            params, errs = curve_fit(
                culum_fit_funciton,
                wl_dataset,
                flux_dataset,
                p0=p0,
                bounds=bounds,
                max_nfev=100000
            )
    except RuntimeError:
        print("Runtime Error! Cumulative fit failed!")
        # print(f"[{gaia_id}] Runtime error, cumulative fit failed...")
        return None, None, None, False
    except ValueError:
        print("Value Error! Cumulative fit failed!")
        # print(f"[{gaia_id}] Value error, cumulative fit failed...")
        return None, None, None, False

    errs = np.sqrt(np.diag(errs))

    r_factor = params[0]
    r_factor_err = errs[0]

    culumv = v_from_doppler_rel(r_factor)
    culumv_errs = v_from_doppler_rel_err(r_factor, r_factor_err)

    errs = np.split(np.array(errs)[1:], len(np.array(errs)[1:]) / 5)
    params = np.split(np.array(params)[1:], len(np.array(params)[1:]) / 5)

    if np.abs(culumv_errs) > fit_config['MAX_ERR'] or culumv_errs == 0.0:
        if len(linelist) > 1:
            u_heights = np.array([height_err(p[4], p[1], p[0], e[4], e[1], e[0]) for p, e in zip(params, errs)])
            exclude_lines.append(linelist[np.argmax(u_heights)])
            # print(f"[{gaia_id}] Cumulative fit failed for line {linelist[np.argmax(u_heights)]}, excluding this line and retrying.")
            return cumulative_shift(filedata, linelist, paramlist, r_facs, exclude_lines, sanlist, gaia_id, subspec_ind)
        else:
            print("Cumulative fit spurious!")
            # print(f"[{gaia_id}] Cumulative fit exceeded allowed error levels...")
            return None, None, None, False

    wl_dataset = np.split(wl_dataset, wl_splitinds)
    flux_dataset = np.split(flux_dataset, flux_splitinds)

    for i, paramset in enumerate(params):
        lines = list(linelist)
        lname = list(lines_to_fit.keys())[list(lines_to_fit.values()).index(lines[i])]

        scaling, gamma, slope, height, eta = paramset
        shift = r_factor * lines[i]
        scalingerr, gammaerr, slopeerr, heighterr, etaerr = errs[i]
        shifterr = r_factor_err * lines[i]

        p_height = pseudo_voigt_height(None, scaling, eta, gamma)
        p_height_err = height_err(eta, gamma, scaling, etaerr, gammaerr, scalingerr)

        if plot_config['SAVE_SINGLE_IMGS']:
            plt.ylabel("Flux [ergs/s/cm^2/Å]")
            plt.xlabel("Wavelength [Å]")
            plt.title(f"cumulative Fit for Line {lname} @ {round(lines[i])}Å")
            plt.axvline(lines[i], linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
            plt.plot(wl_dataset[i], flux_dataset[i])
            wllinspace = np.linspace(wl_dataset[i][0], wl_dataset[i][-1], 1000)
            plt.plot(wllinspace, make_dataset(wllinspace, r_factor, i, paramset))
        if p_height < p_height_err or p_height > 1.5 * np.ptp(flux_dataset[i]):
            if plot_config['SAVE_SINGLE_IMGS']:
                plt.figtext(0.3, 0.95, f"FIT REJECTED!",
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            color="red")
                plt.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}/culum_{round(lines[i])}Å{plot_config['PLOT_FMT']}")
            if len(linelist) > 1:
                exclude_lines.append(lines[i])
                # print(f"[{gaia_id}] Cumulative height of line {lines[i]} is weird, retrying...")
                return cumulative_shift(filedata, linelist, paramlist, r_facs, exclude_lines, sanlist, gaia_id, subspec_ind)
            else:
                print("Cumulative shift rejected!")
                # print(f"[{gaia_id}] Cumulative shift rejected based on line height!")
                return None, None, None, False
        if plot_config['SAVE_SINGLE_IMGS']:
            plt.savefig(
                f"{general_config['OUTPUT_DIR']}/{gaia_id}/{subspec_ind}/culum_{round(lines[i])}Å{plot_config['PLOT_FMT']}")
            if plot_config['SHOW_PLOTS']:
                plt.show()
        plt.close()

        if fit_config['USE_LINE_AVERAGES']:
            if avg_line_fwhm[lname] is not None:
                avg_line_fwhm[lname] = (
                    (avg_line_fwhm[lname][0] * avg_line_fwhm[lname][1] + gamma) / (avg_line_fwhm[lname][1] + 1),
                    avg_line_fwhm[lname][1] + 1)
            else:
                avg_line_fwhm[lname] = (gamma, 1)

        output_table_row = pd.DataFrame({
            "subspectrum": [f"{gaia_id}_{subspec_ind}"],
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
        })

        output_table = pd.concat([output_table, output_table_row], axis=0)

    # print(f"[{gaia_id}] Cumulative fit gave RV of {culumv/1000} ± {culumv_errs/1000} km/s.")

    return culumv, culumv_errs, output_table, True


def int_to_twodigit_string(i):
    if i < 10:
        return "0" + str(i)
    else:
        return str(i)


def open_spec_files(loc, fpres, end=".txt"):
    """
    :param loc: path to spectra
    :param fpres: file prefix(es)
    :param end: file extension
    :return: list of spectrumfiles
    """
    flist = []
    if "." not in end:
        end = "." + end

    for fpre in fpres:
        n = 1
        fname = loc + "/" + fpre + "_" + int_to_twodigit_string(n) + end
        while os.path.isfile(fname):
            flist.append(fname)
            n += 1
            fname = loc + "/" + fpre + "_" + int_to_twodigit_string(n) + end

    return flist


def files_from_catalogue(cat):
    catalogue = pd.read_csv(cat, dtype={"source_id": "S20"})
    catalogue["file"] = catalogue["file"].apply(lambda x: x.replace(".fits", ""))
    # name,source_id,ra,dec,file,SPEC_CLASS,bp_rp,gmag,nspec
    catalogue = catalogue.groupby("source_id", as_index=False).aggregate({
        'name': 'first',
        'source_id': 'first',
        'ra': 'first',
        'dec': 'first',
        'file': lambda x: list(x),
        'SPEC_CLASS': 'first',
        'bp_rp': 'first',
        'gmag': 'first',
        'pmra': 'first',
        'pmdec': 'first',
        'pmra_error': 'first',
        'pmdec_error': 'first',
        'parallax': 'first',
        'parallax_error': 'first',
        'nspec': 'sum',
    })
    catalogue.source_id = catalogue.source_id.str.decode(encoding="ASCII")
    return catalogue


def print_status(file, fileset, gaia_id, file_prefix):
    if general_config["post_progress"]:
        general_config["queue"].put(
            (current_process()._identity[0], 100 * (fileset.index(file) + 1) / (len(fileset)), "progressbar"))
        general_config["queue"].put((current_process()._identity[0], gaia_id, "text"))
    if len(file_prefix) == 1:
        fstr = file_prefix[0]
    else:
        fstr = file_prefix[0] + f" and {len(file_prefix) - 1} more"
    print(f"Doing Fits for System GAIA EDR3 {gaia_id} ({fstr}) [{(fileset.index(file) + 1)}/{(len(fileset))}]")


def gap_detection(arr):
    gaps = np.diff(arr)
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


def plot_rvcurve_brokenaxis(vels, verrs, times, gaia_id, merged=False, extravels=None, extraverrs=None, extratimes=None,
                            fit=None, custom_saveloc=None):
    plt.close()

    vels = vels[np.argsort(times)]
    verrs = verrs[np.argsort(times)]
    times = np.sort(times)

    if len(times) > 2:
        gaps_inds = gap_inds(times)

        widths = [arr[-1] - arr[0] for arr in np.split(np.array(times), gaps_inds)]

        widths = np.array(widths)
        toothin = np.where(widths < 0.05 * np.amax(widths))[0]
        widths[toothin] += 0.05 * np.amax(widths)
    else:
        gaps_inds = []
        widths = np.array([1])
    fig, axs = plt.subplots(1, len(widths), sharey="all", facecolor="w", gridspec_kw={'width_ratios': widths},
                            figsize=(4.8 * 16 / 9, 4.8))
    plt.subplots_adjust(wspace=.075)

    if type(axs) == np.ndarray:
        axs[0].set_ylabel("Radial Velocity [km/s]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'])
    else:
        axs.set_ylabel("Radial Velocity [km/s]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'])

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
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'],
                            labelpad=45)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}",
                           fontsize=plot_config['PLOT_TITLE_FONT_SIZE'], pad=10)
        invis_ax.text(0.5 * (left + right), 0.5 * (bottom + top), 'NO GOOD SUBSPECTRA FOUND!',
                      horizontalalignment='center',
                      verticalalignment='center',
                      transform=invis_ax.transAxes,
                      color="red")
        if custom_saveloc is None:
            if not merged:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation_broken_axis{plot_config['PLOT_FMT']}",
                    dpi=300)
            else:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}_merged/RV_variation_broken_axis{plot_config['PLOT_FMT']}",
                    dpi=300)
            if plot_config['SHOW_PLOTS']:
                plt.show()
            else:
                plt.close()
            return
        else:
            fig.savefig(custom_saveloc)

    if np.all(widths) == 0:
        widths += 1

    normwidths = widths / np.linalg.norm(widths)

    if np.any(normwidths < .5) or np.amax(times) > 100:
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'],
                            labelpad=45)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}",
                           fontsize=plot_config['PLOT_TITLE_FONT_SIZE'], pad=10)
        plt.subplots_adjust(top=0.85, bottom=.2, left=.1, right=.95)
    else:
        invis_ax.set_xlabel("Time from first datapoint onward [Days]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'],
                            labelpad=20)
        invis_ax.set_title(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}",
                           fontsize=plot_config['PLOT_TITLE_FONT_SIZE'], pad=10)
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
                timespace = np.linspace(start, end, 1000)
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
                ax.set_xticks([(start + end) / 2])
                ax.set_xticklabels([round((start + end) / 2, 2)])
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if 0.2 < normwidths[ind] < 0.5:
                N = 2
                if np.amax(np.unique(np.round(np.linspace(start, end - span * 0.15, 2), N), return_counts=True)[1]) > 1:
                    N = 3
                if ind == 0:
                    ax.set_xticks(np.linspace(start, end - span * 0.3, 2).tolist())
                    ax.set_xticklabels(np.round(np.linspace(start, end - span * 0.15, 2), N).tolist())
                else:
                    ax.set_xticks(np.linspace(start + span * 0.3, end - span * 0.3, 2).tolist())
                    ax.set_xticklabels(np.round(np.linspace(start + span * 0.15, end - span * 0.15, 2), N).tolist())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if normwidths[ind] > 0.5:
                N = 2
                if np.amax(np.unique(np.round(np.linspace(start, end - span * 0.15, 4), N), return_counts=True)[1]) > 1:
                    N = 3
                if end > 100:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                if ind == 0:
                    ax.set_xticks(np.linspace(start, end - span * 0.15, 4).tolist())
                    ax.set_xticklabels(np.round(np.linspace(start, end - span * 0.15, 4), N).tolist())
                else:
                    ax.set_xticks(np.linspace(start + span * 0.15, end - span * 0.15, 4).tolist())
                    ax.set_xticklabels(np.round(np.linspace(start + span * 0.15, end - span * 0.15, 4), N).tolist())
            # ax.ticklabel_format(useOffset=False)
    else:
        axs.scatter(times, vels, zorder=5, color=colors[0])
        axs.errorbar(times, vels, yerr=verrs, capsize=3, linestyle='', zorder=1, color=colors[1])

    if custom_saveloc is None:
        if not merged:
            if extravels is None:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation_broken_axis{plot_config['PLOT_FMT']}")
            else:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation_broken_axis_comparison{plot_config['PLOT_FMT']}")
        else:
            if extravels is None:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}_merged/RV_variation_broken_axis{plot_config['PLOT_FMT']}")
            else:
                fig.savefig(
                    f"{general_config['OUTPUT_DIR']}/{gaia_id}_merged/RV_variation_broken_axis_comparison{plot_config['PLOT_FMT']}")
    else:
        fig.savefig(custom_saveloc)

    if plot_config['SHOW_PLOTS']:
        plt.show()
    else:
        plt.close()


def plot_rvcurve(vels, verrs, times, gaia_id, merged=False, custom_saveloc=None):
    plt.close()
    fig, ax1 = plt.subplots(figsize=(4.8 * 16 / 9, 4.8))
    fig.suptitle(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}", fontsize=plot_config['PLOT_TITLE_FONT_SIZE'])
    ax1.set_ylabel("Radial Velocity [km/s]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'])
    ax1.set_xlabel("Time from first datapoint onward [Days]", fontsize=plot_config['PLOT_LABELS_FONT_SIZE'])

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

        if not custom_saveloc:
            if not merged:
                fig.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation{plot_config['PLOT_FMT']}", dpi=300)
            else:
                fig.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}_merged/RV_variation{plot_config['PLOT_FMT']}",
                            dpi=300)
        else:
            fig.savefig(f"custom_saveloc",
                        dpi=300)

        if plot_config['SHOW_PLOTS']:
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
    if not custom_saveloc:
        if not merged:
            fig.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation{plot_config['PLOT_FMT']}", dpi=300)
        else:
            fig.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}_merged/RV_variation{plot_config['PLOT_FMT']}",
                        dpi=300)
    else:
        fig.savefig(f"custom_saveloc",
                    dpi=300)

    if plot_config['SHOW_PLOTS']:
        plt.show()
    else:
        plt.close()


def append_pdf(pdf, merger):
    try:
        merger.append(pdf)
    except PyPDF2.errors.PdfReadError:
        print(f"The file {pdf} is broken!")


def create_pdf():
    result_params = pd.read_csv("result_parameters.csv")
    result_params["source_id"] = result_params["source_id"].astype("U20")
    plotpdf_exists = os.path.exists("all_RV_plots.pdf")
    plotpdfbroken_exists = os.path.exists("all_RV_plots_broken_axis.pdf")

    if not plotpdf_exists or not plotpdfbroken_exists:
        dirname = os.path.dirname(__file__)
        dirs = [f.path for f in os.scandir(os.path.join(dirname, f"{general_config['OUTPUT_DIR']}")) if f.is_dir()]
        files = [os.path.join(d, f"RV_variation{plot_config['PLOT_FMT']}") for d in dirs if
                 os.path.join(d, f"RV_variation{plot_config['PLOT_FMT']}")]
        files_brokenaxis = [os.path.join(d, f"RV_variation_broken_axis{plot_config['PLOT_FMT']}") for d in dirs if
                            os.path.isfile(os.path.join(d, f"RV_variation_broken_axis{plot_config['PLOT_FMT']}"))]

        if os.name == "nt":
            files.sort(key=lambda f: result_params.index[result_params["source_id"] == f.split("\\")[-2]])
            files_brokenaxis.sort(
                key=lambda f: result_params.index[result_params["source_id"] == f.split("\\")[-2]].tolist()[0])
        else:
            files.sort(key=lambda f: result_params.index[result_params["source_id"] == f.split("/")[-2]])
            files_brokenaxis.sort(
                key=lambda f: result_params.index[result_params["source_id"] == f.split("/")[-2]].tolist()[0])

        if plot_config['PLOT_FMT'].lower() != ".pdf":
            if not plotpdf_exists:
                with open("all_RV_plots.pdf", "wb") as f:
                    f.write(img2pdf.convert(files))
            if not plotpdfbroken_exists:
                with open("all_RV_plots_broken_axis.pdf", "wb") as f:
                    f.write(img2pdf.convert(files_brokenaxis))
        else:
            from PyPDF2 import PdfMerger
            if not plotpdfbroken_exists:
                try:
                    merger = PdfMerger()
                    [append_pdf(pdf, merger) for pdf in files_brokenaxis]
                except OSError:
                    merger.close()

                    os.mkdir("temp")
                    subsets = np.array_split(files_brokenaxis, int(len(files_brokenaxis) / 10))
                    tempfiles = []

                    for i, s in enumerate(subsets):
                        merger = PdfMerger()
                        [append_pdf(pdf, merger) for pdf in s]
                        fstr = f"temp/{i}"
                        tempfiles.append(fstr)
                        with open(fstr, "wb") as t_file:
                            merger.write(t_file)

                    merger = PdfMerger()
                    [append_pdf(pdf, merger) for pdf in tempfiles]

                with open("all_RV_plots_broken_axis.pdf", "wb") as new_file:
                    merger.write(new_file)

                if os.path.isdir("temp"):
                    shutil.rmtree("temp", ignore_errors=True)


def initial_variables():
    global catalogue
    catalogue = files_from_catalogue(general_config["CATALOGUE"])
    if not general_config["VERBOSE"]:
        warnings.filterwarnings("ignore")


def main_loop(gaia_id, configs=None):
    if configs is not None:
        global general_config
        global fit_config
        global plot_config
        general_config, fit_config, plot_config = configs
        global catalogue
        catalogue = files_from_catalogue(general_config["CATALOGUE"])
        if not general_config["VERBOSE"]:
            warnings.filterwarnings("ignore")

    if os.path.isfile(f'{general_config["OUTPUT_DIR"]}/{gaia_id}') and not plot_config['REDO_STARS']:
        if general_config["post_progress"]:
            general_config["queue"].put((1, 100 * (1 / len(catalogue)), "progressbar"))
            general_config["queue"].put((1, "Overall Progress", "text"))
        return

    star = catalogue.loc[catalogue["source_id"] == gaia_id]
    file_prefixes = star["file"].iloc[0]

    fileset = open_spec_files(general_config["FILE_LOC"], file_prefixes)
    if os.path.isdir(f'{general_config["OUTPUT_DIR"]}/{gaia_id}') and not plot_config['REDO_STARS']:
        if os.path.isfile(
                f'{general_config["OUTPUT_DIR"]}/{gaia_id}/RV_variation_broken_axis{plot_config["PLOT_FMT"]}'):
            if not plot_config['SAVE_SINGLE_IMGS']:
                if general_config["post_progress"]:
                    general_config["queue"].put((1, 100 * (1 / len(catalogue)), "progressbar"))
                    general_config["queue"].put((1, "Overall Progress", "text"))
                return
            if not plot_config['REDO_IMAGES']:
                if general_config["post_progress"]:
                    general_config["queue"].put((1, 100 * (1 / len(catalogue)), "progressbar"))
                    general_config["queue"].put((1, "Overall Progress", "text"))
                return
            nspec = len(fileset)
            if os.listdir(
                    f'{general_config["OUTPUT_DIR"]}/{gaia_id}/{str(nspec) if len(str(nspec)) != 1 else "0" + str(nspec)}/'):
                if general_config["post_progress"]:
                    general_config["queue"].put((1, 100 * (1 / len(catalogue)), "progressbar"))
                    general_config["queue"].put((1, "Overall Progress", "text"))
                return
    spectimes = []
    spectimes_mjd = []
    culumvs = []
    culumvs_errs = []
    cumulative_output_table = output_table_cols.copy()
    spec_class = star["SPEC_CLASS"].iloc[0]
    for file in fileset:
        print_status(file, fileset, gaia_id, star["file"].iloc[0])
        wl, flx, time, flx_std = load_spectrum(file)
        if np.sum(flx < 0) / len(flx) > general_config["SORT_OUT_NEG_FLX"]:
            continue
        # TODO: remove this, keep only cumulative and streamline that, put func def within
        # TODO: LOGGING!!
        if general_config["SUBDWARF_SPECIFIC_ADJUSTMENTS"]:
            if "He-" in spec_class:
                # print(f'[{gaia_id}] He-dwarf detected, adjusting accordingly.')
                preexcluded_lines = []
                for key, value in lines_to_fit.items():
                    if "H_" in key:
                        preexcluded_lines.append(value)
                culumv, culumv_errs, output_table_spec, success = process_spectrum(file, gaia_id, fileset.index(file) + 1, exclude_lines=preexcluded_lines)
            else:
                culumv, culumv_errs, output_table_spec, success = process_spectrum(file, gaia_id, fileset.index(file) + 1)
        else:
            culumv, culumv_errs, output_table_spec, success = process_spectrum(file, gaia_id, fileset.index(file) + 1)
        if not success:
            continue
        cumulative_output_table = pd.concat([cumulative_output_table, output_table_spec], axis=0)
        culumvs.append(culumv)
        culumvs_errs.append(culumv_errs)
        spectimes.append(time.to_datetime())
        spectimes_mjd.append(time.mjd)

    with np.printoptions(linewidth=10000):
        if not os.path.isdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}"):
            os.mkdir(f"{general_config['OUTPUT_DIR']}/{gaia_id}")
        cumulative_output_table.to_csv(f"{general_config['OUTPUT_DIR']}/{gaia_id}/culum_spec_vals.csv", index=False)

    culumvs = np.array(culumvs) / 1000
    culumvs_errs = np.array(culumvs_errs) / 1000

    rvtable = pd.DataFrame({
        "culum_fit_RV": culumvs,
        "u_culum_fit_RV": culumvs_errs,
        "mjd": spectimes_mjd
    })
    with np.printoptions(linewidth=10000):
        rvtable.to_csv(f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation.csv", index=False)
    if plot_config['SAVE_COMPOSITE_IMG']:
        plot_rvcurve_brokenaxis(culumvs, culumvs_errs, spectimes_mjd, gaia_id)
    if general_config["post_progress"]:
        general_config["queue"].put((current_process()._identity[0], 0, "progressbar"))
        general_config["queue"].put((current_process()._identity[0], gaia_id, "text"))
        general_config["queue"].put((1, 100 * (1 / len(catalogue)), "progressbar"))
        general_config["queue"].put((1, "Overall Progress", "text"))


def interactive_main(configs, queue):
    print("Starting interactive process.")
    configs[0]["queue"] = queue
    configs[0]["post_progress"] = True
    if not os.path.isdir(f'{configs[0]["OUTPUT_DIR"]}'):
        os.mkdir(f"{configs[0]['OUTPUT_DIR']}")
    if not configs[0]["VERBOSE"]:
        warnings.filterwarnings("ignore")
    print("Loading catalogue...")
    catalogue = files_from_catalogue(configs[0]["CATALOGUE"])

    print("Starting the fitting process...")
    with Pool(initializer=initial_variables) as pool:
        pool.starmap(main_loop, zip(catalogue["source_id"].to_numpy(), itertools.repeat(configs)))

    print("Fits are completed, analysing results...")
    result_analysis(catalogue, config=configs[0])
    if configs[2]['CREATE_PDF']:
        create_pdf()
    print("All done!")
    queue.put(["done", None, None])


############################## EXECUTION ##############################


if __name__ == "__main__":
    print("Starting process.")
    if not os.path.isdir(f'{general_config["OUTPUT_DIR"]}'):
        os.mkdir(f"{general_config['OUTPUT_DIR']}")
    if not general_config["VERBOSE"]:
        warnings.filterwarnings("ignore")
    print("Loading catalogue...")
    catalogue = files_from_catalogue(general_config["CATALOGUE"])

    print("Starting the fitting process...")
    pool = Pool(initializer=initial_variables)
    pool.map(main_loop, catalogue["source_id"].to_numpy())

    print("Fits are completed, analysing results...")
    result_analysis(catalogue, config=general_config)
    if plot_config['CREATE_PDF']:
        create_pdf()
    print("All done!")
