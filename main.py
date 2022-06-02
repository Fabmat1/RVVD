import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.constants import c
from astropy.io import fits
import astropy.time as atime
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

EXTENSION = ".txt"
CATALOGUE = "selected_objects.csv"
FILE_LOC = "spectra/"
DATA_TYPE = "numeric"  # dict, numeric
VERBOSE = False
OUTLIER_MAX_SIGMA = 2
CUT_MARGIN = 20
MARGIN = 100
DESIREDERR = 50
SHOW_PLOTS = False
PLOTOVERVIEW = False
AUTO_REMOVE_OUTLIERS = True
SAVE_SINGLE_IMGS = False
SAVE_COMPOSITE_IMG = False
NOISE_STD_LIMIT = 1
CHECK_IF_EXISTS = True
MAX_ALLOWED_SNR = 5
COSMIC_RAY_DETECTION_LIM = 3  # minimum times peak height/flux std required to detect cr, minimum times diff
# std required to detect cr

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

wl_splitinds, flux_splitinds, flux_std_splitinds = [0, 0, 0]
linelist = []


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
        newarr = np.array([])
    if len(newarr) == 0:
        if np.array(arr == arr[arr > lower]).all():
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
    return 1 / np.pi * (gamma / 2) / ((x - x_0) ** 2 + (gamma / 2) ** 2)


def gaussian(x, gamma, x_0):
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp((-(x - x_0) ** 2) / (2 * sigma ** 2))


def v_from_doppler(lambda_o, lambda_s):
    """
    :param lambda_o: Observed Wavelength
    :param lambda_s: Source Wavelength
    :return: Radial Velocity calculated from relativistic doppler effect
    """
    return c * (lambda_s ** 2 - lambda_o ** 2) / (lambda_o ** 2 + lambda_s ** 2)


def v_from_doppler_err(lambda_o, lambda_s, u_lambda_o):
    """
    :param lambda_o: Observed Wavelength
    :param lambda_s: Source Wavelength
    :param u_lambda_o: Uncertainty of Observed Wavelength
    :return: Uncertainty for Radial Velocity calculated from relativistic doppler effect
    """
    return c * ((4 * lambda_o * lambda_s ** 2) / (
            (lambda_o ** 2 + lambda_s ** 2) ** 2) * u_lambda_o)


def v_from_doppler_rel(r_factor):
    """
    :param r_factor: Wavelength reduction factor lambda_o/lambda_s
    :return: Radial Velocity calculated from relativistic doppler effect
    """
    return c * (1 - r_factor ** 2) / (1 + r_factor ** 2)


def v_from_doppler_rel_err(r_factor, u_r_factor):
    """
    :param r_factor: Wavelength reduction factor lambda_o/lambda_s
    :param u_r_factor: Uncertainty for wavelength reduction factor
    :return: Uncertainty for Radial Velocity calculated from relativistic doppler effect
    """
    return 4 * c * r_factor / (np.power((r_factor ** 2 + 1), 2)) * u_r_factor


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
        plt.show()
    if "flux_std" not in vars():
        flux_std = np.zeros(np.shape(flux))
    return wavelength, flux, t, flux_std


def verify_peak(wavelength, sigma, params, errs, local_flux):
    """
    :param wavelength: Wavelength array
    :param sigma: Standard deviation of fit function
    :param params: Fit parameters
    :param errs: Fit errors
    :return: Returns False if the peak width is in the order of one step in the wavelength array and does not have a
             significant amplitude relative to the std of the flux
    """
    d_wl = wavelength[1] - wavelength[0]
    if sigma < d_wl:
        if pseudo_voigt_height(errs, params[0], params[5], params[1])[0] > NOISE_STD_LIMIT * np.std(local_flux):
            return True
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
                Signal-to-Noise ratio
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
        if not round(line) - 2 < round(wavelength_pov) < round(line) + 2:
            _, loind, upind = slicearr(wls, line - CUT_MARGIN, line + CUT_MARGIN)
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
        for_std = np.concatenate([normalized_flux[lloind:lupind].compressed(), normalized_flux[uloind:uupind].compressed()])
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


def plot_peak_region(wavelengthdata, fluxdata, flux_stddata, center, margin, file_prefix, sanitize=False,
                     used_cr_inds=[]):
    wavelength = np.copy(wavelengthdata)
    flux = np.copy(fluxdata)
    flux_std = np.copy(flux_stddata)

    for i in disturbing_lines.values():
        if i != center:
            if i - CUT_MARGIN < center + MARGIN or i + CUT_MARGIN > center - MARGIN:
                sanitize = True

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
        flux, cut_flux, mask = sanitize_flux(flux, center, wavelength)
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

        if len(used_cr_inds) == 0:
            cr, cr_ind = cosmic_ray(slicedwl, flux[loind:upind], params, center)
            if cr and np.sum(cr_ind) > 0:
                cr_ind += loind
                cr_true_inds = wavelengthdata.searchsorted(wavelength[cr_ind])
                plt.cla()
                for i in cr_ind:
                    plt.plot(wavelength[i - 1:i + 2], flux[i - 1:i + 2], color="lightgray", label='_nolegend_')
                return plot_peak_region(np.delete(wavelength, cr_ind), np.delete(flux, cr_ind),
                                        np.delete(flux_std, cr_ind), center, margin, file_prefix,
                                        used_cr_inds=cr_true_inds)

        if not verify_peak(slicedwl, params[1] / (2 * np.sqrt(2 * np.log(2))), params, errs, flux[loind:upind]):
            if sucess:
                warn_text = plt.figtext(0.3, 0.95, f"SNR NOT GOOD ENOUGH!",
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
        plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}", (10, 10), xycoords="figure pixels")
        if SNR < MAX_ALLOWED_SNR:
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
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False], \
               [False, False, False], False, False


def pseudo_voigt_height(errs, scaling, eta, gamma, sigma=False, sigma_err=False):
    if not sigma:
        sigma = to_sigma(gamma)
    if not sigma_err:
        sigma_err = to_sigma(errs[1])
    height = scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))
    err = height_err(eta, sigma, gamma, scaling, errs[5], sigma_err, errs[1], errs[0])
    return height, err


def print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc):
    if VERBOSE:
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
    if VERBOSE:
        print("\n\n##################### SPECTRUM RESULTS ######################")
        print(f"Result for Spectrum {os.path.basename(filename)}:")
        print(f"Velocity: [{round(complete_v_shift / 1000, 2)}±{round(v_std / 1000, 2)}]km/s")
        print("#############################################################\n\n")


def check_for_outliers(array):
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    outlierloc = distance_from_mean < OUTLIER_MAX_SIGMA * standard_deviation
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
        sucess, errs, [scaling, gamma, shift, slope, height, eta], [sstr, nstr,
                                                                    SNR], sanitized, cr_ind = plot_peak_region(wl, flx,
                                                                                                               flx_std,
                                                                                                               loc,
                                                                                                               MARGIN,
                                                                                                               file_prefix)
        print_results(sucess, errs, scaling, gamma, shift, eta, lstr, loc)
        if sucess:
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
    if np.invert(outloc).sum() != 0:
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

    v_std = np.sqrt(np.sum(verrs ** 2))
    complete_v_shift = np.mean(velocities)
    print_single_spec_results(complete_v_shift, v_std, filename)

    return complete_v_shift, v_std, time, file_prefix, output_table


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


def cumulative_shift(output_table_spec, file, n=0):
    global linelist
    wl, flx, time, flx_std = load_spectrum(file)
    linelist = output_table_spec["line_loc"]
    flux_sanitized = output_table_spec["sanitized"]
    cr_ind = output_table_spec["cr_ind"]
    subspec_ind = list(output_table_spec["subspectrum"])[0]
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

            wl_dataset.append(wldata)
            flux_dataset.append(flx[loind:upind])
            flux_std_dataset.append(flx_std[loind:upind])

        else:
            sanflx, cut_flux, mask = sanitize_flux(flx, line, wl)
            sanwl = wl[mask]
            sanflx = sanflx.compressed()
            sanflx_std = flx_std[mask]
            wldata, loind, upind = slicearr(sanwl, line - MARGIN, line + MARGIN)
            wl_dataset.append(wldata)
            flux_dataset.append(sanflx[loind:upind])
            flux_std_dataset.append(sanflx_std[loind:upind])

    global wl_splitinds, flux_splitinds, flux_std_splitinds
    wl_splitinds = [len(wldata) for wldata in wl_dataset]
    flux_splitinds = [len(flxdata) for flxdata in flux_dataset]
    flux_std_splitinds = [len(flxstddata) for flxstddata in flux_std_dataset]
    wl_splitinds = [sum(wl_splitinds[:ind + 1]) for ind, wl in enumerate(wl_splitinds)]
    flux_splitinds = [sum(flux_splitinds[:ind + 1]) for ind, flx in enumerate(flux_splitinds)]
    flux_std_splitinds = [sum(flux_std_splitinds[:ind + 1]) for ind, flxstd in enumerate(flux_std_splitinds)]

    linelist = list(linelist)

    rmean = output_table_spec["reduction_factor"].mean()

    p0 = [rmean]
    bounds = [
        [0],
        [np.inf]
    ]
    # scaling, gamma, slope, height, eta
    for i in range(len(wl_dataset)):
        specparamrow = output_table_spec.loc[output_table_spec["line_loc"] == list(linelist)[i]]
        p0 += [
            specparamrow["scaling"][0],
            specparamrow["gamma"][0],
            specparamrow["slope"][0],
            specparamrow["flux_0"][0],
            specparamrow["eta"][0],
        ]
        bounds[0] += [0, 0, -np.inf, 0, 0]
        bounds[1] += [np.inf, np.inf, np.inf, np.inf, 1]

    wl_dataset = np.concatenate(wl_dataset, axis=0)
    flux_dataset = np.concatenate(flux_dataset, axis=0)
    flux_std_dataset = np.concatenate(flux_std_dataset, axis=0)

    flux_std_dataset[flux_std_dataset == 0] = np.mean(
        flux_std_dataset)  # Zeros in the std-dataset will raise exceptions (And are scientifically nonsensical)

    params, errs = curve_fit(
        culum_fit_funciton,
        wl_dataset,
        flux_dataset,
        p0=p0,
        bounds=bounds,
        sigma=flux_std_dataset,
        max_nfev=100000
    )

    errs = np.sqrt(np.diag(errs))

    r_factor = params[0]
    r_factor_err = errs[0]

    culumv = v_from_doppler_rel(r_factor)
    culumv_errs = v_from_doppler_rel_err(r_factor, r_factor_err)

    errs = np.split(np.array(errs)[1:], len(np.array(errs)[1:]) / 5)
    params = np.split(np.array(params)[1:], len(np.array(params)[1:]) / 5)

    wl_dataset = np.split(wl_dataset, wl_splitinds)
    flux_dataset = np.split(flux_dataset, flux_splitinds)
    flux_std_dataset = np.split(flux_std_dataset, flux_std_splitinds)

    for i, paramset in enumerate(params):
        lname = list(output_table_spec['line_name'])[i]
        lines = list(linelist)
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.title(f"cumulative Fit for Line {lname} @ {round(lines[i])}Å")
        plt.axvline(lines[i], linewidth=0.5, color='grey', linestyle='dashed', zorder=1)
        plt.plot(wl_dataset[i], flux_dataset[i])
        wllinspace = np.linspace(wl_dataset[i][0], wl_dataset[i][-1], 1000)
        plt.plot(wllinspace, make_dataset(wllinspace, r_factor, i, paramset))
        if SAVE_SINGLE_IMGS:
            subspec_ind = str(subspec_ind) if len(str(subspec_ind)) != 1 else "0" + str(subspec_ind)
            plt.savefig(f"output/{file_prefix.split('_')[0]}/{subspec_ind}/culum_{round(lines[i])}Å.png", dpi=300)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.clf()
            plt.cla()

        scaling, gamma, slope, height, eta = paramset
        shift = r_factor * lines[i]
        scalingerr, gammaerr, slopeerr, heighterr, etaerr = errs[i]
        shifterr = r_factor_err * lines[i]

        output_table_row = pd.DataFrame({
            "subspectrum": [list(output_table_spec["subspectrum"])[0]],
            "line_name": [lname],
            "line_loc": [lines[i]],
            "height": ["--"],
            "u_height": ["--"],
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
            "cr_ind": [cr_ind]
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
    catalogue = pd.read_csv(cat)
    return [a.split(".")[0] for a in catalogue["file"]], catalogue


def print_status(file, fileset, catalogue):
    if "catalogue" in vars():
        gaia_id = catalogue["source_id"][file_prefixes.index(file_prefix.split('_')[0])]
    else:
        gaia_id = file_prefix
    print(f"Doing Fits for System GAIA EDR3 {gaia_id}  [{(fileset.index(file) + 1)}/{(len(fileset))}]")


if __name__ == "__main__":
    if not VERBOSE:
        warnings.filterwarnings("ignore")
    file_prefixes, catalogue = files_from_catalogue(CATALOGUE)
    for file_prefix in file_prefixes:
        fileset = open_spec_files(FILE_LOC, file_prefix, end=EXTENSION)
        if os.path.isdir(f'output/{file_prefix}') and CHECK_IF_EXISTS:
            if os.path.isfile(f'output/{file_prefix}/RV_variation.png'):
                if not SAVE_SINGLE_IMGS:
                    continue
                nspec = len(fileset)
                if os.listdir(f'output/{file_prefix}/{str(nspec) if len(str(nspec)) != 1 else "0" + str(nspec)}/'):
                    continue
        spectimes = []
        spectimes_mjd = []
        specvels = []
        specverrs = []
        culumvs = []
        culumvs_errs = []
        single_output_table = output_table_cols.copy()
        cumulative_output_table = output_table_cols.copy()
        for file in fileset:
            print_status(file, fileset, catalogue)
            complete_v_shift, v_std, time, file_prefix, output_table_spec = single_spec_shift(file)
            single_output_table = pd.concat([single_output_table, output_table_spec], axis=0)
            culumv, culumv_errs, output_table_spec = cumulative_shift(output_table_spec, file)
            cumulative_output_table = pd.concat([cumulative_output_table, output_table_spec], axis=0)
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
        cumulative_output_table.drop("sanitized", axis=1)
        cumulative_output_table.to_csv(f"output/{file_prefix.split('_')[0]}/culum_spec_vals.csv", index=False)

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
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex="col")
        fig.suptitle(f"Radial Velocity over Time\n Gaia EDR3 {gaia_id}")
        ax1.set_ylabel("Radial Velocity [km/s]")
        ax1.set_title("RV Curve determined by single Fit")
        ax2.set_ylabel("Radial Velocity [km/s]")
        ax2.set_title("RV Curve determined by cumulative Fit")
        ax2.set_xlabel("Date")
        fig.autofmt_xdate()

        ax1.plot_date(spectimes, specvels, xdate=True, zorder=5)
        ax2.plot_date(spectimes, culumvs, xdate=True, zorder=5)

        ax1.errorbar(spectimes, specvels, yerr=specverrs, capsize=3, linestyle='', zorder=1)
        ax2.errorbar(spectimes, culumvs, yerr=culumvs_errs, capsize=3, linestyle='', zorder=1)

        specvels_range = specverrs.min()
        culumvs_range = culumvs_errs.min()

        if not pd.isnull(specvels_range) or pd.isnull(culumvs_range):
            ax1.set_ylim((specvels.min() - 2 * specvels_range, specvels.max() + 2 * specvels_range))
            ax2.set_ylim((culumvs.min() - 2 * culumvs_range, culumvs.max() + 2 * culumvs_range))

        plt.tight_layout()
        plt.savefig(f"output/{file_prefix.split('_')[0]}/RV_variation.png", dpi=300)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close()
