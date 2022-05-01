# TODO: Fit single Peak by doing continuum fit and gauss-Lorentz fit
# TODO: Error Estimation via monte-Carlo method

import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.special import erf
from astropy.io import fits

FILENAME = "sdss_spec.fits"
PLOTOVERVIEW = False

lines = {
    "H_alpha": 6562,
    "H_beta": 4861,
    "H_gamma": 4340,
    "H_delta": 4101,
    "He_I_4472": 4472,
    "He_I_4922": 4922,
    "He_II_4541": 4541,
    "He_II_4686": 4686,
    "He_II_5412": 5412
}


class NoiseWarning(UserWarning):
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


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    # z = (x + 1.j * gamma) / (sigma * np.sqrt(2))
    # return -scaling * (np.real(faddeeva(z)) / (sigma * np.sqrt(2 * np.pi))) + slope * (x - shift) + height
    return scaling * (
            (1 - eta) / np.pi * (-gamma / (gamma ** 2 + (x - shift) ** 2)) - eta / (
        np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -(x - shift) ** 2 / (2 * sigma ** 2))) + slope * x + height


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
    return wavelength, flux


def verify_peak(wavelength, sigma):
    d_wl = wavelength[1] - wavelength[0]
    if sigma < d_wl:
        return False
    else:
        return True


def calc_SNR(params, flux, wavelength, margin):
    scaling, gamma, shift, slope, height, eta = params
    wlmargin = np.abs(wavelength[0] - wavelength[margin])
    flux = flux - slope * wavelength - height
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))

    slicedwl, loind, upind = slicearr(wavelength, shift - 2 * sigma, shift + 2 * sigma)
    signalstrength = np.mean(np.square(flux[loind:upind]))

    slicedwl, lloind, lupind = slicearr(wavelength, shift - wlmargin, shift - 2 * sigma)
    slicedwl, uloind, uupind = slicearr(wavelength, shift + 2 * sigma, shift + wlmargin)

    noisestrength = np.mean(np.square(np.array(flux[lloind:lupind].tolist() + flux[uloind:uupind].tolist())))

    SNR = 10 * np.log10(signalstrength / noisestrength)

    return signalstrength, noisestrength, SNR


# def calc_SNR(params, flux, wavelength, margin):
#     scaling, gamma, shift, slope, height, eta = params
#     wlmargin = np.abs(wavelength[0] - wavelength[margin])
#     flux = np.abs(flux - slope * wavelength - height)
#     sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
#
#     slicedwl, loind, upind = slicearr(wavelength, shift - sigma, shift+sigma)
#     signalstrength = simpson(flux[loind:upind], slicedwl)/(2*sigma)
#
#     # slicedwl, loind, upind = slicearr(wavelength, shift - sigma, shift + sigma)
#     # signalstrength = np.abs(simpson(pseudo_voigt(slicedwl, scaling, gamma, shift, 0, 0, eta), slicedwl)) / (
#     #         2 * sigma)
#
#     slicedwl, loind, upind = slicearr(wavelength, shift - wlmargin, shift - sigma)
#     noisestrength = simpson(flux[loind:upind], slicedwl)
#     first_sec_width = slicedwl[-1] - slicedwl[0]
#
#     slicedwl, loind, upind = slicearr(wavelength, shift + sigma, shift + wlmargin)
#     noisestrength += simpson(flux[loind:upind], slicedwl)
#
#     noise_width = first_sec_width + slicedwl[-1] - slicedwl[0]
#     noisestrength /= noise_width
#
#     print(noise_width, 2 * sigma)
#
#     print(signalstrength, noisestrength, signalstrength / noisestrength, 10 * np.log10(signalstrength / noisestrength))
#     print("")
#     return signalstrength, noisestrength


def plot_peak_region(wavelength, flux, center, margin, fit):
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
                                     [initial_s, 1, center, 0, initial_h, 0.5],
                                     bounds=(
                                         [0, 0, center - margin / 2, -np.inf, 0, 0],
                                         [np.inf, np.inf, center + margin / 2, np.inf, np.inf, 1]
                                     )
                                     )
            errs = np.sqrt(np.diag(errs))

            print("Wavelenght", center, ":", params, end="\n\n")
            # plt.plot(slicedwl, pseudo_voigt(slicedwl, initial_s, 1, 1, center, 0, initial_h, 0.5))
            if not verify_peak(wavelength, params[1] / (2 * np.sqrt(2 * np.log(2)))):
                plt.figtext(0.3, 0.95, f"FIT SEEMS INACCURATE!",
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            color="red")
                warnings.warn(f"WL {center}Å: Peak seems to be fitted to Noise, SNR and peak may not be accurate.",
                              NoiseWarning)
            sstr, nstr, SNR = calc_SNR(params, flux, wavelength, margin)
            plt.annotate(f"Signal to Noise Ratio: {round(SNR, 2)}dB ", (10, 10), xycoords="figure pixels")
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
    plt.savefig(f"output/plot_{center}Å", dpi=250)
    plt.show()
    return sucess


if __name__ == "__main__":
    wl, flx = load_spectrum(FILENAME)
    for lstr, loc in lines.items():
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.title(f"Fit for Line {lstr} @ {loc}Å")
        plot_peak_region(wl, flx, loc, 100, True)
    # x = np.linspace(-50, 50, 100)
    # plt.plot(x, pseudo_voigt(x, 1, 1, 1, 0, 0, 0))
    # plt.show()
