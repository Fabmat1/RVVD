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


def pseudo_voigt(x, scaling, gamma, shift, slope, height, eta):
    # sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    # z = (x + 1.j * gamma) / (sigma * np.sqrt(2))
    # return -scaling * (np.real(faddeeva(z)) / (sigma * np.sqrt(2 * np.pi))) + slope * (x - shift) + height
    # return scaling * (
    #         (1 - eta) / np.pi * (-gamma / (gamma ** 2 + (x - shift) ** 2))
    #         - eta / (np.sqrt(2 * np.pi * sigma ** 2))
    #         * np.exp(-(x - shift) ** 2 / (2 * sigma ** 2))
    # ) + slope * x + height
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
                                     [initial_s, 5, center, 0, initial_h, 0.5],
                                     # scaling, gamma, shift, slope, height, eta
                                     bounds=(
                                         [0, 0, center - 25, -np.inf, 0, 0],
                                         [np.inf, np.sqrt(2 * np.log(2)) * margin / 4, center + 25, np.inf, np.inf, 1]
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
    plt.savefig(f"output/plot_{round(center)}Å", dpi=500)
    plt.show()
    if sucess:
        return sucess, errs, params
    else:
        return sucess, [False, False, False, False, False, False], [False, False, False, False, False, False]


if __name__ == "__main__":
    wl, flx = load_spectrum(FILENAME)
    for lstr, loc in lines.items():
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        plt.title(f"Fit for Line {lstr} @ {round(loc)}Å")
        sucess, errs, [scaling, gamma, shift, slope, height, eta] = plot_peak_region(wl, flx, loc, 100, True)
        if sucess:
            sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
            print("######################## FIT RESULTS ########################")
            print(f"Result for line {lstr} @ {round(loc)}Å:")
            print(f"\nPeak Height I={scaling * (eta / (sigma * np.sqrt(2 * np.pi)) + (1 - eta) * 2 / (np.pi * gamma))}")
            print(f"Standard deviation σ={sigma}")
            print(f"Peak location x_0={shift}")
            print("#############################################################\n\n")
        else:
            sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
            print("######################## FIT RESULTS ########################")
            print(f"Result for line {lstr} @ {round(loc)}Å:")
            print(f"FIT FAILED!")
            print("#############################################################\n\n")
