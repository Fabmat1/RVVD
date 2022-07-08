import matplotlib.pyplot as plt
import numpy as np
from main import load_spectrum, slicearr, cosmic_ray
from scipy.signal import savgol_filter


############################## SETTINGS ##############################

SAVGOL_WINSIZE = 100  # Window size of Savitzky-Golay filter [Angström at the start of the wavelength array]
                     # Window size might increase toward higher wavelengths in case of logarithmic spacing!
SAVGOL_ORDER = 4  # Order of Savgol filtering

############################## FUNCTIONS ##############################


def detect_peaks(wl, flx, file_pre):
    _, loind, upind = slicearr(wl, np.amin(wl), np.amin(wl)+SAVGOL_WINSIZE)
    winsize = upind-loind
    if winsize % 2 == 0:
        winsize += 1

    filteredflx = savgol_filter(flx, winsize, SAVGOL_ORDER)
    plt.plot(wl, flx)
    plt.plot(wl, filteredflx)
    plt.show()

    norm_flx = flx-filteredflx
    normstd = np.std(norm_flx)

    peaks = scipy.fi

    plt.plot(wl, norm_flx)
    plt.show()


if __name__ == "__main__":
    wl, flx, _, _ = load_spectrum("spectra/spec-2682-54401-0569_10.txt")
    detect_peaks(wl, flx)
