import ctypes
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.ndimage import gaussian_filter
from main import process_spectrum, lines_to_fit
from numpy.ctypeslib import ndpointer

warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1 / (1 + np.exp(-0.002 * (x - 4600)))


def interpolate_arrays(A, B, d_A, d_B):
    # Calculate weights
    w_A = 1 / d_A
    w_B = 1 / d_B

    # Interpolate arrays A and B
    interpolated = []
    for i in range(len(A)):
        interpolated_value = (w_A * A[i] + w_B * B[i]) / (w_A + w_B)
        interpolated.append(interpolated_value)

    return interpolated


lib = ctypes.CDLL("../lib/determine_radvel")

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64,
                                      ndim=1,
                                      flags="C")

lib.determine_radvel.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1, ctypes.c_int, ctypes.c_int]
lib.determine_radvel.restype = ctypes.c_double


def determine_radvel(wl_array, flx_array, lines_to_fit):
    # Invoke the actual C++ function
    result = lib.determine_radvel(wl_array, flx_array, np.array([6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3888.052, 3835.387]), len(wl_array), len(lines_to_fit))
    return result


def rebin_old(x, y, new_dx):
    bin_centers = np.arange(x.min(), x.max(), new_dx)
    ys = np.zeros_like(bin_centers)

    for i in range(len(bin_centers)):
        if i == 0:
            ys[i] = np.sum(y[x < bin_centers[i]])
            upper_mask = np.logical_and(x > bin_centers[i], x < bin_centers[i + 1])
            upper_y = y[upper_mask]
            upper_weights = np.array([(k - bin_centers[i + 1]) / (bin_centers[i] - bin_centers[i + 1]) for k in x[upper_mask]])
            ys[i] += np.sum(upper_y * upper_weights) / np.sum(upper_weights)
        elif i == len(bin_centers) - 1:
            lower_mask = np.logical_and(x > bin_centers[i - 1], x < bin_centers[i])
            ys[i] = np.sum(y[x > bin_centers[i]])
            lower_y = y[lower_mask]
            lower_weights = np.array([(k - bin_centers[i - 1]) / (bin_centers[i] - bin_centers[i - 1]) for k in x[lower_mask]])
            ys[i] += np.sum(lower_y * lower_weights) / np.sum(lower_weights)
        else:
            inv_diff = 1 / (bin_centers[i] - bin_centers[i + 1])
            inv_diff_two = 1 / (bin_centers[i] - bin_centers[i - 1])
            upper_mask = np.logical_and(x > bin_centers[i], x < bin_centers[i + 1])
            lower_mask = np.logical_and(x > bin_centers[i - 1], x < bin_centers[i])
            upper_y = y[upper_mask]
            upper_weights = np.array([(k - bin_centers[i + 1]) * inv_diff for k in x[upper_mask]])
            lower_y = y[lower_mask]
            lower_weights = np.array([(k - bin_centers[i - 1]) * inv_diff_two for k in x[lower_mask]])
            ys[i] += np.sum(upper_y * upper_weights) / np.sum(upper_weights) + np.sum(lower_y * lower_weights) / np.sum(lower_weights)

    return bin_centers[1:-1], ys[1:-1]


def rebin(x, y, new_dx):
    bin_centers = np.arange(x.min(), x.max(), new_dx)
    bins = np.concatenate([bin_centers, np.array([bin_centers[-1] + new_dx])])

    digitized = np.digitize(x, bins) - 1  # Subtract 1 to make it 0-indexed.
    ys = np.zeros_like(bin_centers[1:-1])

    for i in range(len(bin_centers[1:-1])):
        mask = digitized == i
        if np.any(mask):
            weights = (x[mask] - bin_centers[i]) / (bin_centers[i + 1] - bin_centers[i])
            ys[i] = np.average(y[mask], weights=weights)

    return bin_centers[1:-1], ys


def wl_shift(wl, v):
    return wl + wl * v / (c / 1000)


N_TESTS = 1000

gridinfo = pd.read_csv('/home/fabian/workspace/nemeth_grid/gridinfo.txt', delimiter=" ")
gridinfo["he"] = np.log10(gridinfo["he"])
gridinfo["file"] = gridinfo["file"].str[:-2]
accuracy = np.zeros(N_TESTS)

lines_to_fit = np.array(list(lines_to_fit.values()))

rv_diffs = []
rv_diffs_alt = []
success = []
noise_lvls = []
print("Starting test...")
for i in range(N_TESTS):
    print(f"{i}/{N_TESTS}")
    # print("Generating Values...")
    rv = np.random.uniform(low=-500, high=500)
    # noise_percent = np.random.uniform(low=0, high=0.0001)
    noise_percent = np.random.uniform(low=0.005, high=0.15)

    teff = np.random.uniform(low=20000, high=56000)
    logg = np.random.uniform(low=5.0, high=6.0)
    he = np.random.uniform(low=-5.0, high=-4.0)

    # Filter rows where teff, logg, and he are lower than the generated values
    lower_rows = gridinfo[(gridinfo['teff'] <= teff) & (gridinfo['logg'] <= logg) & (gridinfo['he'] <= he)]
    # Find the closest row from the lower_rows
    if not lower_rows.empty:
        lower_row = lower_rows.loc[(lower_rows[['teff', 'logg', 'he']] - [teff, logg, he]).pow(2).sum(axis=1).idxmin()]
    else:
        lower_row = None

    # Filter rows where teff, logg, and he are higher than the generated values
    higher_rows = gridinfo[(gridinfo['teff'] >= teff) & (gridinfo['logg'] >= logg) & (gridinfo['he'] >= he)]
    # Find the closest row from the higher_rows
    if not higher_rows.empty:
        higher_row = higher_rows.loc[(higher_rows[['teff', 'logg', 'he']] - [teff, logg, he]).pow(2).sum(axis=1).idxmin()]
    else:
        higher_row = None

    # print(lower_row["teff"], lower_row["logg"], lower_row["he"])
    # print(teff, logg, he)
    # print(higher_row["teff"], higher_row["logg"], higher_row["he"])
    # print(teff_dist , logg_dist , he_dist)
    # print(lower_dist, higher_dist)

    # print("Rebinning arrays...")
    if lower_row is not None:
        spec_one = np.loadtxt(f'/home/fabian/workspace/nemeth_grid/tlsdhhe{str(lower_row["teff"])[:2]}k/{lower_row["file"]}.sp')
        spec_one = spec_one[np.logical_and(spec_one[:, 0] > 3500, spec_one[:, 0] < 5500)]
        spec_one[:, 1] *= sigmoid(spec_one[:, 0])

        lo_rebin_wl, lo_rebin_flx = rebin(spec_one[:, 0], spec_one[:, 1], new_dx=0.85)

        lo_rebin_flx /= lo_rebin_flx.max()
        # plt.plot(lo_rebin_wl, lo_rebin_flx, 'ro', label='one')
    if higher_row is not None:
        spec_two = np.loadtxt(f'/home/fabian/workspace/nemeth_grid/tlsdhhe{str(higher_row["teff"])[:2]}k/{higher_row["file"]}.sp')
        spec_two = spec_two[np.logical_and(spec_two[:, 0] > 3500, spec_two[:, 0] < 5500)]
        spec_two[:, 1] *= sigmoid(spec_two[:, 0])
        rebin_spec_two = np.zeros_like(spec_two)

        hi_rebin_wl, hi_rebin_flx = rebin(spec_two[:, 0], spec_two[:, 1], new_dx=0.85)

        hi_rebin_flx /= hi_rebin_flx.max()
        # plt.plot(hi_rebin_wl, hi_rebin_flx, 'go', label='two')

    if lower_row is not None and higher_row is not None:
        teff_dist = (teff - lower_row["teff"]) / (higher_row["teff"] - lower_row["teff"])
        logg_dist = (logg - lower_row["logg"]) / (higher_row["logg"] - lower_row["logg"])
        he_dist = (he - lower_row["he"]) / (higher_row["he"] - lower_row["he"])

        lower_dist = np.sqrt(teff_dist ** 2 + logg_dist ** 2 + he_dist ** 2)
        higher_dist = np.sqrt((1 - teff_dist) ** 2 + (1 - logg_dist) ** 2 + (1 - he_dist) ** 2)

        rebin_wl = interpolate_arrays(lo_rebin_wl, hi_rebin_wl, lower_dist, higher_dist)
        rebin_flx = interpolate_arrays(lo_rebin_flx, hi_rebin_flx, lower_dist, higher_dist)
    elif lower_row is not None:
        rebin_wl = lo_rebin_wl
        rebin_flx = lo_rebin_flx
    else:
        rebin_wl = hi_rebin_wl
        rebin_flx = hi_rebin_flx

    rebin_wl = np.array(rebin_wl)
    rebin_flx = np.array(rebin_flx)

    # print(f"Adding Noise (Noise level {noise_percent})...")
    rebin_flx = gaussian_filter(rebin_flx, 3 * 1 / np.mean(np.diff(rebin_wl)))
    rebin_flx += np.random.normal(0, noise_percent, rebin_flx.shape)
    rebin_wl = wl_shift(rebin_wl, rv)

    det_rv, det_rv_err, _, _ = process_spectrum("", "", 1, [], debug_list=[rebin_wl, rebin_flx, 0, np.full(rebin_wl.shape, noise_percent)])
    det_rv_alt = determine_radvel(rebin_wl, rebin_flx, lines_to_fit)

    if det_rv is not None:
        rv_diffs.append(det_rv / 1000 - rv)
        noise_lvls.append(noise_percent)
        success.append(True)
    else:
        noise_lvls.append(noise_percent)
        success.append(False)
    rv_diffs_alt.append(det_rv_alt - rv)

    data = np.genfromtxt("outvel.txt", delimiter=",")
    # plt.plot(data[:, 0], data[:, 1])
    # plt.axvline(det_rv_alt, color='r', linestyle='--')
    # plt.axvline(rv, color='g', linestyle='--')
    # plt.show()
    # plt.plot(rebin_wl, rebin_flx)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

plt.hist(rv_diffs, bins=50, alpha=0.5, color="navy")
plt.hist(rv_diffs_alt, bins=50, alpha=0.5, color="darkgreen")
plt.tight_layout()
plt.show()
