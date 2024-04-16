import os.path
import sys
import threading

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage import maximum_filter, median_filter

from main import pseudo_voigt, slicearr, lines_to_fit, splitname, fit_config, plot_rvcurve_brokenaxis

############################## OPTIONS ##############################

RES_PARAMETER_LIST = "result_parameters.csv"  # Location of the generated result parameter table

NORMALIZE = True  # Whether to normalize spectra before plotting
MED_WINDOW = 20  # Normalization window size
MAX_WINDOW = 50  # Normalization window size
TRUNC_WL = 25  # Wavelength margin by which to truncate the data before normalization
VET_RESULTS = True  # Whether to enable result verification features
CUTLIM = 15
CUTLINES = [6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 4026.19, 4471.4802, 4921.9313, 5015.678, 5875.6, 6678.15, 4541.59, 4685.70, 5411.52]

SEP = (.5, 0) # Separation value to add to each subseqent spectrum, if normalized at index 0, if not normalized at index 1

INDEX_TO_PLOT = 2  # Index of the system to be plotted in result_parameters.csv if not verifying results
COLORMAP = cm.rainbow  # Optional matplotlib colormap

############################## FUNCTIONS ##############################

try:
    RES_PARAMETER_LIST = pd.read_csv(RES_PARAMETER_LIST)
    RES_PARAMETER_LIST["source_id"] = RES_PARAMETER_LIST["source_id"].astype("U20")
except FileNotFoundError:
    pass

def ind_to_strind(ind):
    return "0" + str(ind) if ind < 10 else str(ind)


def get_params_from_filename(paramtable: pd.DataFrame, gaia_id, subspec_ind):
    paramdict = dict.fromkeys(lines_to_fit)
    subspec = str(subspec_ind)
    for name in lines_to_fit.keys():
        prow = paramtable.loc[paramtable["subspectrum"] == gaia_id + "_" + subspec]
        prow = prow.loc[prow["line_name"] == name]
        if len(prow) != 0:
            paramdict[name] = [prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]]

    # scaling, gamma, shift, slope, height, eta
    return paramdict, gaia_id + "_" + subspec


def normalize_spectrum(wl, flx, std=None):
    wl_step = np.abs(np.median(np.diff(wl)[np.diff(wl) != 0]))
    true_med_size = int(np.floor(MED_WINDOW / wl_step))
    true_max_size = int(np.floor(MAX_WINDOW / wl_step))
    if true_max_size == 0 or true_med_size == 0:
        raise ValueError("Medium/Maximum window sizes need to be bigger than a wavelength step!")
    flx_for_interpol = median_filter(flx, size=true_med_size)
    flx_for_interpol = maximum_filter(flx_for_interpol, size=true_max_size)

    cutmask = np.full(np.shape(wl), 1).astype(bool)
    for l in CUTLINES:
        cutmask = np.logical_and(cutmask, np.logical_or(wl > l + CUTLIM, wl < l - CUTLIM))
    flx_for_interpol = flx_for_interpol[cutmask]
    wl_for_interpol = wl[cutmask]

    # getting unique elements from a first array
    wl_for_interpol, indices = np.unique(wl_for_interpol, return_inverse=True)

    # taking mean of respective duplicate values in the second array
    flx_for_interpol = np.bincount(indices, flx_for_interpol) / np.bincount(indices)

    norm_fit = InterpolatedUnivariateSpline(wl_for_interpol, flx_for_interpol, k=3)

    n_flx = flx / norm_fit(wl)

    if std is None:
        return wl, n_flx, norm_fit
    else:
        n_flx_std = std / norm_fit(wl)
        return wl, n_flx, n_flx_std, norm_fit


def comprehend_lstr(lstr):
    l = [j.strip() for j in lstr.split(";")]
    return l


def plot_system_from_file(source_id=None, normalized=False):
    if source_id is None:
        return
    data = np.genfromtxt(f"master_spectra/{source_id}_stacked.txt")

    wl = data[:, 0]
    flx = data[:, 1]
    p = wl.argsort()
    wl = wl[p]
    flx = flx[p]

    if normalized:
        wl, flx, norm = normalize_spectrum(wl, flx)
        plt.plot(wl, flx, color="navy", zorder=5, linewidth=.5)
        for line in lines_to_fit.values():
            plt.axvline(line, color="darkgrey", linestyle="--", linewidth=.5, zorder=4)
        plt.ylim(0, 1.2)
    else:
        plt.plot(wl, flx, color="navy", zorder=5, linewidth=.5)
        for line in lines_to_fit.values():
            plt.axvline(line, color="darkgrey", linestyle="--", linewidth=.5, zorder=4)

    plt.ylabel("Normalized Flux + Offset")
    plt.xlabel("Wavelength [Å]")
    plt.tight_layout()
    plt.show()
    return


def plot_system_from_ind(ind=INDEX_TO_PLOT, outdir="output", verbose=False, savepath=None, custom_xlim=None, use_ind_as_sid=False, normalized=NORMALIZE):
    global RES_PARAMETER_LIST

    if isinstance(RES_PARAMETER_LIST, str):
        try:
            RES_PARAMETER_LIST = pd.read_csv(RES_PARAMETER_LIST)
            RES_PARAMETER_LIST["source_id"] = RES_PARAMETER_LIST["source_id"].astype("U20")
        except FileNotFoundError:
            raise FileNotFoundError("File result_parameters.csv not found. Please retry processing your spectra.")

    if use_ind_as_sid:
        trow = RES_PARAMETER_LIST[RES_PARAMETER_LIST["source_id"] == ind].iloc[0]
    else:
        trow = RES_PARAMETER_LIST.iloc[ind]

    paramtable = pd.read_csv(f"{outdir}/" + trow["source_id"] + "/culum_spec_vals.csv")

    plt.figure(figsize=(4.8 * 16 / 9, 4.8))
    a_temp = trow["associated_files"]
    a_files = [a_temp] if ";" not in a_temp else comprehend_lstr(a_temp)
    gaia_id = trow["source_id"]

    filelist = []
    k = 0
    for fname in a_files:
        i = 1
        file = fname + "_" + ind_to_strind(i) + ".txt"

        while os.path.isfile("spectra_processed/" + file):
            filelist.append(file)
            i += 1
            file = fname + "_" + ind_to_strind(i) + ".txt"

    color = COLORMAP(np.linspace(0, 1, len(filelist)))

    for i, file in enumerate(filelist):
        i += 1
        data = np.genfromtxt("spectra_processed/" + file)

        wl = data[:, 0]
        flx = data[:, 1]
        p = wl.argsort()
        wl = wl[p]
        flx = flx[p]

        params, name = get_params_from_filename(paramtable, gaia_id, i)

        if normalized:
            wl, flx, norm = normalize_spectrum(wl, flx)
            plt.plot(wl, flx + k, color=color[i - 1], label=name, zorder=5, linewidth=.5)
            if verbose:
                print(name)
            for lname, lloc in lines_to_fit.items():
                if params[lname] is not None:
                    wlforfit = np.linspace(params[lname][2] - fit_config["MARGIN"], params[lname][2] + fit_config["MARGIN"], 250)
                    fit = pseudo_voigt(wlforfit, *params[lname])
                    fit /= norm(wlforfit)
                    plt.plot(wlforfit, fit + k, color="black", zorder=6, linewidth=.5)
            for line in lines_to_fit.values():
                plt.axvline(line, color="darkgrey", linestyle="--", linewidth=.5, zorder=4)
            plt.ylim(0, k + 2)
        else:
            plt.plot(wl, flx + k, color=color[i - 1], label=name, zorder=5, linewidth=.5)
            for lname, lloc in lines_to_fit.items():
                if params[lname] is not None:
                    wlforfit = np.linspace(params[lname][2] - fit_config["MARGIN"], params[lname][2] + fit_config["MARGIN"], 250)
                    fit = pseudo_voigt(wlforfit, *params[lname])
                    plt.plot(wlforfit, fit + k, color="black", zorder=6, linewidth=.5)
            for line in lines_to_fit.values():
                plt.axvline(line, color="darkgrey", linestyle="--", linewidth=.5, zorder=4)

        k += 1*SEP[0] if normalized else 1*SEP[1]

    if custom_xlim:
        plt.xlim(custom_xlim)
    plt.legend(fontsize=3)
    plt.ylabel("Normalized Flux + Offset")
    plt.xlabel("Wavelength [Å]")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def correct_indiced_sys(gaia_id, outdir="output"):
    gaia_id = str(gaia_id)
    corr = True if input("Do you want to correct this spectrum? [y/n]").lower() == "y" else False
    if corr:
        abort = False
        while not abort:
            try:
                subspec = input("Which Spectrum is the line in? (or -1 to abort)")
                if subspec == "-1":
                    abort = True
                else:
                    try:
                        csvtable = pd.read_csv(f"{outdir}/" + gaia_id + "/culum_spec_vals.csv")
                        rv = csvtable.loc[csvtable.subspectrum != subspec].iloc[0]["RV"] / 1000
                        csvtable = csvtable[csvtable.subspectrum != subspec]
                        csvtable.to_csv(f"{outdir}/" + gaia_id + "/culum_spec_vals.csv", index=False)

                        rvtable = pd.read_csv(f"{outdir}/" + gaia_id + "/RV_variation.csv")
                        rvtable = rvtable[rvtable["culum_fit_RV"].round(2) != round(rv, 2)]
                        rvtable.to_csv(f"{outdir}/" + gaia_id + "/RV_variation.csv", index=False)

                        vels = rvtable["culum_fit_RV"].to_numpy()
                        verrs = rvtable["u_culum_fit_RV"].to_numpy()
                        times = rvtable["mjd"].to_numpy()

                        plot_rvcurve_brokenaxis(vels, verrs, times, gaia_id)
                    except:
                        csvtable = pd.read_csv(f"{outdir}/" + gaia_id + "/culum_spec_vals.csv")
                        csvtable = pd.DataFrame(columns=csvtable.columns)
                        csvtable.to_csv(f"{outdir}/" + gaia_id + "/culum_spec_vals.csv", index=False)
                        rvtable = pd.read_csv(f"{outdir}/" + gaia_id + "/RV_variation.csv")
                        rvtable = pd.DataFrame(columns=csvtable.columns)
                        rvtable.to_csv(f"{outdir}/" + gaia_id + "/RV_variation.csv", index=False)
                        plot_rvcurve_brokenaxis(np.array([]), np.array([]), np.array([]), gaia_id)

            except ValueError:
                print("Invalid input!")

    return None


if __name__ == "__main__":
    # plot_system_from_ind()
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        if len(sys.argv) > 2:
            MED_WINDOW = float(sys.argv[2])
            MAX_WINDOW = float(sys.argv[3])
            if len(sys.argv) == 5:
                d = sys.argv[4]
            else:
                d = "output"
        else:
            d = "output"
    else:
        res_params = pd.read_csv("result_parameters.csv")
        res_params = res_params.sort_values("deltaRV", ascending=False)
        for ind, row in res_params.iterrows():
            d = "output"
            print("   ")
            print(f"Star #{ind}; delta RV: {round(row['deltaRV'])}; GAIA DR3 {row['source_id']}")
            plot_system_from_ind(ind, outdir=d)
            correct_indiced_sys(row['source_id'], outdir=d)
        exit()
    while True:
        print("   ")
        print(f"Star #{n}")
        res_params = pd.read_csv("result_parameters.csv")
        row = res_params.iloc[n]
        plot_system_from_ind(n, outdir=d)
        if not isinstance(row['source_id'], np.int64):
            correct_indiced_sys(row['source_id'].iloc[0], outdir=d)
        else:
            correct_indiced_sys(row['source_id'], outdir=d)
        n += 1
