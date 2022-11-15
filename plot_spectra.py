import os.path
import matplotlib.colors as mcolor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from scipy.interpolate import interp1d

from main import pseudo_voigt, slicearr, load_spectrum, lines_to_fit, expand_mask, splitname, MARGIN, plot_rvcurve_brokenaxis

############################## OPTIONS ##############################

RES_PARAMETER_LIST = "result_parameters.csv"  # Location of the generated result parameter table

NORMALIZE = True  # Whether to normalize spectra before plotting
NORM_WINDOW = 30  # Normalization window size
TRUNC_WL = 25  # Wavelength margin by which to truncate the data before normalization
VET_RESULTS = True  # Wheter to enable result verification features

INDEX_TO_PLOT = 2  # Index of the system to be plotted in result_parameters.csv if not verifying results
COLORMAP = cm.rainbow  # Optional matplotlib colormap

############################## FUNCTIONS ##############################

RES_PARAMETER_LIST = pd.read_csv(RES_PARAMETER_LIST)


def ind_to_strind(ind):
    return "0" + str(ind) if ind < 10 else str(ind)


def get_params_from_filename(filename, paramtable: pd.DataFrame, simple=True):
    paramdict = dict.fromkeys(lines_to_fit)

    specname, subspec = splitname(filename.split("/")[1].split(".")[0])
    for name in lines_to_fit.keys():
        if simple:
            subspec = int(subspec)
            prow = paramtable.loc[paramtable["subspectrum"] == subspec]
            prow = prow.loc[prow["line_name"] == name]
            if len(prow) != 0:
                paramdict[name] = [prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]]
        else:
            subspec = str(int(subspec))
            prow = paramtable.loc[paramtable["subspectrum"] == subspec + "_" + specname]
            prow = prow.loc[prow["line_name"] == name]
            if len(prow) != 0:
                paramdict[name] = [prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]]
    # scaling, gamma, shift, slope, height, eta
    if simple:
        return paramdict, subspec
    else:
        return paramdict, subspec + "_" + specname


def normalize_spectrum(wl, flx, window_size=NORM_WINDOW):
    wl, loind, upind = slicearr(wl, np.amin(wl) + TRUNC_WL, np.amax(wl) - TRUNC_WL)
    flx = flx[loind:upind]

    if len(wl) != len(flx):
        wl = wl[0:len(flx)]

    wl_for_interpol = np.copy(wl)
    flx_for_interpol = np.copy(flx)

    while len(flx_for_interpol) % window_size != 0:
        wl_for_interpol = wl_for_interpol[:-1]
        flx_for_interpol = flx_for_interpol[:-1]

    window = len(flx_for_interpol) / window_size

    flxs = np.split(flx_for_interpol, window)
    maxima = []

    for i, sub in enumerate(flxs):
        flxdiff = np.diff(sub, prepend=flx[0])
        threesig = np.std(flxdiff)

        mask = flxdiff < threesig
        mask = np.logical_and(np.logical_and(mask, np.roll(mask, 1)), np.roll(mask, -1))

        sub = sub[mask]

        maxima.append((i * window_size) + np.argmax(sub))

    flx_for_interpol = np.take(flx_for_interpol, maxima)
    wl_for_interpol = np.take(wl_for_interpol, maxima)

    wl_for_interpol = wl_for_interpol.squeeze()
    flx_for_interpol = flx_for_interpol.squeeze()

    for line in lines_to_fit.values():
        mask = np.logical_and(wl_for_interpol < line + 20, wl_for_interpol > line - 20)
        wl_for_interpol = wl_for_interpol[~mask]
        flx_for_interpol = flx_for_interpol[~mask]

    f = interp1d(wl_for_interpol, flx_for_interpol, "linear", fill_value="extrapolate")
    flx /= f(wl)

    return wl, flx, f


def comprehend_lstr(lstr):
    l = [j.strip() for j in lstr.replace("[", "").replace("]", "").replace("'", "").split(",")]
    return l


def plot_system_from_ind(ind=INDEX_TO_PLOT, res_params=RES_PARAMETER_LIST):
    trow = RES_PARAMETER_LIST.iloc[ind]
    paramtable = pd.read_csv("output/" + trow["source_folder"] + "/culum_spec_vals.csv")

    a_temp = trow["associated_files"]
    a_files = [a_temp] if "," not in a_temp else comprehend_lstr(a_temp)

    k = 1
    for fname in a_files:
        i = 1
        file = fname + "_" + ind_to_strind(i) + ".txt"
        filelist = []

        while os.path.isfile("spectra/" + file):
            filelist.append(file)
            i += 1
            file = fname + "_" + ind_to_strind(i) + ".txt"

        color = COLORMAP(np.linspace(0, 1, len(filelist)))
        for i, file in enumerate(filelist):
            data = np.genfromtxt("spectra/" + file)

            wl = data[:, 0]
            flx = data[:, 1]

            params, name = get_params_from_filename("spectra/" + file, paramtable, True if len(a_files) == 1 else False)

            if NORMALIZE:
                wl, flx, norm = normalize_spectrum(wl, flx)
                plt.plot(wl, flx + k, color=color[i - 1], label=name, zorder=5)
                print(name)
                for lname, lloc in lines_to_fit.items():
                    if params[lname] is not None:
                        wlforfit = np.linspace(params[lname][2] - MARGIN, params[lname][2] + MARGIN, 250)
                        fit = pseudo_voigt(wlforfit, *params[lname])
                        fit /= norm(wlforfit)
                        plt.plot(wlforfit, fit + k, color="black", zorder=6)
                for line in lines_to_fit.values():
                    plt.axvline(line, color="darkgrey", linestyle="--", linewidth=1, zorder=4)
            k += 1

    plt.ylim(0, k+0.2)
    plt.legend(fontsize=3)
    plt.ylabel("Normalized Flux + Offset")
    plt.xlabel("Wavelength [Å]")
    plt.tight_layout()
    plt.show()
    return None


def correct_indiced_sys(ind):
    corr = True if input("Do you want to correct this spectrum? [y/n]").lower() == "y" else False
    if corr:
        abort = False
        while not abort:
            try:
                subspec = input("Which Spectrum is the line in? (or -1 to abort)")
                if subspec == "-1":
                    abort = True
                else:
                    trow = RES_PARAMETER_LIST.iloc[ind]
                    csvtable = pd.read_csv("output/" + trow["source_folder"] + "/culum_spec_vals.csv")
                    rv = csvtable.loc[csvtable.subspectrum != subspec].iloc[0]["RV"] / 1000
                    csvtable = csvtable[csvtable.subspectrum != subspec]
                    csvtable.to_csv("output/" + trow["source_folder"] + "/culum_spec_vals.csv", index=False)

                    rvtable = pd.read_csv("output/" + trow["source_folder"] + "/RV_variation.csv")
                    rvtable = rvtable[rvtable["culum_fit_RV"].round(3) != round(rv, 3)]
                    rvtable.to_csv("output/" + trow["source_folder"] + "/RV_variation.csv", index=False)

                    vels = rvtable["culum_fit_RV"].to_numpy()
                    verrs = rvtable["u_culum_fit_RV"].to_numpy()
                    times = rvtable["mjd"].to_numpy()
                    fpre = trow["source_folder"]
                    gaia_id = trow["source_id"]

                    plot_rvcurve_brokenaxis(vels, verrs, times, fpre, gaia_id)

            except ValueError:
                print("Invalid input!")

    return None


if __name__ == "__main__":
    # plot_system_from_ind()
    n = 23
    while True:
        print("   ")
        print(f"Star #{n}")
        plot_system_from_ind(n)
        correct_indiced_sys(n)
        n += 1
