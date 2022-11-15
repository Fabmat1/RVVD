import os.path

import matplotlib.colors as mcolor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from main import pseudo_voigt, slicearr, load_spectrum, lines_to_fit, expand_mask, splitname, MARGIN

SPECFILE_NAME = ['spec-56572-GAC077N35B1_sp12-222', 'spec-57064-GAC082N38B2_sp10-181']
GAIA_ID = "944390774983674496"
TITLE = ""
SUBSPEC = "all"
LINE_LOC = "all"
PLOT_OVERVIEW = True
OVERVIEW_SEP = .25
PLOT_INDIVIDUAL = False
PLOT_FITTED_LINE = True
CULUMFIT = True
COLORS = ["navy", "crimson"]
MANUALLIMITS = False
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 15
SUBSPECCOLORS = ["navy"]
MARK_LINE_LOCATIONS = True
XLIM = None
YLIM = None

plt.rcParams["figure.figsize"] = (4.8 * 16 / 9, 4.8)
plt.rcParams["figure.dpi"] = 300

filenames = []


def get_params_from_filename(filename, paramtable: pd.DataFrame, sfilename=None):
    paramdict = dict.fromkeys(lines_to_fit)
    if sfilename is None:
        sfilename = SPECFILE_NAME
    specname, subspec = splitname(filename.split("/")[1].split(".")[0])
    for name in lines_to_fit.keys():
        if type(sfilename) == list:
            subspec = str(int(subspec))
            prow = paramtable.loc[paramtable["subspectrum"] == subspec + "_" + specname]
            prow = prow.loc[prow["line_name"] == name]
            if len(prow) != 0:
                paramdict[name] = [prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]]
        else:
            subspec = int(subspec)
            prow = paramtable.loc[paramtable["subspectrum"] == subspec]
            prow = prow.loc[prow["line_name"] == name]
            if len(prow) != 0:
                paramdict[name] = [prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]]
    # scaling, gamma, shift, slope, height, eta
    return paramdict


if __name__ == "__main__":
    if type(SUBSPEC) == str:
        if SUBSPEC.lower() == "all":
            n = 1
            if type(SPECFILE_NAME) == str:
                while True:
                    filename = f"spectra/{SPECFILE_NAME}_{n if len(str(n)) > 1 else '0' + str(n)}.txt"
                    if os.path.isfile(filename):
                        n += 1
                        filenames.append(filename)
                    else:
                        break
            if type(SPECFILE_NAME) == list:
                for spfile in SPECFILE_NAME:
                    while True:
                        filename = f"spectra/{spfile}_{n if len(str(n)) > 1 else '0' + str(n)}.txt"
                        if os.path.isfile(filename):
                            n += 1
                            filenames.append(filename)
                        else:
                            break
                    n = 1


    else:
        for specn in SUBSPEC:
            if len(SPECFILE_NAME) > 0:
                filenames.append(f"spectra/{SPECFILE_NAME}_{specn if len(str(specn)) > 1 else '0' + str(specn)}.txt")
            else:
                spectra_table = pd.read_csv("selected_objects.csv")
                spectra_table = spectra_table.loc[spectra_table["source_id"] == GAIA_ID]
                filenames.append(
                    f"spectra/{spectra_table['file'][0]}_{specn if len(str(specn)) > 1 else '0' + str(specn)}.txt")

    if type(SPECFILE_NAME) == list:
        for f in SPECFILE_NAME:
            if os.path.isdir(f"output/{f}_merged"):
                csvpath = f"output/{f}_merged/culum_spec_vals.csv"
                break
    else:
        csvpath = f"output/{SPECFILE_NAME}/culum_spec_vals.csv"
    culumfit_table = pd.read_csv(csvpath)
    RV_table = pd.read_csv(csvpath.replace("culum_spec_vals", "RV_variation"))
    RV_vals = RV_table["culum_fit_RV"].to_numpy()
    RV_vals -= np.amin(RV_vals)
    RV_vals /= np.amax(RV_vals)

    # cmap = mcolor.LinearSegmentedColormap.from_list('blue_to_red', ['darkblue', 'gray', 'darkred'])

    # color = cmap(RV_vals)
    # fit_colors = color[:, 0:3] * 0.6
    color = cm.rainbow(np.linspace(0, 1, len(filenames)))
    if PLOT_OVERVIEW:
        for line in lines_to_fit.values():
            if MARK_LINE_LOCATIONS:
                plt.axvline(line, color="darkgrey", linestyle="--", linewidth=1)
        for ind, filename in enumerate(filenames):
            params = get_params_from_filename(filename, culumfit_table)
            wl, flux, _, flux_std = load_spectrum(filename)
            plt.plot(wl, flux + OVERVIEW_SEP * ind * np.mean(flux), color=color[ind])  # color="navy" if np.amax(wl) < 10000 else "darkred")
            for lname, lloc in lines_to_fit.items():
                if params[lname] is not None and PLOT_FITTED_LINE:
                    wlforfit = np.linspace(params[lname][2] - MARGIN, params[lname][2] + MARGIN, 250)
                    plt.plot(wlforfit, pseudo_voigt(wlforfit, *params[lname]) + OVERVIEW_SEP * ind * np.mean(flux), color="black")
        if GAIA_ID is None:
            plt.title(f"Overview of {SPECFILE_NAME}")  # , fontsize=TITLE_SIZE)
        else:
            plt.title(fr"Vicinity of the $H_\beta$ line for different spectra of GAIA EDR3 {GAIA_ID}")
        plt.ylabel("Flux [ergs/s/cm^2/Å] + Offset")  # , fontsize=LABEL_SIZE)
        plt.xlabel("Wavelength [Å]")  # , fontsize=LABEL_SIZE)
        plt.xticks()  # fontsize=TICK_SIZE)
        plt.yticks()  # fontsize=TICK_SIZE)
        if XLIM is not None:
            plt.xlim(XLIM)
        if YLIM is not None:
            plt.ylim(YLIM)
        plt.tight_layout()
        if GAIA_ID is None:
            plt.savefig(f"images/{SPECFILE_NAME}_overviewplot.png")
        else:
            plt.savefig(f"images/{GAIA_ID}_overviewplot.pdf")
        plt.show()

    if PLOT_INDIVIDUAL:
        for i, filename in enumerate(filenames):
            if type(SUBSPEC) == list:
                nspec = SUBSPEC[filenames.index(filename)]
            else:
                nspec = i + 1
            wl, flux, _, flux_std = load_spectrum(filename)
            if type(LINE_LOC) == str:
                if LINE_LOC.lower() == "all":
                    linelist = lines_to_fit.values()
            else:
                linelist = [LINE_LOC]
            for line in linelist:
                slicedwl, loind, upind = slicearr(wl, line - MARGIN, line + MARGIN)
                slicedflux = flux[loind:upind]
                slicedflux_std = flux_std[loind:upind]
                if CULUMFIT:
                    params = pd.read_csv(f"output/{SPECFILE_NAME}/culum_spec_vals.csv")
                else:
                    params = pd.read_csv(f"output/{SPECFILE_NAME}/single_spec_vals.csv")
                params = params.loc[params["subspectrum"] == nspec]
                params = params.loc[params["line_loc"] == line]
                if len(params.index) == 0:
                    continue
                if list(params["cr_ind"])[0] != "[]":
                    crind_list = list(params["cr_ind"])[0].replace("[", "").replace("]", "").split(" ")
                    cr_ind = np.array([int(i) for i in crind_list])
                    cr_ind -= loind
                    mask = np.ones(slicedwl.shape, bool)
                    mask[cr_ind] = False

                    for i in cr_ind:
                        crmask = expand_mask(mask)
                        plt.plot(slicedwl[~crmask], slicedflux[~crmask], color="lightgray", label='_nolegend_')

                    slicedwl = np.ma.MaskedArray(slicedwl, ~mask)
                plt.rcParams["figure.figsize"] = (8, 6)
                plt.plot(slicedwl, slicedflux, color=SUBSPECCOLORS[0])

                if PLOT_FITTED_LINE:
                    if not CULUMFIT:
                        gamma = params['gamma'][0]
                        shift = params["lambda_0"][0]
                        eta = params["eta"][0]
                        scaling = params["scaling"][0]
                        slope = params["slope"][0]
                        heigth = params["flux_0"][0]
                    else:
                        gamma = params['gamma'].values[0]
                        shift = params["lambda_0"].values[0]
                        eta = params["eta"].values[0]
                        scaling = params["scaling"].values[0]
                        slope = params["slope"].values[0]
                        heigth = params["flux_0"].values[0]
                    wlspace = np.linspace(slicedwl.min(), slicedwl.max(), 10000)
                    plt.plot(wlspace, pseudo_voigt(wlspace, scaling, gamma, shift, slope, heigth, eta), color=COLORS[1])

                plt.title(TITLE, fontsize=TITLE_SIZE)
                plt.ylabel("Flux [ergs/s/cm^2/Å]", fontsize=LABEL_SIZE)
                plt.xlabel("Wavelength [Å]", fontsize=LABEL_SIZE)
                plt.xticks(fontsize=TICK_SIZE)
                plt.yticks(fontsize=TICK_SIZE)
                if PLOT_FITTED_LINE:
                    plt.legend(["Flux data", "Best Fit"])
                if MANUALLIMITS:
                    plt.ylim(YLIM)
                    plt.xlim(XLIM)
                plt.tight_layout()
                plt.savefig(f"images/{SPECFILE_NAME}_{line}A_plot.png")
                plt.show()
