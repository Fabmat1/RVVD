import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from main import pseudo_voigt, slicearr, load_spectrum, lines_to_fit, expand_mask

SPECFILE_NAME = "spec-2318-54628-0236"
TITLE = ""
SUBSPEC = "all"
GAIA_ID = ""
MARGIN = 100
LINE_LOC = 5015.678  # "all"
PLOT_OVERVIEW = True
OVERVIEW_SEP = 1
PLOT_INDIVIDUAL = False
PLOT_FITTED_LINE = False
CULUMFIT = True
COLORS = ["navy", "crimson"]
MANUALLIMITS = False
YLIM = (75, 110)
XLIM = (6550, 6575)
TITLE_SIZE = 17
LABEL_SIZE = 14
TICK_SIZE = 12
SUBSPECCOLORS = ["navy"]

# plt.rcParams["figure.figsize"] = (10, 4.5)
plt.rcParams["figure.dpi"] = 300

filenames = []


def get_params_from_filename(filename, paramtable: pd.DataFrame, sfilename=None):
    if sfilename is None:
        sfilename = SPECFILE_NAME
    specname, subspec = filename.split("/")[1].split(".")[0].split("_")
    if type(sfilename) == list:
        subspec = str(int(subspec))
        prow = paramtable.loc[paramtable["subspectrum"] == subspec + "_" + specname]
    else:
        subspec = int(subspec)
        prow = paramtable.loc[paramtable["subspectrum"] == subspec]
    # scaling, gamma, shift, slope, height, eta
    if len(prow) != 0:
        return prow["scaling"].iloc[0], prow["gamma"].iloc[0], prow["lambda_0"].iloc[0], prow["slope"].iloc[0], prow["flux_0"].iloc[0], prow["eta"].iloc[0]
    else:
        return None, None, None, None, None, None


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

    color = cm.rainbow(np.linspace(0, 1, len(filenames)))
    if PLOT_OVERVIEW:
        for line in lines_to_fit.values():
            plt.axvline(line, color="darkgrey", linestyle="--", linewidth=1)
        for ind, filename in enumerate(filenames):
            params = get_params_from_filename(filename, culumfit_table)
            wl, flux, _, flux_std = load_spectrum(filename)
            plt.plot(wl, flux + OVERVIEW_SEP * ind * np.mean(flux), color=color[ind])
            if params[1] is not None:
                wlforfit = np.linspace(params[2] - MARGIN, params[2] + MARGIN, 100)
                plt.plot(wlforfit, pseudo_voigt(wlforfit, *params) + OVERVIEW_SEP * ind * np.mean(flux))
        plt.title(f"Overview of {SPECFILE_NAME}")  # , fontsize=TITLE_SIZE)
        plt.ylabel("Flux [ergs/s/cm^2/Å] + Offset")  # , fontsize=LABEL_SIZE)
        plt.xlabel("Wavelength [Å]")  # , fontsize=LABEL_SIZE)
        plt.xticks()  # fontsize=TICK_SIZE)
        plt.yticks()  # fontsize=TICK_SIZE)
        # plt.xlim((4720, 5000))
        # plt.ylim((200, 1000))
        plt.tight_layout()
        plt.savefig(f"images/{SPECFILE_NAME}_overviewplot.png")
        plt.show()

    if PLOT_INDIVIDUAL:
        for filename in filenames:
            nspec = SUBSPEC[filenames.index(filename)]
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
