import numpy as np
from matplotlib import pyplot as plt
from main import pseudo_voigt, slicearr, load_spectrum, v_from_doppler_rel, lines
import pandas as pd

SPECFILE_NAME = "spec-2682-54401-0569"  # spec-4067-55361-0224"
TITLE = "The H-Alpha line of the same star,\n observed at different times"  # A H-Alpha line of a spectrum with a cosmic ray event\n in an unfavorable place"
SUBSPEC = [7]
GAIA_ID = ""
MARGIN = 100
LINE_LOC = "all"
PLOT_OVERVIEW = True
PLOT_FITTED_LINE = True
CULUMFIT = True
COLORS = ["navy", "crimson"]
MANUALLIMITS = False
YLIM = (75, 110)
XLIM = (6550, 6575)
SUBSPECCOLORS = ["darkred", "darkgray", "gold"]

filenames = []

for specn in SUBSPEC:
    if len(SPECFILE_NAME) > 0:
        filenames.append(f"spectra/{SPECFILE_NAME}_{specn if len(str(specn)) > 1 else '0' + str(specn)}.txt")
    else:
        spectra_table = pd.read_csv("selected_objects.csv")
        spectra_table = spectra_table.loc[spectra_table["source_id"] == GAIA_ID]
        filenames.append(
            f"spectra/{spectra_table['file'][0]}_{specn if len(str(specn)) > 1 else '0' + str(specn)}.txt")

for filename in filenames:
    nspec = filenames.index(filename)+1
    wl, flux, _, flux_std = load_spectrum(filename)
    if LINE_LOC.lower() == "all":
        linelist = lines.values()
    else:
        linelist = [LINE_LOC]
    for line in linelist:
        if PLOT_OVERVIEW:
            plt.plot(wl, flux, color="navy")
            plt.title("Overview of a non-coadded Spectrum")
            plt.ylabel("Flux [ergs/s/cm^2/Å]")
            plt.xlabel("Wavelength [Å]")
            plt.savefig("images/overviewplot.png", dpi=300)
        plt.show()
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
        # if list(params["cr_ind"])[0] != "[]":
        #     crind_list = list(params["cr_ind"])[0].split("\n")[0].split("[[")[1].replace("]]", "").split(",")
        #     cr_ind = np.array([int(i) for i in crind_list])
        #     cr_ind -= loind
        #
        #     for i in cr_ind:
        #         plt.plot(slicedwl[i - 1:i + 2], flux[i - 1:i + 2], color="lightgray", label='_nolegend_')
        #     mask = np.ones(slicedwl.shape, bool)
        #     mask[cr_ind] = False
        #     slicedwl = np.ma.MaskedArray(slicedwl, ~mask)
        plt.plot(slicedwl, slicedflux, color=SUBSPECCOLORS[nspec-1])

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

        plt.title(TITLE)
        plt.ylabel("Flux [ergs/s/cm^2/Å]")
        plt.xlabel("Wavelength [Å]")
        if PLOT_FITTED_LINE:
            plt.legend(["Flux data", "Best Fit"])
        if MANUALLIMITS:
            plt.ylim(YLIM)
            plt.xlim(XLIM)
        plt.savefig(f"images/{SPECFILE_NAME}_{line}A_plot.png", dpi=500)
        plt.show()
