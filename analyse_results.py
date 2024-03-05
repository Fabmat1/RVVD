import ast
import os
import shutil
import urllib.parse
from itertools import repeat
from multiprocessing import Pool
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.coordinates import get_body, get_sun
import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astroquery.mast import Catalogs
from matplotlib import pyplot as plt
from scipy import stats

from fit_rv_curve import fit_rv_curve
import astroquery.exceptions

OUTPUT_DIR = "output"
ADD_BIBC_TIC = True

# Big words: catalogue, study, indeterminate, irrelevant
# Prefixes, Suffixes: OG, HV, RV, WD, phot, spec, pulsation
paper_associations = {
    "2022A&A...662A..40C": "catalogue",
    "2019A&A...621A..38G": "catalogue",
    "2020A&A...635A.193G": "catalogue",
    "2017A&A...600A..50G": "catalogue",
    "2021ApJS..256...28L": "catalogue_RV",
    "2015MNRAS.448.2260G": "WD_catalogue",
    "2011MNRAS.417.1210G": "WD_catalogue",
    "2019MNRAS.482.4570G": "WD_catalogue",
    "2021MNRAS.508.3877G": "WD_catalogue",
    "2019ApJ...881....7L": "catalogue_RV",
    "2019MNRAS.486.2169K": "WD_catalogue",
    "2015MNRAS.446.4078K": "WD_catalogue",
    "2017ApJ...845..171B": "pulsation_catalogue",
    "2006ApJS..167...40E": "OG_WD_catalogue",
    "1986ApJS...61..305G": "OG_catalogue",
    "2017MNRAS.469.2102A": "WD_catalogue_RV",
    "2013ApJS..204....5K": "WD_catalogue",
    "1988SAAOC..12....1K": "OG_catalogue",
    "2019ApJ...881..135L": "catalogue",
    "2021A&A...650A.205V": "phot_study",
    "2020MNRAS.491.2280S": "irrelevant",
    "2016yCat....1.2035M": "WD_catalogue",
    "2020ApJ...901...93B": "irrelevant",
    "2016MNRAS.455.3413K": "catalogue",
    "2018ApJ...868...70L": "catalogue_RV",
    "2003AJ....126.1455S": "OG_spec_study",
    "2023ApJ...942..109L": "spec_study",
    "2004ApJ...607..426K": "OG_WD_catalogue",
    "2009yCat....1.2023S": "irrelevant",
    "2015MNRAS.452..765G": "WD_catalogue",
    "2008AJ....136..946M": "OG_variability_study",
    "2004A&A...426..367M": "OG_variability_study",
    "2020ApJ...889..117L": "catalogue",
    "2020ApJ...898...64L": "catalogue_RV",
    "2016ApJ...818..202L": "catalogue",
    "2022A&A...661A.113G": "variability_study",
    "1992AJ....104..203W": "OG_catalogue",
    "2016MNRAS.457.3396P": "spec_catalogue",
    "2015A&A...577A..26G": "variability_study",
    "2010A&A...513A...6O": "pulsation_study",
    "2021A&A...654A.107C": "catalogue",
    "2010MNRAS.407..681M": "WD_catalogue_RV",
    "2007ApJ...660..311B": "OG_HV_study",
    "2011A&A...530A..28G": "variability_study",
    "1957BOTT....2p...3I": "OG_catalogue",
    "2015MNRAS.450.3514K": "variability_study",
    "2013A&A...551A..31D": "irrelevant",
    "2019MNRAS.482.5222T": "WD_catalogue",
    "2012MNRAS.427.2180N": "spec_catalogue",
    "1997A&A...317..689T": "OG_catalogue_RV",
    "2005RMxAA..41..155S": "OG_catalogue",
    "2012ApJ...751...55B": "HV_study_RV",
    "2000A&AS..147..169B": "irrelevant",
    "1977A&AS...28..123B": "OG_catalogue",
    "2010AJ....139...59B": "irrelevant",
    "2015A&A...576A..44K": "variability_study",
    "2008ApJ...684.1143X": "irrelevant",
    "2019MNRAS.490.3158C": "irrelevant",
    "2017MNRAS.472.4173R": "WD_catalogue_RV",
    "2018MNRAS.475.2480P": "catalogue_RV",
    "2019MNRAS.488.2892P": "WD_catalogue",
    "2013MNRAS.429.2143C": "variability_study",
    "1980AnTok..18...55N": "OG_catalogue",
    "1984AnTok..20..130K": "OG_catalogue",
    "2011ApJ...730..128T": "WD_spec_catalogue",
    "2019PASJ...71...41L": "spec_catalogue",
    "2016A&A...596A..49B": "catalogue",
    "1990A&AS...86...53M": "OG_catalogue",
    "2007ApJ...671.1708B": "OG_HV_study_RV",
}


def vrad_pvalue(vrad, vrad_err):
    """
    :param vrad: Radial Velocity array
    :param vrad_err: Array of corresponding Errors
    :return: logp value
    """
    ndata = len(vrad)
    if ndata < 2:
        return np.nan
    nfit = 1

    vrad_wmean = np.sum(vrad / vrad_err) / np.sum(1 / vrad_err)

    chi = (vrad - vrad_wmean) / vrad_err

    chisq = chi ** 2
    chisq_sum = np.sum(chisq)

    dof = ndata - nfit

    pval = stats.chi2.sf(chisq_sum, dof)
    logp = np.log10(pval)

    if pval == 0:
        return -500
    if np.isnan(logp):
        return 0

    return logp


def replace_first_col(array, file_pre):
    array[:, 0] = np.core.defchararray.add(array[:, 0].astype(str), "_" + file_pre)
    return array


def mergedir(dirs_to_combine, gaia_id, add_dirs=[]):
    dirname = os.path.dirname(__file__)
    rvdata = np.concatenate([np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{f}/RV_variation.csv"), delimiter=',', dtype=object)[1:] for f in dirs_to_combine])
    single_fit_data = np.concatenate([replace_first_col(np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{f}/single_spec_vals.csv"), delimiter=',', dtype=object)[1:], f) for f in dirs_to_combine])
    culum_fit_data = np.concatenate([replace_first_col(np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{f}/culum_spec_vals.csv"), delimiter=',', dtype=object)[1:], f) for f in dirs_to_combine])

    rvdata = rvdata[rvdata[:, 4].argsort()]
    firstdir = dirs_to_combine[0]

    rvheader = np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{dirs_to_combine[0]}/RV_variation.csv"), delimiter=',', dtype=str)[:1]
    single_fit_header = np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{dirs_to_combine[0]}/single_spec_vals.csv"), delimiter=',', dtype=str)[:1]
    culum_fit_header = np.genfromtxt(os.path.join(dirname, f"{OUTPUT_DIR}/{dirs_to_combine[0]}/culum_spec_vals.csv"), delimiter=',', dtype=str)[:1]

    try:
        os.mkdir(os.path.join(dirname, f"{OUTPUT_DIR}/{firstdir}_merged/"))
    except FileExistsError:
        pass

    rv_array = np.concatenate([rvheader, rvdata], axis=0)
    single_fit_array = np.concatenate([single_fit_header, single_fit_data], axis=0)
    culum_fit_array = np.concatenate([culum_fit_header, culum_fit_data], axis=0)

    np.savetxt(os.path.join(dirname, f"{OUTPUT_DIR}/{firstdir}_merged/RV_variation.csv"), rv_array.astype(np.str_), delimiter=",", fmt="%s")
    np.savetxt(os.path.join(dirname, f"{OUTPUT_DIR}/{firstdir}_merged/single_spec_vals.csv"), single_fit_array.astype(np.str_), delimiter=",", fmt="%s")
    np.savetxt(os.path.join(dirname, f"{OUTPUT_DIR}/{firstdir}_merged/culum_spec_vals.csv"), culum_fit_array.astype(np.str_), delimiter=",", fmt="%s")

    from main import plot_rvcurve, plot_rvcurve_brokenaxis
    rvdata = rvdata[:, :-1].astype(float)

    plot_rvcurve(rvdata[:, 0], rvdata[:, 1], rvdata[:, 4], firstdir, gaia_id, True)
    plot_rvcurve_brokenaxis(rvdata[:, 0], rvdata[:, 1], rvdata[:, 4], firstdir, gaia_id, True)

    for d in dirs_to_combine + add_dirs:
        open(os.path.join(dirname, f"{OUTPUT_DIR}/.{d}"), 'a').close()
        dpath = os.path.join(dirname, f"{OUTPUT_DIR}/{d}/")
        subdirs = [f.path for f in os.scandir(dpath) if f.is_dir()]
        for sdir in subdirs:
            os.rename(sdir, sdir + "_" + d)
            shutil.move(sdir + "_" + d, os.path.join(dirname, f"{OUTPUT_DIR}/{firstdir}_merged/"))
        shutil.rmtree(dpath)


def result_statistics(analysis_params, catalogue):
    dirname = os.path.dirname(__file__)
    dirs = [f.path for f in os.scandir(os.path.join(dirname, f"{OUTPUT_DIR}")) if f.is_dir()]

    mags = []
    frate = []

    for dir in dirs:
        if os.name == "nt":
            specname = dir.split("\\")[-1] if "\\" in dir else dir.split("/")[-1]
        else:
            specname = dir.split("/")[-1] if "\\" in dir else dir.split("/")[-1]

        if "spec" not in specname.split("_")[0]:
            continue
        if "_merged" in specname:
            specname = specname.replace("_merged", "")
        gmag = catalogue.loc[catalogue["file"] == specname]["gmag"].iloc[0]
        n_ges = sum(os.path.isdir(os.path.join(dir, i)) for i in os.listdir(dir))
        if n_ges == 0:
            continue
        data = np.genfromtxt(dir + "\\" if "\\" + "RV_variation.csv" in dir else dir + "/" + "RV_variation.csv", delimiter=',')[1:]
        n_success = data.shape[0] if data.ndim == 2 else 0
        frate.append(1 - n_success / n_ges)
        mags.append(gmag)
    plt.title("Error Rate of the script over g-band magnitude")
    plt.xlabel("G-band Magnitude")
    plt.ylabel("Program error rate per subspectrum")
    plt.scatter(mags, frate, 5, color="lightgray")
    try:
        binned_frate = stats.binned_statistic(mags, frate, "mean", 25)
        plt.plot(np.linspace(np.amin(mags), np.amax(mags), 25), binned_frate[0], color="darkred")
        plt.tight_layout()
        plt.savefig("images/errorstatistic.pdf")
    except ValueError:
        pass


def compare_results(input_catalogue, plot_comp=True):
    catalog = pd.read_csv("catalogues/sd_catalogue_v56_pub.csv", delimiter=",")
    idents = dict(zip(catalog.NAME, [str(g).replace("Gaia EDR3 ", "") for g in catalog.GAIA_DESIG.to_list()]))
    comparetable = pd.read_csv("rvs_all_final.csv", delimiter=",", names=["Identifier", "mjd", "RV", "u_RV", "source"]).replace({"Identifier": idents})
    RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["RV"].apply(list).to_frame()
    u_RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["u_RV"].apply(list).to_frame()
    times = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["mjd"].apply(list).to_frame()

    restable = pd.read_csv("result_parameters.csv", delimiter=",")

    differences = np.array([])
    error_differences = np.array([])
    allerrratio = np.array([])
    overlap = np.array([])
    gmags = []
    nspecs = []
    original_vals = np.array([])
    other_vals = np.array([])
    from main import plot_rvcurve_brokenaxis
    for ind, row in RVs.iterrows():
        try:
            gaia_id = np.int64(ind)
        except ValueError:
            continue
        u_RVlist = np.array(u_RVs.loc[ind].iloc[0]).astype(float)
        RVlist = np.array(row.iloc[0]).astype(float)
        o_timelist = np.array(times.loc[ind].iloc[0]).astype(float)

        timelist = o_timelist - np.amin(o_timelist)

        corr = restable.loc[restable["source_id"] == gaia_id]
        m = False

        if len(corr) == 1:
            specfile = corr["associated_files"].iloc[0]
            if "," in specfile:
                m = True
                specfiles = [s for s in specfile.split("'") if "spec" in s]
                for s in specfiles:
                    dirname = os.path.dirname(__file__)
                    if os.path.exists(os.path.join(dirname, f"{OUTPUT_DIR}/{s}_merged")):
                        specfile = s
            otherRVvals = np.genfromtxt(os.path.join(os.path.dirname(__file__), f"{OUTPUT_DIR}/{corr['source_folder'].iloc[0]}/RV_variation.csv"), delimiter=",")

            if otherRVvals.ndim == 1:
                continue

            otherRVvals = otherRVvals[1:, :]
            otherRVlist = otherRVvals[:, 0]
            otheruRVlist = otherRVvals[:, 1]
            o_othertimelist = otherRVvals[:, 4]
            othertimelist = o_othertimelist - np.amin(o_othertimelist)

            if plot_comp:
                plot_rvcurve_brokenaxis(
                    otherRVlist,
                    otheruRVlist,
                    othertimelist,
                    specfile,
                    gaia_id,
                    merged=m,
                    extravels=RVlist,
                    extraverrs=u_RVlist,
                    extratimes=timelist
                )
            if not len(otherRVlist) == len(RVlist):
                otherlist = pd.DataFrame({
                    "time": np.round(o_othertimelist, 3) + 2400000,
                    "otherRV": otherRVlist,
                    "u_otherRV": otheruRVlist,
                })
                thislist = pd.DataFrame({
                    "time": np.round(o_timelist, 3),
                    "thisRV": RVlist,
                    "u_thisRV": u_RVlist,
                })
                comblist = otherlist.merge(thislist, on="time").dropna()
                if not comblist.empty:
                    otherRVlist = comblist["otherRV"].to_numpy()
                    RVlist = comblist["thisRV"].to_numpy()
                    otheruRVlist = comblist["u_otherRV"].to_numpy()
                    u_RVlist = comblist["u_thisRV"].to_numpy()
                else:
                    otherRVlist = np.array([])
                    RVlist = np.array([])
                    otheruRVlist = np.array([])
                    u_RVlist = np.array([])
            else:
                if not sum([round(a, 3) != round(b, 3) for a, b in zip(timelist, othertimelist)]) == 0:
                    otherlist = pd.DataFrame({
                        "time": np.round(o_othertimelist, 3) + 2400000,
                        "otherRV": otherRVlist,
                        "u_otherRV": otheruRVlist,
                    })
                    thislist = pd.DataFrame({
                        "time": np.round(o_timelist, 3),
                        "thisRV": RVlist,
                        "u_thisRV": u_RVlist,
                    })
                    comblist = otherlist.merge(thislist, on="time").dropna()
                    if not comblist.empty:
                        otherRVlist = comblist["otherRV"].to_numpy()
                        RVlist = comblist["thisRV"].to_numpy()
                        otheruRVlist = comblist["u_otherRV"].to_numpy()
                        u_RVlist = comblist["u_thisRV"].to_numpy()
                    else:
                        otherRVlist = np.array([])
                        RVlist = np.array([])
                        otheruRVlist = np.array([])
                        u_RVlist = np.array([])

            for k, val in enumerate(RVlist):
                if val < -400:
                    if otherRVlist[k] < -300:
                        print(gaia_id, val, otherRVlist[k])
            original_vals = np.concatenate([original_vals, RVlist])
            other_vals = np.concatenate([other_vals, otherRVlist])
            differences = np.concatenate([differences, np.abs(otherRVlist - RVlist)])
            error_differences = np.concatenate([error_differences, np.abs(otheruRVlist - u_RVlist)])
            allerrratio = np.concatenate([allerrratio, np.abs(otheruRVlist / u_RVlist)])
            overlap = np.concatenate([overlap, np.logical_or(np.logical_and(otherRVlist - otheruRVlist < RVlist + u_RVlist, otherRVlist > RVlist), np.logical_and(otherRVlist + otheruRVlist > RVlist - u_RVlist, otherRVlist < RVlist))])
            if sum(np.abs(otheruRVlist / u_RVlist) > 3.5) > 0:
                print(gaia_id)
            for i in range(len(otheruRVlist)):
                nspecs.append(corr["Nspec"].iloc[0])
            gmag = input_catalogue.loc[input_catalogue["source_id"] == gaia_id]["gmag"].iloc[0]
            for i in range(len(otheruRVlist)):
                gmags.append(gmag)

    from main import plot_config
    PLOT_FMT = plot_config["PLOT_FMT"]
    from PyPDF2 import PdfMerger
    dirname = os.path.dirname(__file__)
    dirs = [f.path for f in os.scandir(os.path.join(dirname, f"{OUTPUT_DIR}")) if f.is_dir()]
    files = [os.path.join(d, f"RV_variation_broken_axis_comparison{PLOT_FMT}") for d in dirs if os.path.isfile(os.path.join(d, f"RV_variation_broken_axis_comparison{PLOT_FMT}"))]
    merger = PdfMerger()
    [merger.append(pdf) for pdf in files]
    with open("all_RV_plots_comparison.pdf", "wb") as new_file:
        merger.write(new_file)

    plt.scatter(original_vals, other_vals, color="darkgray", marker="x", zorder=2)
    plt.plot([-1000, 1000], [-1000, 1000], linestyle="--", color="navy", zorder=3)
    plt.grid(True, zorder=1)
    plt.ylim(np.amin(other_vals) - np.ptp(other_vals) * 0.025, np.amax(other_vals) + np.ptp(other_vals) * 0.025)
    plt.xlim(np.amin(original_vals) - np.ptp(original_vals) * 0.025, np.amax(original_vals) + np.ptp(original_vals) * 0.025)
    plt.xlabel("RV Value (Geier et al., 2022) [kms$^{-1}$]", fontsize=15)
    plt.ylabel("RV Value (this work) [kms$^{-1}$]", fontsize=15)
    plt.gca().tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.savefig(f"images/RVdiffscatter{PLOT_FMT}")
    plt.show()
    plt.hist(differences, np.linspace(0, 200, 25), color="crimson", edgecolor="black", zorder=10)
    plt.xlabel("RV difference [km/s]", fontsize=15)
    plt.ylabel("#", fontsize=15)
    plt.gca().tick_params(axis='both', which='major', labelsize=13)
    plt.grid(True, linestyle="--", color="darkgray")
    # plt.title("Statistic of RV differences between the different methods")
    plt.tight_layout()
    plt.savefig(f"images/RVdiffstat{PLOT_FMT}")
    plt.show()
    binned_erratio = stats.binned_statistic(gmags, allerrratio, "mean", 25)
    plt.figure(figsize=(4.8 * 16 / 9, 4.8))
    plt.scatter(gmags, allerrratio, color="lightgrey")
    plt.plot(np.linspace(np.amin(gmags), np.amax(gmags), 25), binned_erratio[0], color="darkred")
    plt.grid(True)
    plt.xlabel("G Band magnitude of the host star [mag]", fontsize=15)
    plt.ylabel("Error ratio [no unit]", fontsize=15)
    plt.legend(["Datapoints", "Binned average"], fontsize=13)
    plt.gca().tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.savefig(f"images/erratio_gmag{PLOT_FMT}")
    plt.show()
    print(f"Average error ratio: {np.mean(allerrratio)}")
    print(f"Overlap in {np.sum(overlap)} out of {len(overlap)} cases (Ratio of {np.sum(overlap) / len(overlap)})")


def round_or_string(num_or_str, to_digit=0):
    if isinstance(num_or_str, str):
        return num_or_str
    else:
        return round(num_or_str, to_digit)


def overview(restable, catalogue):
    preamble = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage{xltabular}",
        r"\usepackage[landscape=true, margin=1.0in]{geometry}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large \textbf{Result Parameters}}",
        r"\end{center}",
        r"\begin{xltabular}{\textwidth}{l|l|l|l|l|l|l|l|X|X}",
        r"\textbf{\#} & \textbf{Name} &\textbf{GAIA Identifier} & \textbf{RA}[$^{\circ}$] & \textbf{Dec.}[$^{\circ}$] & \textbf{Class.} & \textbf{$N_{\mathrm{spec}}$} & $\mathrm{RV}_{\mathrm{avg}}$ [$\mathrm{kms}^{-1}$] & $\Delta\mathrm{RV}_{\max}$ [$\mathrm{kms}^{-1}$] & $\log p$ [no unit]\\",
        r"\hline"
    ]
    postamble = [
        r"\end{xltabular}",
        r"\end{document}"
    ]

    with open("result_parameters.tex", "w") as outtex:
        for line in preamble:
            outtex.write(line + "\n")
        n = 0
        for index, row in restable.iterrows():
            catrow = catalogue.loc[catalogue["source_id"] == row["source_id"]]
            name = catrow["name"].iloc[0]
            name = "" if name == "-" else name
            name = "" if name is None else name
            n += 1

            minusstring = ""
            rvavg = round_or_string(row['RVavg'])
            if not isinstance(rvavg, str):
                minusstring = r""

            cl = row["spec_class"]
            if len(cl) > 9:
                cl = cl[:8]

            if row["logp"] == 0:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"],
                                                                                                                                              4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & NaN\\" + "\n")
            elif row["logp"] == -500:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"],
                                                                                                                                              4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & $<-500$\\" + "\n")
            else:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"],
                                                                                                                                              4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & ${round_or_string(row['logp'], 2)}$\\" + "\n")
        for line in postamble:
            outtex.write(line + "\n")
    os.system("lualatex result_parameters.tex")


def truncate_simbad(file_path, isbibcodes=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    try:
        error_index = lines.index("::error:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
    except ValueError:
        error_index = None

    try:
        data_index = lines.index("::data::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
    except ValueError:
        print("The specified line is not found in the file.")
        return

    if isbibcodes:
        err_lines = []
        if error_index is not None:
            err_lines = lines[error_index + 2:data_index - 1]
            err_lines = ["start" + e.split(":")[-1].upper() for e in err_lines]

        del lines[:data_index + 2]

        lines = err_lines + lines

    else:
        del lines[:data_index + 2]

        lines = [l.replace("TIC ", "").replace("GAIA DR3 ", "") for l in lines]
        lines = ["source_id,tic\n"] + lines

    with open(file_path, 'w') as f:
        f.writelines(lines)


def simbad_request(resparams, script_location, output_location, isbibcodes=True):
    if isbibcodes:
        if os.path.isfile(output_location):
            sids = resparams.source_id.to_list()
            with open(output_location, "r") as sbfile:
                for l in sbfile.readlines():
                    if "start GAIA DR3 " in l:
                        l = l.replace("start GAIA DR3 ", "").replace(":", "").strip()
                        sids.remove(l)
                if len(sids) == 0:
                    return
    else:
        if os.path.isfile(output_location):
            sids = resparams.source_id.to_list()
            try:
                sid_tics_table_column = pd.read_csv(output_location)["source_id"].tolist()
                result = all([s in sid_tics_table_column for s in sids])
                if result:
                    return
            except:
                pass

    # Define the headers

    # Define the URL
    url = 'http://simbad.u-strasbg.fr/simbad/sim-script'

    # Open the file in binary mode
    with open(script_location, 'rb') as f:
        # Make the post request with the file and headers
        response = requests.post(url, files={'scriptFile': f}, data={'submit': 'submit file'})

    # Write the response content to a file
    with open(output_location, 'w') as f:
        f.write(response.text)

    truncate_simbad(output_location, isbibcodes)


def plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=None, date=None):
    plt.figure(figsize=(4.8 * 16 / 9, 4.8))
    plt.grid(color="lightgray", zorder=1)
    if date:
        plt.title(f"Visibility for {date}")

    delta_midnight = delta_midnight.to_value()
    sunaltazs_obsnight = sunaltazs_obsnight.alt.to_value()
    moonaltazs_obsnight = moonaltazs_obsnight.alt.to_value()
    obj_altazs_obsnight = obj_altazs_obsnight.alt.to_value()

    plt.plot(delta_midnight, sunaltazs_obsnight, color='r', label='Sun', zorder=3)
    plt.plot(delta_midnight, moonaltazs_obsnight, color=[0.75] * 3, ls='--', label='Moon', zorder=3)
    plt.plot(delta_midnight, obj_altazs_obsnight, label='Target', color='lime', zorder=3)
    plt.fill_between(delta_midnight, 0, 90,
                     sunaltazs_obsnight < -0, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, 0 * u.deg, 90,
                     sunaltazs_obsnight < -18, color='k', zorder=0)
    plt.legend(loc='upper left')
    plt.xlim(-8, 8)
    plt.xticks((np.arange(9) * 2 - 8))
    plt.ylim(0, 90)
    plt.xlabel('Hours from EDT Midnight')
    plt.ylabel('Altitude [deg]')
    plt.tight_layout()
    if saveloc:
        plt.savefig(saveloc)
        plt.close()
    else:
        plt.show()


def plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=None):
    plt.figure(figsize=(3, 3))
    plt.grid(color="lightgray", zorder=1)

    delta_midnight = delta_midnight.to_value()
    sunaltazs_obsnight = sunaltazs_obsnight.alt.to_value()
    moonaltazs_obsnight = moonaltazs_obsnight.alt.to_value()
    obj_altazs_obsnight = obj_altazs_obsnight.alt.to_value()

    # plt.plot(delta_midnight, sunaltazs_obsnight.alt, color='r', label='Sun', zorder=3)
    # plt.plot(delta_midnight, moonaltazs_obsnight.alt, color=[0.75] * 3, ls='--', label='Moon', zorder=3)
    plt.plot(delta_midnight, obj_altazs_obsnight, label='Target', color='lime', zorder=3)
    plt.fill_between(delta_midnight, 0, 90,
                     sunaltazs_obsnight < -0, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, 0, 90,
                     sunaltazs_obsnight < -18, color='k', zorder=0)
    plt.xlabel("")
    plt.ylabel("")

    plt.xlim(-8, 8)
    plt.ylim(0, 90)

    ax = plt.gca()
    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    plt.axhline(30, color="gold", linestyle="--")
    plt.axhline(20, color="red", linestyle="--")

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if saveloc:
        plt.savefig(saveloc)
        plt.close()
    else:
        plt.show()  #


def get_visibility(frame, ra, dec):
    observed_obj = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    obj_altazs_obsnight = observed_obj.transform_to(frame)

    return obj_altazs_obsnight


def individual_visibility(stuff, delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight, date):
    ra, dec, sid = stuff
    obj_altazs_obsnight = get_visibility(frame_obsnight, ra, dec)
    try:
        observability = np.ptp(delta_midnight[np.logical_and(obj_altazs_obsnight.alt > 30 * u.deg, sunaltazs_obsnight.alt < 0 * u.deg)])
        if observability != 0:
            plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/visibility.pdf", date=date)
            plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/tiny_visibility.pdf")
        observability = observability.value
    except ValueError:
        observability = 0

    return sid, observability


def get_frame(date='2023-10-12 00:00:00', utc_offset=-4):
    # Location of the SOAR telescope
    location = EarthLocation.of_site("Cerro Pachon")

    utcoffset = utc_offset * u.hour
    midnight = Time(date) - utcoffset

    delta_midnight = np.linspace(-8, 8, 1000) * u.hour
    times_obsnight = midnight + delta_midnight
    frame_obsnight = AltAz(obstime=times_obsnight, location=location)
    sunaltazs_obsnight = get_sun(times_obsnight).transform_to(frame_obsnight)

    moon_obsnight = get_body("moon", times_obsnight)
    moonaltazs_obsnight = moon_obsnight.transform_to(frame_obsnight)

    return delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight


def get_visibility_for_night(ra, dec, sid, date, utc_offset):
    fileloc = "local_observing_conditions/" + date.replace("-", "_").replace(" 00:00:00", "") + ".csv"
    if not os.path.isfile(fileloc):
        delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight = get_frame(date, utc_offset)

        assert len(ra) == len(dec) == len(sid)

        stuff_list = [(r, d, s) for r, d, s in zip(ra, dec, sid)]

        with Pool() as pool:
            results = pool.starmap(individual_visibility, zip(stuff_list, repeat(delta_midnight), repeat(frame_obsnight), repeat(sunaltazs_obsnight), repeat(moonaltazs_obsnight), repeat(date)))

        obs_conditions = pd.DataFrame(results, columns=['sid', 'observability'])

        if not os.path.isdir("local_observing_conditions"):
            os.mkdir("local_observing_conditions")

        obs_conditions.to_csv(fileloc, index=False)
    else:
        obs_conditions = pd.read_csv(fileloc)
        obs_conditions["sid"] = obs_conditions["sid"].astype(str)

    return obs_conditions


def calculate_spatial_velocity(rv, pmra, pmdec, parallax, rv_err=None, pmra_err=None, pmdec_err=None, parallax_err=None):
    if not all([isinstance(k, float) for k in [rv, pmra, pmdec, parallax, rv_err, pmra_err, pmdec_err, parallax_err]]):
        return np.nan, np.nan

    parallax = np.abs(parallax/1000)
    parallax_err /= 1000
    distance = 1 / parallax

    ra_speed = pmra * np.pi/648000000 * distance * 978462  # pc/yr to km/s
    dec_speed = pmdec * np.pi/648000000 * distance * 978462  # pc/yr to km/s

    spatial_vel = np.sqrt(rv ** 2 + ra_speed ** 2 + dec_speed ** 2)

    if parallax_err < parallax / 2:
        distance_err = parallax_err / parallax ** 2
        ra_speed_err = np.sqrt((distance_err * distance * np.pi/648000000 * 978462) ** 2 + (pmra_err * pmra * np.pi/648000000 * 978462) ** 2)
        dec_speed_err = np.sqrt((distance_err * distance * np.pi/648000000 * 978462) ** 2 + (pmdec_err * pmdec * np.pi/648000000 * 978462) ** 2)

        spatial_vel_err = np.sqrt((rv_err * rv / ((rv ** 2 + ra_speed ** 2 + dec_speed ** 2) ** (3 / 2))) ** 2 +
                                  (ra_speed_err * ra_speed / ((rv ** 2 + ra_speed ** 2 + dec_speed ** 2) ** (3 / 2))) ** 2 +
                                  (dec_speed_err * dec_speed / ((rv ** 2 + ra_speed ** 2 + dec_speed ** 2) ** (3 / 2))) ** 2)
        return spatial_vel, spatial_vel_err
    else:
        parallax = np.abs(parallax) + parallax_err
        distance = 1 / parallax

        ra_speed = pmra * distance * np.pi/648000000 * 978462  # pc/yr to km/s
        dec_speed = pmdec * distance * np.pi/648000000 * 978462 # pc/yr to km/s

        spatial_vel = np.sqrt(rv ** 2 + ra_speed ** 2 + dec_speed ** 2)

        return spatial_vel, spatial_vel


def add_useful_columns(res_params, catalogue, config):
    res_params = res_params.reset_index().drop(columns="index")
    with open("simbad_script.s", "w", encoding="utf-8") as outfile:
        outfile.write('format object f1 "start %OBJECT\\n"+\n"%BIBCODELIST"\n')
        for ind, row in res_params.iterrows():
            outfile.write(f"query id GAIA DR3 {row['source_id']}\n")

    simbad_request(res_params, "simbad_script.s", "simbad.txt", True)

    source_tic_table = pd.DataFrame(columns=["source_id"])
    source_tic_table["source_id"] = res_params["source_id"]

    if "GET_TICS" in config:
        with open("tic_script.s", "w") as ticscript:
            ticscript.write(r'format object f1 "%OBJECT,%IDLIST(TIC)\n"' + "\n")
            for ind, row in res_params.iterrows():
                ticscript.write(f"query id GAIA DR3 {row['source_id']}\n")
        simbad_request(res_params, "tic_script.s", "simbadtics.txt", False)
        tictable = pd.read_csv("simbadtics.txt")
        tictable["source_id"] = tictable["source_id"].astype("U20")
        tictable["tic"] = tictable["tic"].astype(int, errors="ignore").astype(str)
        source_tic_table = pd.merge(source_tic_table, tictable, on='source_id', how='left')
    else:
        source_tic_table = pd.DataFrame(columns=["source_id", "tic"])
        source_tic_table["source_id"] = res_params["source_id"]
        source_tic_table["tic"] = res_params["source_id"]
        source_tic_table["tic"] = np.full(np.shape(res_params["source_id"].to_numpy()), "notconfigured")

    source_tic_table.to_csv("gaia_tic_reftable.csv", index=False)

    try:
        stephancat = pd.read_csv("logp_values_rvvpaper.csv")
        stephancat["GaiaDR2"].str.strip()
    except:
        stephancat = None

    list_items_dict = {}

    with open("simbad.txt", "r") as f:
        source_id = None
        bibcodes = []
        for line in f:
            if line.startswith("start GAIA DR3"):
                if source_id is not None:
                    list_items_dict[source_id] = bibcodes
                source_id = line.split(" ")[3].strip()
                bibcodes = []
                if stephancat is not None:
                    try:
                        logp = stephancat.loc[stephancat["GaiaDR2"] == str(source_id)].iloc[0]["logP"]
                        if logp < -1.3:
                            bibcodes.append("2022A&A...661A.113G")
                    except IndexError:
                        pass
            else:
                bibcodes.append(line.strip())

    if source_id is not None and bibcodes:
        list_items_dict[source_id] = bibcodes  # add last block

    def joiner(x):
        try:
            return str(x)
        except:
            if pd.isnull(x):
                return "-"

    source_tic_table["bibcodes"] = source_tic_table["source_id"].map(list_items_dict)
    source_tic_table["bibcodes"] = source_tic_table["bibcodes"].apply(joiner)
    source_tic_table.to_csv("gaia_tic_bib_table.csv", index=False)

    try:
        res_params = res_params.drop("tic", axis=1)
        res_params = res_params.drop("bibcodes", axis=1)
    except KeyError:
        pass

    source_tic_table["source_id"] = source_tic_table["source_id"].astype(str)

    # Merge the two DataFrames based on the 'source_id' column
    merged_df = res_params.merge(source_tic_table[['source_id', 'tic', 'bibcodes']], on='source_id', how='inner')

    out_df = pd.DataFrame(
        {
            "source_id": [],
            "alias": [],
            "ra": [],
            "dec": [],
            "gmag": [],
            "bp_rp": [],
            "spec_class": [],
            "logp": [],
            "deltaRV": [],
            "deltaRV_err": [],
            "RVavg": [],
            "RVavg_err": [],
            "Nspec": [],
            "parallax": [],
            "parallax_err": [],
            "pmra": [],
            "pmra_err": [],
            "pmdec": [],
            "pmdec_err": [],
            "bibcodes": [],
            "tic": [],
            "associated_files": []
        }
    )

    for i, star in merged_df.iterrows():
        cat_dict = dict(catalogue.loc[catalogue["source_id"] == star["source_id"]].iloc[0])
        gmag = cat_dict["gmag"]
        bp_rp = cat_dict["bp_rp"]
        sp_class = cat_dict["SPEC_CLASS"]
        alias = cat_dict["name"]
        plx = cat_dict["parallax"]
        plx_err = cat_dict["parallax_error"]
        pmra = cat_dict["pmra"]
        pmra_err = cat_dict["pmra_error"]
        pmdec = cat_dict["pmdec"]
        pmdec_err = cat_dict["pmdec_error"]

        out_df = pd.concat([out_df, pd.DataFrame({
            "source_id": [star["source_id"]],
            "alias": [alias],
            "tic": [star["tic"]],
            "ra": [star["ra"]],
            "dec": [star["dec"]],
            "gmag": [gmag],
            "bp_rp": [bp_rp],
            "spec_class": [sp_class],
            "logp": [star["logp"]],
            "deltaRV": [star["deltaRV"]],
            "deltaRV_err": [star["deltaRV_err"]],
            "RVavg": [star["RVavg"]],
            "RVavg_err": [star["RVavg_err"]],
            "Nspec": [star["Nspec"]],
            "parallax": [plx],
            "parallax_err": [plx_err],
            "pmra": [pmra],
            "pmra_err": [pmra_err],
            "pmdec": [pmdec],
            "pmdec_err": [pmdec_err],
            "bibcodes": [star["bibcodes"]],
            "associated_files": [star["associated_files"]],
        })])

    if config["GET_VISIBILITY"]:
        print("Getting visibility conditions...")
        obs_conditions = get_visibility_for_night(out_df.ra.to_numpy(),
                                                  out_df.dec.to_numpy(),
                                                  out_df.source_id.to_numpy(),
                                                  config["FOR_DATE"] + " 00:00:00",
                                                  -4)

        merged_df = pd.merge(out_df, obs_conditions[['sid', 'observability']], left_on='source_id', right_on='sid', how='left')

        # Drop the redundant 'sid' column after merging if needed
        merged_df.drop('sid', axis=1, inplace=True)

        out_df = merged_df

    if config["TAG_KNOWN"]:
        out_df["known_category"] = "unknown"
        out_df["flags"] = None
        for ind, row in out_df.iterrows():
            bibcodes = ast.literal_eval(row["bibcodes"])
            known_category = "unknown"
            flags = []
            bibcode_associations = []

            if len(bibcodes) >= 10:
                known_category = "indeterminate"
            if isinstance(row["RVavg"], float):
                if np.abs(row["RVavg"]) > 250:
                    flags.append("HV-detection")
            if isinstance(row["logp"], float):
                if row["logp"] < -4:
                    flags.append("rvv-detection")
                elif row["logp"] < -1.3:
                    flags.append("rvv-candidate")

            for b in bibcodes:
                try:
                    bibcode_associations.append(paper_associations[b])
                except KeyError:
                    known_category = "indeterminate"

            for association in bibcode_associations:
                if association == "variability_study":
                    known_category = "known"
                    break
                if "study" in association:
                    known_category = "likely_known"
                split_assoc = association.split("_")
                for flg in ["OG", "HV", "RV", "phot", "spec", "pulsation"]:
                    if flg in split_assoc:
                        known_category = "likely_known"
                        if flg not in flags:
                            flags.append(flg)
                if "WD" in split_assoc:
                    flags.append("WD")
            if "+" in row["spec_class"]:
                known_category = "known"

            out_df.at[ind, "known_category"] = known_category
            out_df.at[ind, "flags"] = flags

    applied_thingy = pd.DataFrame(list(out_df.apply(lambda x: calculate_spatial_velocity(x['RVavg'],
                                                                                                   x['pmra'],
                                                                                                   x['pmdec'],
                                                                                                   x['parallax'],
                                                                                                   x['RVavg_err'],
                                                                                                   x['pmra_err'],
                                                                                                   x['pmdec_err'],
                                                                                                   x['parallax_err']), axis=1)))
    applied_thingy.columns = ["spatial_vels", "spatial_vels_err"]

    out_df["spatial_vels"] = applied_thingy["spatial_vels"]
    out_df["spatial_vels_err"] = applied_thingy["spatial_vels_err"]

    return out_df


def result_analysis(catalogue: pd.DataFrame = None, outdir="output", config=None):
    if config is None:
        config = {}
    global OUTPUT_DIR
    OUTPUT_DIR = outdir

    analysis_params = pd.DataFrame(
        {
            "source_id": [],
            "ra": [],
            "dec": [],
            "spec_class": [],
            "logp": [],
            "deltaRV": [],
            "deltaRV_err": [],
            "RVavg": [],
            "RVavg_err": [],
            "Nspec": [],
            "associated_files": []
        }
    )
    for i, star in catalogue.iterrows():
        sid = star["source_id"]
        ra = star["ra"]
        dec = star["dec"]
        specclass = star["SPEC_CLASS"]

        filedata = np.genfromtxt(f"{OUTPUT_DIR}/" + sid + "/RV_variation.csv", delimiter=",", skip_header=True)
        files = ";".join(star["file"])

        if filedata.ndim == 1 and len(filedata) == 0:
            analysis_params = pd.concat([analysis_params, pd.DataFrame({
                "source_id": [sid],
                "ra": [ra],
                "dec": [dec],
                "spec_class": [specclass],
                "logp": [0],
                "deltaRV": [""],
                "deltaRV_err": [""],
                "RVavg": [""],
                "RVavg_err": [""],
                "Nspec": [0],
                "timespan": [""],
                "associated_files": [files]
            })])
            continue
        elif filedata.ndim == 1 and len(filedata) != 0:
            culumvs = np.array([filedata[0]])
            culumv_errs = np.array([filedata[1]])
            analysis_params = pd.concat([analysis_params, pd.DataFrame({
                "source_id": [sid],
                "ra": [ra],
                "dec": [dec],
                "spec_class": [specclass],
                "logp": [0],
                "deltaRV": [""],
                "deltaRV_err": [""],
                "RVavg": [np.mean(culumvs)],
                "RVavg_err": [np.sqrt(np.sum(np.square(culumv_errs))) / len(culumv_errs)],
                "Nspec": [1],
                "timespan": [""],
                "associated_files": [files]
            })])
            continue

        culumvs = filedata[:, 0]
        culumv_errs = filedata[:, 1]
        timespan = np.ptp(filedata[:, 2])
        logp = vrad_pvalue(culumvs, culumv_errs)
        deltarv = np.ptp(culumvs)

        analysis_params = pd.concat([analysis_params, pd.DataFrame({
            "source_id": [sid],
            "ra": [ra],
            "dec": [dec],
            "spec_class": [specclass],
            "logp": [logp],
            "deltaRV": [deltarv],
            "deltaRV_err": [np.sqrt(culumv_errs[np.argmax(culumvs)] ** 2 + culumv_errs[np.argmin(culumvs)] ** 2) / 2],
            "RVavg": [np.mean(culumvs)],
            "RVavg_err": [np.sqrt(np.sum(np.square(culumv_errs))) / len(culumv_errs)],
            "Nspec": [len(culumvs)],
            "timespan": [timespan],
            "associated_files": [files]
        })])

    analysis_params = analysis_params.sort_values("logp", axis=0, ascending=True)
    if ADD_BIBC_TIC:
        analysis_params = add_useful_columns(analysis_params, catalogue, config)
    output_params = analysis_params.copy()
    output_params.to_csv("result_parameters.csv", index=False)
    print("Creating Statistics...")
    overview(analysis_params, catalogue)
    result_statistics(analysis_params, catalogue)


if __name__ == "__main__":
    # from main import CATALOGUE, create_pdf
    # result_analysis(True, pd.read_csv(CATALOGUE))
    # create_pdf()
    compare_results(False)
