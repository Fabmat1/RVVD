import os
import shutil

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from matplotlib import pyplot as plt
from scipy import stats

from fit_rv_curve import fit_rv_curve
import astroquery.exceptions

OUTPUT_DIR = "output"

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
        specname = dir.split("\\")[-1] if "\\" in dir else dir.split("/")[-1]
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


def compare_results(plot_comp=True):
    catalog = pd.read_csv("catalogues/sd_catalogue_v56_pub.csv", delimiter=",")
    idents = dict(zip(catalog.NAME, [str(g).replace("Gaia EDR3 ", "") for g in catalog.GAIA_DESIG.to_list()]))
    comparetable = pd.read_csv("rvs_all_final.csv", delimiter=",", names=["Identifier", "mjd", "RV", "u_RV", "source"]).replace({"Identifier": idents})
    RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["RV"].apply(list).to_frame()
    u_RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["u_RV"].apply(list).to_frame()
    times = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["mjd"].apply(list).to_frame()

    restable = pd.read_csv("result_parameters.csv", delimiter=",")
    input_catalogue = pd.read_csv("all_objects_withlamost.csv")

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
            n += 1

            minusstring = ""
            rvavg = round_or_string(row['RVavg'])
            if not isinstance(rvavg, str):
                minusstring = r""

            cl = row["spec_class"]
            if len(cl) > 9:
                cl = cl[:8]

            if row["logp"] == 0:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"], 4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & NaN\\" + "\n")
            elif row["logp"] == -500:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"], 4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & $<-500$\\" + "\n")
            else:
                outtex.write(f"{n}&" + name + "&" + row["source_id"] + r"&" + str(round_or_string(row["ra"], 4)) + r"&" + str(round_or_string(row["dec"], 4)) + r"&" + cl + f"&${int(row['Nspec'])}$" + "&" + rf"{minusstring} ${rvavg}\pm{round_or_string(row['RVavg_err'])}$" + rf"& ${round_or_string(row['deltaRV'])}\pm{round_or_string(row['deltaRV_err'])}$" + rf" & ${round_or_string(row['logp'], 2)}$\\" + "\n")
        for line in postamble:
            outtex.write(line + "\n")
    os.system("lualatex result_parameters.tex")


def add_bibcodes_and_tics(res_params):


    if os.path.isfile("gaia_tic_reftable.csv"):
        source_tic_table = pd.read_csv("gaia_tic_reftable.csv")
    else:
        source_tic_table = pd.DataFrame(columns=["source_id", "tic"])

    for ind, row in res_params.iterrows():
        if ind < len(source_tic_table):
            continue
        print(f"Getting TIC for object nr. {ind + 1}/{len(res_params)}")
        gaia_id = row["source_id"]
        try:
            star_info = Catalogs.query_object("Gaia DR3 " + str(gaia_id), catalog="TIC")
        except astroquery.exceptions.ResolverError:
            # star_info = Catalogs.query_region(SkyCoord(row["ra"], row["dec"], frame="icrs", unit="deg"))
            star_info = [{"ID": "-"}]

        tic_id = star_info[0]['ID']
        source_tic_table = pd.concat([source_tic_table, pd.DataFrame({"source_id": [gaia_id], "tic": [tic_id]})])
        source_tic_table.to_csv("gaia_tic_reftable.csv", index=False)

    source_tic_table.to_csv("gaia_tic_reftable.csv", index=False)

    stephancat = pd.read_csv("logp_values_rvvpaper.csv")
    stephancat["GaiaDR2"].str.strip()

    list_items_dict = {}

    with open("simbad.txt", "r") as f:
        source_id = None
        bibcodes = []
        for line in f:
            if line.startswith("start GAIA DR3"):
                if source_id is not None:
                    list_items_dict[source_id] = bibcodes
                source_id = int(line.split()[3].strip(":"))
                bibcodes = []
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
            return ';'.join(x)
        except:
            if pd.isnull(x):
                return "-"
            print(x)
            exit(-1)

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

    return merged_df


def result_analysis(catalogue: pd.DataFrame = None, outdir="output"):

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

        filedata = np.genfromtxt(f"{OUTPUT_DIR}/"+sid+"/RV_variation.csv", delimiter=",", skip_header=True)
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
                "RVavg_err": [np.sqrt(np.sum(np.square(culumv_errs)))],
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
            "RVavg_err": [np.sqrt(np.sum(np.square(culumv_errs)))],
            "Nspec": [len(culumvs)],
            "timespan": [timespan],
            "associated_files": [files]
        })])

    analysis_params = analysis_params.sort_values("logp", axis=0, ascending=True)
    analysis_params = add_bibcodes_and_tics(analysis_params)
    output_params = analysis_params.copy()
    output_params.to_csv("result_parameters.csv", index=False)
    print("Creating Statistics...")
    overview(analysis_params, catalogue)
    result_statistics(analysis_params, catalogue)


def result_analysis_with_rvfit():
    dirname = os.path.dirname(__file__)
    dirs = os.walk(os.path.join(dirname, OUTPUT_DIR))
    dirs = [d[0] for d in dirs if "spec" in d[0].split("\\")[-1]]
    files = [os.path.join(d, "RV_variation.csv") for d in dirs]

    analysis_params = pd.DataFrame(
        {
            "spec": [],
            "logp": [],
            "K": [],
            "u_K": [],
            "p": [],
            "u_p": []
        }
    )

    for file in files:
        specname = file.split("\\")[-2]
        filedata = np.genfromtxt(file, delimiter=',')
        culumvs = filedata[1:, 0]
        culumv_errs = filedata[1:, 1]
        logp = vrad_pvalue(culumvs, culumv_errs)
        if logp < -4 and len(culumvs) > 4:
            [A, p], [u_A, u_p] = fit_rv_curve(specname, "auto", False, False)
            if type(A) != str:
                if A / u_A < 2:
                    A = p = u_A = u_p = "--"
        else:
            A = p = u_A = u_p = "--"

        analysis_params = pd.concat([analysis_params, pd.DataFrame({
            "spec": [specname],
            "logp": [logp],
            "K": [A],
            "u_K": [u_A],
            "p": [p],
            "u_p": [u_p]
        })])

    analysis_params = analysis_params.sort_values("logp", axis=0, ascending=True)
    analysis_params = add_bibcodes_and_tics(analysis_params)
    analysis_params.to_csv("result_parameters.csv", index=False)


if __name__ == "__main__":
    # from main import CATALOGUE, create_pdf
    # result_analysis(True, pd.read_csv(CATALOGUE))
    # create_pdf()
    compare_results(False)
