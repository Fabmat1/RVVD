import os
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from fit_rv_curve import fit_rv_curve


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
    rvdata = np.concatenate([np.genfromtxt(os.path.join(dirname, f"output/{f}/RV_variation.csv"), delimiter=',', dtype=object)[1:] for f in dirs_to_combine])
    single_fit_data = np.concatenate([replace_first_col(np.genfromtxt(os.path.join(dirname, f"output/{f}/single_spec_vals.csv"), delimiter=',', dtype=object)[1:], f) for f in dirs_to_combine])
    culum_fit_data = np.concatenate([replace_first_col(np.genfromtxt(os.path.join(dirname, f"output/{f}/culum_spec_vals.csv"), delimiter=',', dtype=object)[1:], f) for f in dirs_to_combine])

    rvdata = rvdata[rvdata[:, 4].argsort()]
    firstdir = dirs_to_combine[0]

    rvheader = np.genfromtxt(os.path.join(dirname, f"output/{dirs_to_combine[0]}/RV_variation.csv"), delimiter=',', dtype=str)[:1]
    single_fit_header = np.genfromtxt(os.path.join(dirname, f"output/{dirs_to_combine[0]}/single_spec_vals.csv"), delimiter=',', dtype=str)[:1]
    culum_fit_header = np.genfromtxt(os.path.join(dirname, f"output/{dirs_to_combine[0]}/culum_spec_vals.csv"), delimiter=',', dtype=str)[:1]

    try:
        os.mkdir(os.path.join(dirname, f"output/{firstdir}_merged/"))
    except FileExistsError:
        pass

    rv_array = np.concatenate([rvheader, rvdata], axis=0)
    single_fit_array = np.concatenate([single_fit_header, single_fit_data], axis=0)
    culum_fit_array = np.concatenate([culum_fit_header, culum_fit_data], axis=0)

    np.savetxt(os.path.join(dirname, f"output/{firstdir}_merged/RV_variation.csv"), rv_array.astype(np.str_), delimiter=",", fmt="%s")
    np.savetxt(os.path.join(dirname, f"output/{firstdir}_merged/single_spec_vals.csv"), single_fit_array.astype(np.str_), delimiter=",", fmt="%s")
    np.savetxt(os.path.join(dirname, f"output/{firstdir}_merged/culum_spec_vals.csv"), culum_fit_array.astype(np.str_), delimiter=",", fmt="%s")

    from main import plot_rvcurve, plot_rvcurve_brokenaxis
    rvdata = rvdata[:, :-1].astype(float)

    plot_rvcurve(rvdata[:, 0], rvdata[:, 1], rvdata[:, 4], firstdir, gaia_id, True)
    plot_rvcurve_brokenaxis(rvdata[:, 0], rvdata[:, 1], rvdata[:, 4], firstdir, gaia_id, True)

    for d in dirs_to_combine + add_dirs:
        open(os.path.join(dirname, f"output/.{d}"), 'a').close()
        dpath = os.path.join(dirname, f"output/{d}/")
        subdirs = [f.path for f in os.scandir(dpath) if f.is_dir()]
        for sdir in subdirs:
            os.rename(sdir, sdir + "_" + d)
            shutil.move(sdir + "_" + d, os.path.join(dirname, f"output/{firstdir}_merged/"))
        shutil.rmtree(dpath)


def result_statistics(analysis_params, catalogue):
    dirname = os.path.dirname(__file__)
    dirs = [f.path for f in os.scandir(os.path.join(dirname, "output")) if f.is_dir()]
    files = [os.path.join(d, "RV_variation.csv") for d in dirs]
    # catalogue = catalogue.groupby('source_id').agg({'source_id': 'first',
    #                                                 'file': ', '.join,
    #                                                 'SPEC_CLASS': 'first',
    #                                                 'bp_rp': 'first',
    #                                                 'gmag': 'first',
    #                                                 'nspec': 'sum'}).reset_index()
    # catalogue['logp'] = catalogue['source_id'].map(analysis_params.set_index('source_id').region)
    # print(catalogue)
    #
    # mag = catalogue[["gmag"]].to_numpy()
    # bp_rp = catalogue[["bp_rp"]].to_numpy()
    # logp = catalogue[["logp"]].to_numpy()
    #
    # plt.scatter(bp_rp, mag, logp**2, alpha=0.7)
    # plt.gca().invert_yaxis()
    # plt.show()

    mags = []
    frate = []
    for dir in dirs:
        specname = dir.split("\\")[-1]
        if "spec" not in specname.split("_")[0] or "_merged" in specname:
            continue
        gmag = catalogue.loc[catalogue["file"] == specname]["gmag"].iloc[0]
        n_ges = sum(os.path.isdir(os.path.join(dir, i)) for i in os.listdir(dir))
        if n_ges == 0:
            continue
        data = np.genfromtxt(dir + "\\RV_variation.csv", delimiter=',')[1:]
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


def compare_results():
    catalog = pd.read_csv("sd_catalogue_v56_pub.csv", delimiter=",")
    idents = dict(zip(catalog.NAME, [str(g).replace("Gaia EDR3 ", "") for g in catalog.GAIA_DESIG.to_list()]))
    comparetable = pd.read_csv("rvs_all_final.csv", delimiter=",", names=["Identifier", "mjd", "RV", "u_RV", "source"]).replace({"Identifier": idents})
    RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["RV"].apply(list).to_frame()
    u_RVs = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["u_RV"].apply(list).to_frame()
    times = comparetable.loc[comparetable["source"] == "SDSS"].groupby(["Identifier"])["mjd"].apply(list).to_frame()

    restable = pd.read_csv("result_parameters.csv", delimiter=",")

    differences = np.array([])
    allerrratio = np.array([])
    overlap = np.array([])

    for ind, row in RVs.iterrows():
        try:
            gaia_id = np.int64(ind)
        except ValueError:
            continue
        u_RVlist = np.array(u_RVs.loc[ind].iloc[0])
        RVlist = np.array(row.iloc[0])
        timelist = np.array(times.loc[ind].iloc[0])

        timelist -= np.amin(timelist)

        specfile = restable.loc[restable["source_id"] == gaia_id]
        m = False

        if len(specfile) == 1:
            specfile = specfile["associated_files"].iloc[0]
            if "," in specfile:
                m = True
                specfiles = [s for s in specfile.split("'") if "spec" in s]
                for s in specfiles:
                    dirname = os.path.dirname(__file__)
                    if os.path.exists(os.path.join(dirname, f"output/{s}_merged")):
                        specfile = s
            if "," in specfile:
                continue
            if not m:
                otherRVvals = np.genfromtxt(os.path.join(os.path.dirname(__file__), f"output/{specfile}/RV_variation.csv"), delimiter=",")
            else:
                otherRVvals = np.genfromtxt(os.path.join(os.path.dirname(__file__), f"output/{specfile}_merged/RV_variation.csv"), delimiter=",")

            if otherRVvals.ndim == 1:
                continue

            otherRVvals = otherRVvals[1:, :]
            otherRVlist = otherRVvals[:, 0]
            otheruRVlist = otherRVvals[:, 1]
            othertimelist = otherRVvals[:, 4]
            othertimelist -= np.amin(othertimelist)

            from main import plot_rvcurve_brokenaxis
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
            if len(otherRVlist) == len(RVlist):
                differences = np.concatenate([differences, np.abs(otherRVlist - RVlist)])
                allerrratio = np.concatenate([allerrratio, np.abs(otheruRVlist / u_RVlist)])
                overlap = np.concatenate([overlap, np.logical_or(np.logical_and(otherRVlist - otheruRVlist < RVlist + u_RVlist, otherRVlist > RVlist), np.logical_and(otherRVlist + otheruRVlist > RVlist - u_RVlist, otherRVlist < RVlist))])

    from main import PLOT_FMT
    from PyPDF2 import PdfFileMerger
    dirname = os.path.dirname(__file__)
    dirs = [f.path for f in os.scandir(os.path.join(dirname, "output")) if f.is_dir()]
    files = [os.path.join(d, f"RV_variation_broken_axis_comparison{PLOT_FMT}") for d in dirs if os.path.isfile(os.path.join(d, f"RV_variation_broken_axis_comparison{PLOT_FMT}"))]
    merger = PdfFileMerger()
    [merger.append(pdf) for pdf in files]
    with open("all_RV_plots_comparison.pdf", "wb") as new_file:
        merger.write(new_file)

    plt.hist(differences, np.linspace(0, 200, 25), color="crimson", edgecolor="black", zorder=10)
    plt.xlabel("RV difference [km/s]")
    plt.ylabel("#")
    plt.grid(True, linestyle="--", color="darkgray")
    plt.title("Statistic of RV differences between the different methods")
    plt.savefig(f"images/RVdiffstat{PLOT_FMT}")
    plt.show()
    print(f"Average error ratio: {np.mean(allerrratio)}")
    print(f"Overlap in {np.sum(overlap)} out of {len(overlap)} cases (Ratio of {np.sum(overlap) / len(overlap)})")


def result_analysis(check_doubles=False, catalogue: pd.DataFrame = None):
    dirname = os.path.dirname(__file__)
    dirs = [f.path for f in os.scandir(os.path.join(dirname, "output")) if f.is_dir()]
    files = [os.path.join(d, "RV_variation.csv") for d in dirs]

    analysis_params = pd.DataFrame(
        {
            "source_id": [],
            "logp": [],
            "associated_files": []
        }
    )

    if check_doubles and catalogue is not None:
        duplicated_stars = catalogue.loc[catalogue.duplicated(["source_id"])]
        sourceids = np.array([s for s in duplicated_stars["source_id"]])
    else:
        sourceids = []

    used_sids = []
    for file in files:
        specname = file.split("\\")[-2]
        if "_merged" in specname:
            sid = catalogue.loc[catalogue["file"] == specname.replace("_merged", "")]["source_id"].iloc[0]
            filedata = np.genfromtxt(file, delimiter=',')[1:]
            files_combined = catalogue.loc[catalogue["source_id"] == sid]["file"].tolist()
            if filedata.ndim == 1:
                analysis_params = pd.concat([analysis_params, pd.DataFrame({
                    "source_id": [str(sid)],
                    "logp": [0],
                    "associated_files": [files_combined]
                })])
                continue

            culumvs = filedata[:, 0]
            culumv_errs = filedata[:, 1]
            logp = vrad_pvalue(culumvs, culumv_errs)

            analysis_params = pd.concat([analysis_params, pd.DataFrame({
                "source_id": [str(sid)],
                "logp": [logp],
                "associated_files": [files_combined]
            })])
            continue

        sid = catalogue.loc[catalogue["file"] == specname]["source_id"].iloc[0]
        if sid in used_sids:
            continue
        if sid in sourceids.tolist():
            files_to_combine = catalogue.loc[catalogue["source_id"] == sid]["file"].tolist()
            data_to_combine = [np.genfromtxt(os.path.join(dirname, f"output/{f}/RV_variation.csv"), delimiter=',')[1:] for f in files_to_combine]
            gooddata = [d for d in data_to_combine if d.ndim == 2]
            if len(files_to_combine) == 0 or len(gooddata) == 0:
                analysis_params = pd.concat([analysis_params, pd.DataFrame({
                    "source_id": [str(sid)],
                    "logp": [0],
                    "associated_files": [files_to_combine]
                })])
                used_sids.append(sid)
                continue
            filedata = np.concatenate(gooddata)
            mask = [True if d.ndim == 2 else False for d in data_to_combine]
            nfiles_to_combine = [i for (i, v) in zip(files_to_combine, mask) if v]
            additional_files = [i for (i, v) in zip(files_to_combine, mask) if not v]
            mergedir(nfiles_to_combine, sid, additional_files)
            used_sids.append(sid)
        else:
            filedata = np.genfromtxt(file, delimiter=',')[1:]
            if filedata.ndim <= 1:
                analysis_params = pd.concat([analysis_params, pd.DataFrame({
                    "source_id": [str(sid)],
                    "logp": [0],
                    "associated_files": [specname]
                })])
                continue
            files_to_combine = specname
        culumvs = filedata[:, 0]
        culumv_errs = filedata[:, 1]
        logp = vrad_pvalue(culumvs, culumv_errs)

        analysis_params = pd.concat([analysis_params, pd.DataFrame({
            "source_id": [str(sid)],
            "logp": [logp],
            "associated_files": [files_to_combine]
        })])

    analysis_params = analysis_params.sort_values("logp", axis=0, ascending=True)
    analysis_params.to_csv("result_parameters.csv", index=False)
    print("Creating Statistics...")
    result_statistics(analysis_params, catalogue)


def result_analysis_with_rvfit():
    dirname = os.path.dirname(__file__)
    dirs = os.walk(os.path.join(dirname, "output"))
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
    analysis_params.to_csv("result_parameters.csv", index=False)


if __name__ == "__main__":
    # from main import CATALOGUE, create_pdf
    # result_analysis(True, pd.read_csv(CATALOGUE))
    # create_pdf()
    compare_results()
