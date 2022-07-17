import os
import shutil

import numpy as np
import pandas as pd
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

    return logp


def mergedir(dirs_to_combine, gaia_id):
    dirname = os.path.dirname(__file__)
    rvdata = np.concatenate([np.genfromtxt(os.path.join(dirname, f"output/{f}/RV_variation.csv"), delimiter=',', dtype=object)[1:] for f in dirs_to_combine])
    single_fit_data = np.concatenate([np.genfromtxt(os.path.join(dirname, f"output/{f}/single_spec_vals.csv"), delimiter=',', dtype=object)[1:] for f in dirs_to_combine])
    culum_fit_data = np.concatenate([np.genfromtxt(os.path.join(dirname, f"output/{f}/culum_spec_vals.csv"), delimiter=',', dtype=object)[1:] for f in dirs_to_combine])

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

    for d in dirs_to_combine:
        open(os.path.join(dirname, f"output/.{d}"), 'a').close()
        shutil.rmtree(os.path.join(dirname, f"output/{d}/"))


def result_analysis(check_doubles=False, catalogue: pd.DataFrame = None):
    dirname = os.path.dirname(__file__)
    dirs = os.walk(os.path.join(dirname, "output"))
    dirs = [d[0] for d in dirs if "spec" in d[0].split("\\")[-1]]
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
            continue
        sid = catalogue.loc[catalogue["file"] == specname]["source_id"].iloc[0]
        if sid in used_sids:
            continue
        if sid in sourceids.tolist():
            files_to_combine = catalogue.loc[catalogue["source_id"] == sid]["file"].tolist()
            filedata = np.concatenate([np.genfromtxt(os.path.join(dirname, f"output/{f}/RV_variation.csv"), delimiter=',')[1:] for f in files_to_combine])
            mergedir(files_to_combine, sid)
            used_sids.append(sid)
        else:
            filedata = np.genfromtxt(file, delimiter=',')[1:]
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
    result_analysis()
