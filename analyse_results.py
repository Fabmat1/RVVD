import os
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


def result_analysis():
    dirname = os.path.dirname(__file__)
    dirs = os.walk(os.path.join(dirname, "output"))
    dirs = [d[0] for d in dirs if "spec" in d[0].split("\\")[-1]]
    files = [os.path.join(d, "RV_variation.csv") for d in dirs]

    analysis_params = pd.DataFrame(
        {
            "spec": [],
            "logp": []
        }
    )

    for file in files:
        specname = file.split("\\")[-2]
        filedata = np.genfromtxt(file, delimiter=',')
        culumvs = filedata[1:, 0]
        culumv_errs = filedata[1:, 1]
        logp = vrad_pvalue(culumvs, culumv_errs)

        analysis_params = pd.concat([analysis_params, pd.DataFrame({
            "spec": [specname],
            "logp": [logp]
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
                if A/u_A < 2:
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
