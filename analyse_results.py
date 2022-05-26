import os
import numpy as np
import pandas as pd
from scipy import stats


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

    vrad_wmean = np.sum(vrad/vrad_err) / np.sum(1/vrad_err)

    chi = (vrad - vrad_wmean) / vrad_err

    chisq = chi**2
    chisq_sum = np.sum(chisq)

    dof = ndata - nfit

    pval = stats.chi2.sf(chisq_sum, dof)
    logp = np.log10(pval)

    return logp


dirname = os.path.dirname(__file__)
dirs = os.walk(os.path.join(dirname, "output"))
dirs = [d[0] for d in dirs if "spec" in d[0].split("\\")[-1]]
files = [os.path.join(d, "RV_variation.csv") for d in dirs]

var_over_err = pd.DataFrame(
    {
        "spec": [],
        "var_over_err": []
    }
)

std_reduction = []

for file in files:
    specname = file.split("\\")[-2]
    filedata = np.genfromtxt(file, delimiter=',')
    culumvs = filedata[1:, 0]
    culumv_errs = filedata[1:, 1]
    singlevs = filedata[1:, 2]
    singlev_errs = filedata[1:, 3]
    singlev_errs[singlev_errs == 0] = np.mean(singlev_errs)
    std_red = np.divide(culumv_errs, singlev_errs)
    std_red = np.mean(std_red)
    std_reduction.append(std_red)
    var_over_err = pd.concat([var_over_err, pd.DataFrame({
        "spec": [specname],
        "logp": [vrad_pvalue(culumvs, culumv_errs)],
        "var_over_err": np.std(culumvs)/np.mean(culumv_errs)
    })])

var_over_err = var_over_err.sort_values("logp", axis=0, ascending=False)
var_over_err.to_csv("logps.csv", index=False)
print(round(1-np.mean(np.array(std_reduction)), 3))
