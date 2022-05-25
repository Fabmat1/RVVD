import os
import numpy as np
import pandas as pd


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
        "var_over_err": np.std(culumvs)/np.mean(culumv_errs)
    })])

var_over_err = var_over_err.sort_values("var_over_err", axis=0)
var_over_err.to_csv("var_over_err.csv", index=False)
print(round(1-np.mean(np.array(std_reduction)), 3))
