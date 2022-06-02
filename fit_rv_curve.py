import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import astropy.time as atime
import numpy as np

data = pd.read_csv(r"C:\Users\fabia\PycharmProjects\RVVD\output\spec-2682-54401-0569\RV_variation.csv")


def sinus(x, A, b, h, shift):
    return h + A*np.sin(b*np.pi*(x-shift))


stime = np.array([atime.Time(t, format="mjd").to_value(format="mjd") for t in data["mjd"]])
stime -= np.amin(stime)
RV = data["culum_fit_RV"]
RV_err = data["u_culum_fit_RV"]

periodguess = 0.12  # in days
p0 = [80, 2/(periodguess), -30, 0]

params, errs = curve_fit(sinus,
                         stime,
                         RV,
                         p0=p0,
                         sigma=RV_err)

A, b, h, shift = params
u_A, u_b, u_h, u_shift = np.sqrt(np.diag(errs))

print(f"{np.abs(A)}+-{u_A}")
print(f"{1 / b}+-{u_b / b**2}")
#print(f"{time.strftime('%H:%M:%S', time.gmtime(1 / b))}+-{time.strftime('%H:%M:%S', time.gmtime(u_b / b**2))}")

mjd_linspace = np.linspace(stime.min(), stime.max(), 1000)

plt.title(f"Radial Velocity over Time\n Gaia EDR3 944390774983674496")
plt.ylabel("Radial Velocity [km/s]")
plt.xlabel("Date")
plt.scatter(stime, RV, zorder=5)
plt.plot(mjd_linspace, sinus(mjd_linspace, *params), zorder=2)
# plt.plot(mjd_linspace, sinus(mjd_linspace, *p0), zorder=2)
plt.errorbar(stime, RV, yerr=RV_err, capsize=3, linestyle='', zorder=3)
plt.savefig(r"C:\Users\fabia\PycharmProjects\RVVD\output\spec-2682-54401-0569\RV_var_plusfit.png", dpi=300)
plt.show()

