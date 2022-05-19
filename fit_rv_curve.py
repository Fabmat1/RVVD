import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import astropy.time as atime
import numpy as np

data = pd.read_csv(r"C:\Users\fabia\PycharmProjects\RVVD\output\spec-2682-54401-0569\RV_variation.csv")


def sinus(x, A, b, h, shift):
    return h + A*np.sin(b*np.pi*(x-shift))


stime = np.array([atime.Time(t, format="mjd").to_value("unix", 'long') for t in data["mjd"]])
datetimes = [atime.Time(t, format="mjd").to_datetime() for t in data["mjd"]]
RV = data["RV"]
RV_err = data["u_RV"]


p0 = [100, 2/10000, 10, 1193471886]

params, errs = curve_fit(sinus,
                         stime,
                         RV,
                         p0=p0,
                         sigma=RV_err)

A, b, h, shift = params
u_A, u_b, u_h, u_shift = np.sqrt(np.diag(errs))

print(f"{np.abs(A)}+-{u_A}")
print(f"{1 / b}+-{u_b / b**2}")
print(f"{time.strftime('%H:%M:%S', time.gmtime(1 / b))}+-{time.strftime('%H:%M:%S', time.gmtime(u_b / b**2))}")

unix_linspace = np.linspace(stime.min(), stime.max(), 1000)
fit_datetimes = [atime.Time(t, format="unix").to_datetime() for t in unix_linspace]

plt.title(f"Radial Velocity over Time\n Gaia EDR3 944390774983674496")
plt.ylabel("Radial Velocity [km/s]")
plt.xlabel("Date")
plt.plot_date(datetimes, RV, xdate=True, zorder=5)
plt.plot(fit_datetimes, sinus(unix_linspace, *params), zorder=2)
plt.errorbar(datetimes, RV, yerr=RV_err, capsize=3, linestyle='', zorder=3)
plt.gcf().autofmt_xdate()
plt.savefig(r"C:\Users\fabia\PycharmProjects\RVVD\output\spec-2682-54401-0569\RV_var_plusfit.png", dpi=300)
plt.show()

