import numpy as np
from scipy.constants import h, c, k
from main import pseudo_voigt


MODE = "random"  # random, randspec, randspec_multi

# def planck_law(wl):
#     return 2*h*()


if MODE == "random":
    wl = np.linspace(3566, 10385, 4643)
    flux = np.random.normal(0, 1, 4643)
    final = np.stack((wl, flux), 1)
    np.savetxt("random.csv", final, delimiter=",")
if MODE == "randspec":
    wl = np.linspace(3566, 10385, 4643)


