import numpy as np
from matplotlib import pyplot as plt


MODE = "random"  # random, randspec, randspec_multi

if MODE == "random":
    wl = np.linspace(3566, 10385, 4643)
    flux = np.random.normal(0, 1, 4643)
    final = np.stack((wl, flux), 1)
    np.savetxt("random.csv", final, delimiter=",")

