import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from main import open_spec_files

N_STARS = 100

stars = []
with open("specs_info.txt") as spinfo:
    n = 1
    for star in spinfo:
        if "#" not in star:
            stars.append([
                int(star[2:21].strip()),
                float(star[24:42].strip()),
                float(star[45:65].strip()),
                star[68:].strip()
            ])

startable = Table(np.array(stars), names=["source_id", "ra", "dec", "File"], dtype=['i8', 'f8', "f8", "U25"])
reftable = pd.read_csv("sd_catalogue_v56_pub.csv")

fits_reftable = fits.open("hotSD_gaia_edr3_catalogue.fits")
fits_reftable = Table.read(fits_reftable[1])
fits_reftable = fits_reftable.to_pandas()

for_selection = pd.DataFrame({
    "source_id": [],
    "ra": [],
    "dec": [],
    "file": [],
    "SPEC_CLASS": [],
    "bp_rp": [],
    "gmag": []
})

n = 0
l = len(startable)
dirfiles = os.listdir("spectra/")
for star in startable:
    n += 1
    if n % 100 == 0:
        print(f"{n}/{l}")
    reference = reftable.loc[reftable['GAIA_DESIG'] == f'Gaia EDR3 {star["source_id"]}']
    try:
        spclass = reference["SPEC_CLASS"].iloc[0]
        ra = reference["RA"].iloc[0]
        dec = reference["DEC"].iloc[0]
        bp_rp = reference["BP_GAIA"].iloc[0] - reference["RP_GAIA"].iloc[0]
        gmag = reference["G_GAIA"].iloc[0]
        source_id = str(star["source_id"])
        file = star["File"]
    except IndexError:
        reference = fits_reftable.loc[fits_reftable['source_id'] == star["source_id"]]
        try:
            spclass = "unknown"
            ra = reference["ra"].iloc[0]
            dec = reference["dec"].iloc[0]
            bp_rp = reference["bp_rp"].iloc[0]
            gmag = reference["phot_g_mean_mag"].iloc[0]
            source_id = str(star["source_id"])
            file = star["File"]
        except IndexError:
            continue

    flist = open_spec_files(dirfiles, file.split(".")[0])

    stardata = {
        "source_id": [source_id],
        "ra": [ra],
        "dec": [dec],
        "file": [file],
        "SPEC_CLASS": [spclass],
        "bp_rp": [bp_rp],
        "gmag": [gmag],
        "nspec": [len(flist)]
    }
    for_selection = pd.concat([for_selection, pd.DataFrame(stardata)], ignore_index=True)

for_selection["plot_color"] = "darkgray"
for_selection.loc[(["sdOB" in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "darkred"
for_selection.loc[(["sdO" in c and "sdOB" not in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "red"
for_selection.loc[(["sdB" in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "gold"

final_table = copy.deepcopy(for_selection)

final_table.drop("plot_color", axis=1).to_csv("all_objects.csv", index=False)

print(for_selection)
for_selection = for_selection.groupby(["source_id"], as_index=False).agg({
    'ra': 'first',
    'dec': 'first',
    'file': 'first',
    "SPEC_CLASS": "first",
    'plot_color': 'first',
    'bp_rp': 'first',
    'gmag': 'first',
    "nspec": "sum"})

print(for_selection)

mag = for_selection[["gmag"]].to_numpy()
color = for_selection[["bp_rp"]].to_numpy()
nspec = for_selection[["nspec"]].to_numpy()

plt.scatter(color, mag, 0.7 * nspec ** 2, color=list(for_selection["plot_color"]), alpha=0.7,
            label="_nolegend_")

plt.scatter([], [], color="red")
plt.scatter([], [], color="darkred")
plt.scatter([], [], color="gold")
plt.scatter([], [], color="darkgray")

plt.legend(["sdO", "sdOB", "sdB", "uncategorized"])

plt.title("Hot subdwarfs with single, non-coadded spectra")
plt.xlabel("BP-RP color [mag]")
plt.ylabel("G band magnitude [mag]")
plt.xlim((-0.7, 0.6))
plt.ylim((13, 21))
plt.grid(which='major', axis='y', linestyle='--')
plt.gca().invert_yaxis()
plt.savefig("images/hotsdBs.png", dpi=250)
plt.show()

sdb_subset = for_selection.loc[["sdB" in c or "sdOB" in c for c in for_selection["SPEC_CLASS"]]]
sdb_subset["gmag-nspec"] = sdb_subset["gmag"] - sdb_subset["nspec"] / 2
for_selection = sdb_subset.nsmallest(N_STARS, 'gmag-nspec')

mag = for_selection[["gmag"]].to_numpy()
color = for_selection[["bp_rp"]].to_numpy()
nspec = for_selection[["nspec"]].to_numpy()
print(np.amin(nspec), np.amax(nspec))

plt.scatter(color, mag, 0.7 * nspec ** 2, color=list(for_selection["plot_color"]), alpha=0.7,
            label="_nolegend_")

plt.scatter([], [], color="red")
plt.scatter([], [], color="darkred")
plt.scatter([], [], color="gold")
plt.scatter([], [], color="darkgray")

plt.legend(["sdO", "sdOB", "sdB", "uncategorized"])

plt.title("Initial subset of stars")
plt.xlabel("BP-RP color [mag]")
plt.ylabel("G band magnitude [mag]")
plt.xlim((-0.7, 0.6))
plt.ylim((13, 21))
plt.grid(which='major', axis='y', linestyle='--')
plt.gca().invert_yaxis()
plt.savefig("subset.png", dpi=250)
plt.show()
