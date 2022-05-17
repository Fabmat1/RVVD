import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import numpy as np
from astropy.io import fits


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
    "file": [],
    "SPEC_CLASS": [],
    "bp_rp": [],
    "gmag": []
})

for star in startable:
    reference = reftable.loc[reftable['GAIA_DESIG'] == f'Gaia EDR3 {star["source_id"]}']
    try:
        spclass = reference["SPEC_CLASS"].to_numpy()[0]
        bp_rp = reference["BP_GAIA"].to_numpy()[0]-reference["RP_GAIA"].to_numpy()[0]
        gmag = reference["G_GAIA"].to_numpy()[0]
        source_id = str(star["source_id"])
        file = star["File"]
    except IndexError:
        reference = fits_reftable.loc[fits_reftable['source_id'] == star["source_id"]]
        try:
            spclass = "unknown"
            bp_rp = reference["bp_rp"].to_numpy()[0]
            gmag = reference["phot_g_mean_mag"].to_numpy()[0]
            source_id = str(star["source_id"])
            file = star["File"]
        except IndexError:
            continue
    stardata = {
        "source_id": [source_id],
        "file": [file],
        "SPEC_CLASS": [spclass],
        "bp_rp": [bp_rp],
        "gmag": [gmag]
    }
    for_selection = pd.concat([for_selection, pd.DataFrame(stardata)], ignore_index=True)


for_selection["plot_color"] = "black"
for_selection.loc[(["sdOB" in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "darkred"
for_selection.loc[(["sdO" in c and "sdOB" not in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "red"
for_selection.loc[(["sdB" in c for c in for_selection["SPEC_CLASS"]]), "plot_color"] = "gold"

print(for_selection)

ax = for_selection.plot.scatter("bp_rp", "gmag", color=list(for_selection["plot_color"]))
plt.show()

sdb_subset = for_selection.loc[["sdB" in c for c in for_selection["SPEC_CLASS"]]]
final_table = sdb_subset.nsmallest(N_STARS, 'gmag')
final_table.drop("plot_color", axis=1).to_csv("selected_objects.csv", index=False)