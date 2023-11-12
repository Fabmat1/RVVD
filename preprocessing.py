import os
from tkinter import messagebox

import numpy as np
import pandas
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

header = ['name', 'source_id', 'ra', 'dec', 'file', 'SPEC_CLASS', 'bp_rp', 'gmag', 'nspec']

def load_catalogues():
    known_catalogue = pd.read_csv("catalogues/sd_catalogue_v56_pub.csv")
    candidate_catalogue = fits.open("catalogues/hotSD_gaia_edr3_catalogue.fits")
    candidate_catalogue = Table.read(candidate_catalogue[1])
    candidate_catalogue = candidate_catalogue.to_pandas()
    candidate_catalogue["source_id"] = candidate_catalogue["source_id"].astype("U64")
    return known_catalogue, candidate_catalogue

try:
    known_catalogue, candidate_catalogue = load_catalogues()
except FileNotFoundError:
    known_catalogue, candidate_catalogue = None, None

n = 1


def preprocessing(fendassociations, prep_settings):
    dataframes = []
    wrn = False
    for fend, association in fendassociations:
        flist = []
        for f in os.listdir("spectra_raw"):
            if f.endswith(fend):
                flist.append(f)
        if association == "Generic ASCII":
            ga, wrn = generic_ascii(flist, prep_settings["coordunit"])
            dataframes.append(ga)
        elif association == "Generic FITS":
            dataframes.append(generic_fits(flist))
        elif association == "LAMOST Low Resolution":
            dataframes.append(low_res_lamost(flist))
        elif association == "LAMOST Medium Resolution":
            dataframes.append(med_res_lamost(flist))

    final_df = pd.concat(dataframes)
    final_df.to_csv("object_catalogue.csv", index=False)
    if wrn:
        messagebox.showwarning("Some coordinates could not be associated with a star", "Some coordinates could not be associated with a star.\nPlease make sure your RA and DEC header parameters are given in degrees, or give the star's gaia id in the "
                                                                                       "source_id parameter, and remember that this program was made with only hot subdwarfs in mind.")


def clean_string(string):
    removables = ["(", ")", ",", "'", '"']

    for r in removables:
        string = string.replace(r, "")

    string = string.strip()

    return string


def radec_to_string(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

    # Format coordinates as strings in the desired format
    ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True)

    return ra_str, dec_str


def getsign(number):
    if number >= 0:
        return "+"
    elif number < 0:
        return "-"


def generic_ascii(filelist, coordunits):
    global known_catalogue, candidate_catalogue
    if known_catalogue is None or candidate_catalogue is None:
        known_catalogue, candidate_catalogue = load_catalogues()
    global n
    warn = False
    data = []
    for y, file in enumerate(filelist):
        source_id = None
        ra = None
        dec = None
        filename = file
        with open(r"./spectra_raw/" + file, "r") as f:
            content = f.readlines()
            for i in content[0].split("), ("):
                if "HJD" in i:
                    with open(r"./spectra_processed/" + file.split(".")[0] + "_mjd.txt", "w") as outfile:
                        outfile.write(clean_string(i.split(",")[-1]))
                i = i.split(",")
                i[0] = clean_string(i[0])
                i[1] = clean_string(i[1])

                if i[0].lower() == "ra":
                    ra = float(i[-1].strip())
                if i[0].lower() == "dec":
                    dec = float(i[-1].strip())
                if i[0].lower() == "source_id":
                    source_id = i[-1].strip()

            candrow = False
            row = None

            if source_id is not None:
                if "GAIA EDR3 " + source_id in known_catalogue["GAIA_DESIG"].values():
                    row = known_catalogue[known_catalogue["GAIA_DESIG"] == "GAIA EDR3 " + source_id].iloc[0]
                elif source_id in candidate_catalogue["source_id"].values():
                    row = candidate_catalogue[candidate_catalogue["source_id"] == source_id].iloc[0]

                    candrow = True
            elif ra is not None and dec is not None:
                if coordunits == "deg,deg":
                    # Compute distances from (ra,dec) to every point in known_catalogue
                    known_distances = np.sqrt((known_catalogue['RA'] - ra) ** 2 + (known_catalogue['DEC'] - dec) ** 2)
                    min_known_distance = known_distances.min()

                    if min_known_distance < 0.025:
                        # If minimal distance is less than threshold, select corresponding row
                        row = known_catalogue.loc[known_distances.idxmin()]

                    else:
                        # Compute distances from (ra,dec) to every point in candidate_catalogue
                        candidate_distances = np.sqrt((candidate_catalogue['ra'] - ra) ** 2 + (candidate_catalogue['dec'] - dec) ** 2)
                        min_candidate_distance = candidate_distances.min()

                        if min_candidate_distance < 0.025:
                            # If minimal distance is less than threshold, select corresponding row
                            row = candidate_catalogue.loc[candidate_distances.idxmin()]
                            candrow = True
                elif coordunits == "h,deg":
                    # Compute distances from (ra,dec) to every point in known_catalogue
                    known_distances = np.sqrt(((known_catalogue['RA'].to_numpy() * u.deg).to_value(u.hourangle) - ra) ** 2 + (known_catalogue['DEC'] - dec) ** 2)
                    min_known_distance = known_distances.min()

                    if min_known_distance < 0.025:
                        # If minimal distance is less than threshold, select corresponding row
                        row = known_catalogue.loc[known_distances.idxmin()]

                    else:
                        # Compute distances from (ra,dec) to every point in candidate_catalogue
                        candidate_distances = np.sqrt(((candidate_catalogue['ra'].to_numpy() * u.deg).to_value(u.hourangle) - ra) ** 2 + (candidate_catalogue['dec'] - dec) ** 2)
                        min_candidate_distance = candidate_distances.min()

                        if min_candidate_distance < 0.025:
                            # If minimal distance is less than threshold, select corresponding row
                            row = candidate_catalogue.loc[candidate_distances.idxmin()]
                            candrow = True
            if row is not None:
                row = row.to_dict()

            with open(r"./spectra_processed/" + file.split(".")[0] + f"_01.txt", "w") as of:
                of.writelines(content[1:])

            if candrow:
                data.append(["-", f"{row['source_id']}", round(row["ra"], 2), round(row["dec"], 4), filename.split(".")[0]+".fits", "unknown", round(row["bp_rp"], 4), round(row["phot_g_mean_mag"], 2), 1])
            elif row is None:
                if ra is not None and dec is not None:
                    searchdf = pandas.DataFrame(data, columns=header)
                    try:
                        star = searchdf[np.logical_and(searchdf.ra == ra, searchdf.dec == dec)].iloc[0]
                    except IndexError:
                        star = None
                    if star is not None:
                        star["file"] = filename
                        data.append(star.tolist())
                    else:
                        rstr, dstr = radec_to_string(ra, dec)
                        data.append([f"J{rstr}{getsign(dec)}{dstr}", f"J{rstr}{getsign(dec)}{dstr}", ra, dec, filename.split(".")[0]+".fits", "unknown", 0, 0, 1])
                        warn = True
                else:
                    data.append([f"unknown star {n}", f"unknown star {n}", ra if ra is not None else 0, dec if dec is not None else 0, filename.split(".")[0]+".fits", "unknown", 0, 0, 1])
                    n += 1
                    warn = True
            else:
                data.append([row["NAME"], row["GAIA_DESIG"].replace("GAIA EDR3 ", ""), round(row["RA"], 4), round(row["DEC"], 4),  filename.split(".")[0]+".fits", row["SPEC_CLASS"], round(row["BP_GAIA"] - row["RP_GAIA"], 2), round(row["G_GAIA"], 2), 1])

    print(data)
    outframe = pd.DataFrame(data=data, columns=header)
    return outframe, warn


def generic_fits(filelist):
    pass


def med_res_lamost(filelist):
    pass


def low_res_lamost(filelist):
    pass
