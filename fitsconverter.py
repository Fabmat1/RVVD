import os
import aotsAPI
import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.table import Table


keyconformtofitsdict = {
    "source_id": "SID",
    "ra": "RA",
    "dec": "DEC",
    "spec_class": "SPCLASS",
    "logp": "LOGP",
    "deltaRV": "DRV",
    "deltaRV_err": "U_DRV",
    "RVavg": "RVAVG",
    "RVavg_err": "U_RVAVG",
    "Nspec": "NSPEC",
    "associated_files": "ASOC_F",
    "timespan": "TSPAN"
}


colnames_correspondence = {
    "culum_fit_RV": "RV",
    "u_culum_fit_RV": "RVERR",
    "mjd": "MJD"
}


def save_to_fits(data, metadata, filename):
    # Convert pandas dataframe to astropy table
    tab = Table.from_pandas(data)
    if len(tab) == 0:
        return

    column_names = {"COLNAMES": ";".join([colnames_correspondence[n] for n in data.columns])}

    # Convert metadata to fits header
    meta_hdr = fits.Header()
    for key, value in metadata.items():
        key = keyconformtofitsdict[key]
        value = value.iloc[0]
        if isinstance(value, float):
            if not np.isnan(value):
                meta_hdr[key] = value
        else:
            meta_hdr[key] = value

    # Create fits hdu from data and metadata
    # Create Primary HDU and set EXTEND keyword to True
    primary_hdu = fits.PrimaryHDU(header=meta_hdr)
    primary_hdu.header['EXTEND'] = True
    data_hdu = fits.BinTableHDU(tab, header=column_names)

    # Create HDU list
    hdul = fits.HDUList([primary_hdu, data_hdu])

    # Write HDU list to fits file
    hdul.writeto(filename, overwrite=True)


reftable = pd.read_csv("result_parameters.csv")

for sid in os.listdir("output"):
    print(sid)
    file = "output/" + sid + "/RV_variation.csv"

    # source_id,ra,dec,spec_class,logp,deltaRV,deltaRV_err,RVavg,RVavg_err,Nspec,associated_files,timespan
    data = pd.read_csv(file)
    metadata = reftable.loc[reftable["source_id"] == int(sid)]

    #save_to_fits(data, metadata,  "output/" + sid + "/RV_variation.fits")
    if os.path.isfile("output/" + sid + "/RV_variation.fits"):
        os.remove("output/" + sid + "/RV_variation.fits")

    success = aotsAPI.processing.preprocess_rvcurve(metadata, data["mjd"].to_numpy(), data["culum_fit_RV"].to_numpy(), data["u_culum_fit_RV"].to_numpy(), data["u_culum_fit_RV"].to_numpy(), "output/" + sid + "/RV_variation.fits")

    # test = fits.open("test.fits")
    #
    # print(dict(test[0].header))
    # print(dict(test[1].header))
    # print(test[1].data)

