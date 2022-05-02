from astroquery.sdss import SDSS
from astropy import coordinates as coords

pos = coords.SkyCoord('00h03m07.06s +24d12m11.61s', frame='icrs')

xid = SDSS.query_region(pos, spectro=True)

sp = SDSS.get_spectra(matches=xid)

for s in sp:
    s.writeto(f"data/sdss_spec{sp.index(s)}.fits")
