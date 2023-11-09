import ctypes
import os
from astropy.time import Time
from matplotlib.colors import Normalize, LogNorm
from scipy.constants import speed_of_light
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time, TimeDelta
from scipy.ndimage import sobel, median_filter, gaussian_filter, minimum_filter
from scipy.interpolate import interp1d
from astropy.io import fits
import cv2
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u
from main import load_spectrum

COADD_SIDS = []#[2806984745409075328]
N_COADD = 2
SKYFLUXSEP = 100


def detect_spectral_area(flats_image):
    # Edge detection
    minumum_truncation = 5

    image_data = flats_image.astype(np.float64)[minumum_truncation:-minumum_truncation, minumum_truncation:-minumum_truncation]

    x_img = sobel(image_data, axis=0, mode="nearest")
    y_img = sobel(image_data, axis=1, mode="nearest")

    edge_detection = np.sqrt(x_img ** 2 + y_img ** 2)
    edge_detection *= 1 / np.max(edge_detection)
    edge_detection[edge_detection > 0.075] = 1
    edge_detection[edge_detection < 1] = 0

    edge_detection = (255 * edge_detection / edge_detection.max()).astype(np.uint8)

    lines = cv2.HoughLinesP(edge_detection, 1, np.pi / 180, 50, None, 500, 0)

    x_intercepts = []
    y_intercepts = []
    # Loop through the detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edge_detection, (x1, y1), (x2, y2), (255,), 1)

        if y2 == y1:
            y_intercepts.append(y1)
            continue

        if x2 == x1:
            x_intercepts.append(x1)
            continue

        m = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - m * x1
        x_intercept = -y_intercept / m if m != 0 else None

        if x_intercept is not None:
            if 0 < x_intercept < edge_detection.shape[0]:
                y_intercepts.append(x_intercept + minumum_truncation)

        if 0 < y_intercept < edge_detection.shape[1]:
            x_intercepts.append(y_intercept + minumum_truncation)

    x_intercepts = np.array(x_intercepts)
    y_intercepts = np.array(y_intercepts)

    u_x = x_intercepts[x_intercepts > edge_detection.shape[1] / 2]
    l_x = x_intercepts[x_intercepts < edge_detection.shape[1] / 2]

    u_y = y_intercepts[y_intercepts > edge_detection.shape[0] / 2]
    l_y = y_intercepts[y_intercepts < edge_detection.shape[0] / 2]

    if len(u_x) == 0:
        u_x = image_data.shape[1] - 5
    else:
        u_x = np.min(u_x) - 5
    if len(l_x) == 0:
        l_x = 5
    else:
        l_x = np.max(l_x) + 5
    if len(u_y) == 0:
        u_y = image_data.shape[0] - 5
    else:
        u_y = np.min(u_y) - 5
    if len(l_y) == 0:
        l_y = 5
    else:
        l_y = np.max(l_y) + 5

    # Display the image using Matplotlib
    # plt.imshow(image_data, zorder=1, cmap="Greys_r")
    # plt.axvline(l_x, color="lime", zorder=5)
    # plt.axvline(u_x, color="lime", zorder=5)
    # plt.axhline(l_y, color="lime", zorder=5)
    # plt.axhline(u_y, color="lime", zorder=5)
    # plt.axis('off')  # Turn off axis labels
    # plt.show()

    return [(l_x, u_x), (l_y, u_y)]


def crop_image(image, xlim, ylim):
    return image[ylim[0]:ylim[1], xlim[0]:xlim[1]]


def create_master_image(image_list, hdu_id, bounds, master_bias=None, master_continuum=None):
    image_data = crop_image(fits.open(image_list[0])[hdu_id].data, *bounds)

    master = np.zeros(image_data.shape, dtype=np.uint32)
    master += image_data
    for image in image_list[1:]:
        image = crop_image(fits.open(image)[hdu_id].data, *bounds)
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]  #
        master += image

    master //= len(image_list)

    if master_continuum is not None:
        master = master.astype(np.float64)
        master /= master_continuum
        master /= master.max()
        master *= 65535

    master = master.astype(np.uint16)
    return master


def create_master_flat(image_list, second_image_list, hdu_id, master_bias=None, bounds=None):
    if bounds is None:
        image_data = fits.open(image_list[0])[hdu_id].data
    else:
        image_data = crop_image(fits.open(image_list[0])[hdu_id].data, *bounds)
    master = np.zeros(image_data.shape, dtype=np.float64)
    master2 = np.copy(master)
    master += image_data
    for image in image_list[1:]:
        image = fits.open(image)[hdu_id].data
        if bounds is not None:
            image = crop_image(image, *bounds)
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]
        master += image

    for image in second_image_list:
        image = fits.open(image)[hdu_id].data
        if bounds is not None:
            image = crop_image(image, *bounds)
        if master_bias is not None:
            image[image < master_bias] = 0
            image[image >= master_bias] = (image - master_bias)[image >= master_bias]
        master2 += image

    # Create smooth master, divide by that to get rid of high frequency noise

    master *= 1 / master.max()
    master2 *= 1 / master2.max()

    master = np.minimum(master, master2)
    # master *= 1 / master.max()

    smooth_master = median_filter(master, 25)

    if bounds is not None:
        master /= smooth_master
        smooth_master /= smooth_master.max()
        return master, smooth_master
    else:
        master /= master.max()
        return master


def gaussian(x, a, mean, std_dev, h):
    return a / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)+h


def line(x, w, v, u, m, n):
    return w * x ** 4 + v * x ** 3 + u * x ** 2 + x * m + n
    # return v * x ** 3 + u * x ** 2 + x * m + n


def get_flux(image, x_ind, y_ind, width):
    fluxsum = np.sum(image[int(np.ceil(y_ind - width)):int(np.floor(y_ind + width)), int(x_ind)])
    upperfraction = image[int(np.ceil(y_ind - width)) - 1, int(x_ind)] * (np.abs(np.ceil(y_ind - width) - (y_ind - width)))
    lowerfraction = image[int(np.floor(y_ind + width)) + 1, int(x_ind)] * (np.abs(np.floor(y_ind + width) - (y_ind + width)))

    fluxsum += upperfraction
    fluxsum += lowerfraction
    return fluxsum


def get_lines_from_intensity(intensity):
    # noirlist = pd.read_csv("FeAr_lines.csv")
    # noirlist["original_wl"] = noirlist.wl
    # noirlist.wl = noirlist.wl.round(2)
    NISTlist = pd.read_csv("FeAr_NIST.csv")
    NISTlist.wl = NISTlist.wl.str.replace('="', '')
    NISTlist.wl = NISTlist.wl.str.replace('"', '')
    NISTlist.wl = NISTlist.wl.astype(np.float64).round(2)
    NISTlist.intens = NISTlist.intens.str.replace('="', '')
    NISTlist.intens = NISTlist.intens.str.replace('"', '')
    NISTlist.intens = pd.to_numeric(NISTlist["intens"], errors='coerce')

    def makeident(elem, num):
        if num == 1:
            return elem.strip() + "I"
        if num == 2:
            return elem.strip() + "II"

    NISTlist["ident"] = NISTlist.apply(lambda x: makeident(x.element, x.sp_num), axis=1)
    # NISTlist["Aki(s^-1)"] = NISTlist["Aki(s^-1)"].str.replace('="', '')
    # NISTlist["Aki(s^-1)"] = NISTlist["Aki(s^-1)"].str.replace('"', '')
    # NISTlist["Aki(s^-1)"] = pd.to_numeric(NISTlist["Aki(s^-1)"], errors='coerce')
    # NISTlist = NISTlist.drop(columns=['Type', 'Unnamed: 7'])
    #
    # merged_df = noirlist.merge(NISTlist, on='wl', how='left')
    # merged_df = merged_df[merged_df['element'].notna()]
    # merged_df = merged_df[merged_df['original_wl'].notna()]
    # merged_df = merged_df[merged_df['Aki(s^-1)'].notna()]
    NISTlist = NISTlist[NISTlist["intens"] >= intensity]

    return NISTlist["wl"].to_numpy(), NISTlist


class WavelenthPixelTransform():
    def __init__(self, wstart, dwdp=None, dwdp2=None, dwdp3=None, dwdp4=None, polyorder=3):
        self.wstart = wstart  # wavelength at pixel 0
        self.dwdp = dwdp  # d(wavelength)/d(pixel)
        self.dwdp2 = dwdp2  # d(wavelength)^2/d(pixel)^2
        self.dwdp3 = dwdp3  # d(wavelength)^3/d(pixel)^3
        self.dwdp4 = dwdp4  # d(wavelength)^4/d(pixel)^4
        self.polyorder = polyorder

    def wl_to_px(self, wl_arr):
        pxspace = np.linspace(0, 2500, 2500)
        f = interp1d(self.px_to_wl(pxspace), pxspace)
        return f(wl_arr)

    def px_to_wl(self, px_arr):
        if self.polyorder == 4:
            return line(px_arr, self.dwdp4, self.dwdp3, self.dwdp2, self.dwdp, self.wstart)
        elif self.polyorder == 3:
            return self.wstart + self.dwdp * px_arr + self.dwdp2 * px_arr ** 2 + self.dwdp3 * px_arr ** 3


def wlshift(wl, vel_corr):
    wl_shift = vel_corr/speed_of_light * wl
    return wl+wl_shift


def extract_spectrum(image_path, master_bias, master_flat, crop, master_comp, mjd, location, ra, dec):
    if os.name == "nt":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    image = fits.open(image_path)[0].data.astype(np.uint16)

    image = crop_image(image, *crop)

    image[image < master_bias] = 0
    image[image >= master_bias] = (image - master_bias)[image >= master_bias]

    image = np.floor_divide(image.astype(np.float64), master_flat)
    image *= 65535 / image.max()
    image = image.astype(np.uint16)

    # plt.imshow(image, cmap="Greys_r", zorder=1, norm=Normalize(vmin=0, vmax=750))
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    ycenters = []
    xcenters = []
    width = []

    # .82~.86 Angströms per pixel

    for i in np.linspace(10, image.shape[1] - 10, 20):
        data = np.min(image[:, int(i - 5):int(i + 5)], axis=1)
        data = median_filter(data, 5)
        xarr = np.arange(len(data))

        params, _ = curve_fit(gaussian,
                              xarr,
                              data,
                              p0=[
                                  np.max(data), len(data) / 2, 2, 0
                              ],
                              maxfev=100000)

        width.append(params[2])
        xcenters.append(int(i))
        ycenters.append(params[1])

    width = 2 * np.mean(width)
    params, _ = curve_fit(line,
                          xcenters,
                          ycenters)

    xcenters = np.array(xcenters)
    ycenters = np.array(ycenters)

    resids = np.abs(ycenters-line(xcenters, *params))
    outsidestd = resids > 2*np.std(resids)
    if np.sum(outsidestd.astype(int)) > 0:
        params, _ = curve_fit(line,
                              xcenters[~outsidestd],
                              ycenters[~outsidestd])


    # xspace = np.linspace(0, image.shape[1], 1000)
    # fig, axs = plt.subplots(2, 1, figsize=(4.8 * 16 / 9, 4.8))
    # axs[0].plot(xspace, line(xspace, *params), zorder=1)
    # axs[0].scatter(xcenters, ycenters, color="red", marker="x", zorder=5)
    # axs[1].imshow(image, cmap="Greys_r", norm=LogNorm(1, 1000))
    # axs[1].plot(xspace, line(xspace, *params), color="lime", linewidth=0.5)
    # axs[1].plot(xspace, line(xspace, *params)-SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, line(xspace, *params)+SKYFLUXSEP, color="red", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, line(xspace, *params)+width, color="lime", linestyle="--", linewidth=0.5)
    # axs[1].plot(xspace, line(xspace, *params)-width, color="lime", linestyle="--", linewidth=0.5)
    # plt.tight_layout()
    # plt.show()

    image64 = image.astype(np.float64)
    master_comp64 = master_comp.astype(np.float64)

    pixel = np.arange(image.shape[1]).astype(np.float64)
    flux = np.array([get_flux(image64, p, line(p, *params), width) for p in pixel])
    compflux = np.array([get_flux(master_comp64, p, line(p, *params), width) for p in pixel])
    uskyflx = np.array([get_flux(image64, p, line(p, *params)+SKYFLUXSEP, width) for p in pixel])
    lskyflx = np.array([get_flux(image64, p, line(p, *params)-SKYFLUXSEP, width) for p in pixel])

    skyflx = np.minimum(uskyflx, lskyflx)
    flux -= skyflx

    realcflux = fits.open("compspec.fits")[0]
    zeropoint = realcflux.header["CRVAL1"]
    delta = realcflux.header["CDELT1"]
    realcflux = realcflux.data
    realcflux *= compflux.max() / realcflux.max()
    realcwl = np.arange(len(realcflux)) * delta + zeropoint

    compflux_cont = minimum_filter(compflux, 10)
    compflux -= compflux_cont

    realcflux = gaussian_filter(realcflux, 3)

    # linelist, linetable = get_lines_from_intensity(5000)

    real_interp = interp1d(realcwl, realcflux)

    def fitwrapper(pixel, wstart, dwdp, dwdp2, dwdp3):
        nonlocal compflux, real_interp
        tsf = WavelenthPixelTransform(wstart, dwdp, dwdp2, dwdp3, polyorder=3)

        wls = tsf.px_to_wl(pixel)

        ref_flux = real_interp(wls)
        residuals = np.sqrt((compflux - ref_flux) ** 2)
        return residuals

    params, errs = curve_fit(
        fitwrapper,
        pixel,
        np.zeros(pixel.shape),
        # p0 = [3605.7455517169524, 0.734833446586154, 0.00025963958918475143, -2.4636866019464887e-07, 7.6347512244437e-11]
        p0=[3589.075643191824, 0.8497538883401163, -7.2492132499996624e-06, 2.007096874944881e-10]
    )

    wpt = WavelenthPixelTransform(*params)

    # print(params, np.sqrt(np.diag(errs)))

    # residgrid = np.zeros([200, 50])
    # for i in range(200):
    #     for j in range(50):
    #         wpt.wstart = 3550 + i/2
    #         wpt.dwdp = 0.73 + j/75
    #         shiftedwl = wpt.px_to_wl(pixel)
    #
    #         reference = interp1d(realcwl, realcflux)
    #
    #         residgrid[i, j] = np.sum(compflux**2-reference(shiftedwl)**2)

    # plt.imshow(residgrid, aspect='auto')
    # plt.show()
    final_wl_arr = wpt.px_to_wl(pixel)

    # plt.plot(realcwl, realcflux)
    # plt.plot(final_wl_arr, compflux.min() - flux.max() + flux, linewidth=1, color="gray")
    # plt.plot(final_wl_arr, compflux, color="darkred")
    # plt.scatter(linelist, intensities/10, s=2, marker="x", color="red")
    # for i, l in enumerate(linelist):
    #     ident = linetable.iloc[i]["ident"].strip()
    #     color = {"FeI": "green",
    #              "FeII": "blue",
    #              "ArI": "red",
    #              "ArII": "orange"}[ident]
    #     plt.axvline(l, .9, .95, color=color, linewidth=np.log(linetable.iloc[i]["intens"])/2)
    # plt.tight_layout()
    # plt.show()

    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(mjd, format="mjd"), location=location)
    barycorr.to(u.km / u.s)
    barycorr = barycorr.value

    final_wl_arr = wlshift(final_wl_arr, barycorr)

    return final_wl_arr, flux

    # Save to file in other function


def get_star_info(file, catalogue=None):
    if catalogue is None:
        catalogue = pd.read_csv("all_objects_withlamost.csv")

    header = dict(fits.open(file)[0].header)
    sid = np.int64(header["OBJECT"])
    sinfo = catalogue[catalogue["source_id"] == sid].iloc[0]
    if os.name == "nt":
        sinfo["file"] = file.split("/")[-1]
    else:
        sinfo["file"] = file.split("/")[-1]

    time = Time(header["DATE-OBS"], format='isot', scale='utc')
    time += TimeDelta(header["EXPTIME"], format='sec')

    return sinfo, time.mjd


def save_to_ascii(wl, flx, mjd, trow):
    dir = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR"
    outtablefile = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv"
    if os.path.isfile(outtablefile):
        outtable = pd.read_csv(outtablefile)
        outtable = pd.concat([outtable, pd.DataFrame([trow])])
    else:
        outtable = pd.DataFrame([trow])
    if not os.path.isdir(dir):
        os.mkdir(dir)

    fname = trow["file"].replace(".fits", "_01.txt")
    fname = dir + "/" + fname

    with open(fname.replace("_01.", "_mjd."), "w") as datefile:
        datefile.write(str(mjd))

    outtable.to_csv(outtablefile, index=False)
    outdata = np.stack((wl, flx, np.zeros(wl.shape)), axis=-1)
    np.savetxt(fname, outdata, fmt='%1.4f')


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


if __name__ == "__main__":
    print("Starting data reduction...")
    catalogue = pd.read_csv("all_objects_withlamost.csv")
    allfiles = sorted(os.listdir(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR"))

    print("Searching files...")
    flat_list = []
    shifted_flat_list = []
    for file in allfiles:
        if "quartz" in file and "test" not in file and "bias" not in file and "shifted" not in file:
            flat_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)
        elif "quartz" in file and "test" not in file and "bias" not in file and "shifted" in file:
            shifted_flat_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

    bias_list = []
    for file in allfiles:
        if "bias" in file and "test" not in file:
            bias_list.append(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

    print("Cropping images...")
    master_flat = create_master_flat(flat_list, shifted_flat_list, 0)
    crop = detect_spectral_area(master_flat)

    # plt.imshow(master_flat, cmap="Greys_r", zorder=1)
    # plt.axvline(crop[0][0], color="lime", zorder=5)
    # plt.axvline(crop[0][1], color="lime", zorder=5)
    # plt.axhline(crop[1][0], color="lime", zorder=5)
    # plt.axhline(crop[1][1], color="lime", zorder=5)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    print("Creating Master Bias...")
    master_bias = create_master_image(bias_list, 0, crop)
    # plt.imshow(master_bias, cmap="Greys_r", zorder=1)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    print("Creating Master Flat...")
    master_flat, master_continuum = create_master_flat(flat_list, shifted_flat_list, 0, master_bias=master_bias, bounds=crop)

    soardf = pd.DataFrame({
        "name": [],
        "source_id": [],
        "ra": [],
        "dec": [],
        "file": [],
        "SPEC_CLASS": [],
        "bp_rp": [],
        "gmag": [],
        "nspec": [],
    })

    print("Extracting Spectra...")
    if os.path.isfile(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv"):
        os.remove(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv")
    for file in allfiles:
        file = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file
        if "bias" not in file and "quartz" not in file and "test" not in file and "FeAr" not in file and ".txt" not in file and "RED" not in file:
            compfiles = []

            if os.name == "nt":
                int_file_index = int(file.split("/")[-1][:4])
            else:
                int_file_index = int(file.split("/")[-1][:4])

            for i in range(6):
                i += 1
                searchind = int_file_index + i
                file_index = str(int_file_index)

                if len(file_index) == 3:
                    file_index = "0" + file_index
                if len(str(searchind)) == 3:
                    searchind = "0" + str(searchind)

                cfile = file.replace(file_index, str(searchind)).replace(".fits", "_FeAr.fits")
                if os.path.isfile(cfile):
                    compfiles.append(cfile)

            master_comp = create_master_image(compfiles, 0, crop, master_bias)

            trow, mjd = get_star_info(file, catalogue)
            print(f'Working on index {int_file_index}, GAIA DR3 {trow["source_id"]}...')
            soardf = pd.concat([soardf, trow])

            cerropachon = EarthLocation.of_site('Cerro Pachon')
            wl, flx = extract_spectrum(
                file,
                master_bias,
                master_flat,
                crop,
                master_comp,
                mjd,
                cerropachon,
                trow["ra"],
                trow["dec"])
            save_to_ascii(wl, flx, mjd, trow)

    if len(COADD_SIDS) > 0:
        directory = r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR"
        labeltable = pd.read_csv(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv")

        notincoaddtable = labeltable[~labeltable["source_id"].isin(COADD_SIDS)]
        notincoaddtable.to_csv(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR.csv", index=False)

        for sid in COADD_SIDS:
            thissidlist = labeltable[labeltable["source_id"] == sid]
            filelist = thissidlist["file"].to_numpy()
            for_coadd = split_given_size(filelist, N_COADD)
            for coadd_list in for_coadd:
                n_file = coadd_list[0].replace(".fits", "_01.txt")
                trow, _ = get_star_info(r"/home/fabian/Documents/PycharmProjects/auxillary/spectra_processed/SOAR" + "/" + coadd_list[0])
                mjds = []
                for c in coadd_list:
                    with open(directory + "/" + c.replace(".fits", "_mjd.txt")) as mjdfile:
                        mjds.append(float(mjdfile.read()))  #
                mean_mjd = np.mean(mjds)

                allflx = []
                allwl = []
                for f in coadd_list:
                    wl, flx, _, _ = load_spectrum(directory + "/" + f.replace(".fits", "_01.txt"))
                    allwl.append(wl)
                    allflx.append(flx)

                allwl = np.vstack(allwl)
                allflx = np.vstack(allflx)

                allwl = np.mean(allwl, axis=0)
                allflx = np.sum(allflx, axis=0) / len(allflx)

                save_to_ascii(allwl, allflx, mean_mjd, trow)
