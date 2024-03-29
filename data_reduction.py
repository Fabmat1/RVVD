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

COADD_SIDS = []  # [2806984745409075328]
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

    if bounds is not None:
        # Get rid of littrow ghost
        center_diff = np.median(master) - np.median(master2)
        master2 += center_diff

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
    return a / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + h


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


def get_fluxfraction(image, x_ind, y_ind, width):
    upperfraction = image[int(np.ceil(y_ind - width)) - 1, int(x_ind)] * (np.abs(np.ceil(y_ind - width) - (y_ind - width)))
    lowerfraction = image[int(np.floor(y_ind + width)) + 1, int(x_ind)] * (np.abs(np.floor(y_ind + width) - (y_ind + width)))
    lens = len(image[int(np.ceil(y_ind - width)):int(np.floor(y_ind + width)), int(x_ind)])

    return upperfraction, lowerfraction, lens


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
    # wl_shift = vel_corr/speed_of_light * wl
    # return wl+wl_shift
    return wl / (1 - (vel_corr / (speed_of_light / 1000)))


def fluxstatistics(wl, flux):
    med = median_filter(flux, 5)
    flux_norm = flux / med - 1
    std = pd.Series(flux_norm).rolling(min_periods=1, window=20, center=True).std().to_numpy()

    # plt.plot(flux_norm)
    # plt.plot(3*std)
    # plt.tight_layout()
    # plt.show()

    flux = flux[flux_norm < 3 * std]
    wl = wl[flux_norm < 3 * std]

    med = median_filter(flux, 5)
    flux_norm = flux / med - 1
    std = pd.Series(flux_norm).rolling(min_periods=1, window=20, center=True).std().to_numpy()

    # plt.plot(flux_norm)
    # plt.plot(3 * std)
    # plt.tight_layout()
    # plt.show()

    flx_std = flux * std

    # plt.plot(wl, flux)
    # plt.fill_between(wl, flux-flx_std, flux+flx_std, color="red", alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    return wl, flux, flx_std


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

    # 0.82 to 0.86 Angströms per pixel is usual for SOAR

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

    resids = np.abs(ycenters - line(xcenters, *params))
    outsidestd = resids > 2 * np.std(resids)
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
    uskyflx = np.array([get_flux(image64, p, line(p, *params) + SKYFLUXSEP, width) for p in pixel])
    lskyflx = np.array([get_flux(image64, p, line(p, *params) - SKYFLUXSEP, width) for p in pixel])
    fractions = np.array([get_fluxfraction(image64, p, line(p, *params), width) for p in pixel])
    uf = fractions[:, 0]
    lf = fractions[:, 1]
    lens = fractions[:, 2]

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

    final_wl_arr = wpt.px_to_wl(pixel)

    plt.plot(realcwl, realcflux)
    plt.plot(final_wl_arr, compflux.min() - flux.max() + flux, linewidth=1, color="gray")
    plt.plot(final_wl_arr, compflux, color="darkred")
    plt.plot(final_wl_arr, uf)
    plt.plot(final_wl_arr, lf)
    plt.plot(final_wl_arr, lens)
    plt.show()

    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(mjd, format="mjd"), location=location)
    barycorr = barycorr.to(u.km / u.s)
    barycorr = barycorr.value

    final_wl_arr = wlshift(final_wl_arr, barycorr)

    final_wl_arr, flux, flx_std = fluxstatistics(final_wl_arr, flux)

    return final_wl_arr, flux, flx_std

    # Save to file in other function


def get_star_info(file, catalogue=None):
    if catalogue is None:
        catalogue = pd.read_csv("object_catalogue.csv")

    header = dict(fits.open(file)[0].header)
    try:
        sid = np.int64(header["OBJECT"])
    except ValueError:
        sid = np.int64(header["OBJECT"].replace("Gaia eDR3 ", "").strip())
    sinfo = catalogue[catalogue["source_id"] == sid].iloc[0]
    if os.name == "nt":
        sinfo["file"] = file.split("/")[-1]
    else:
        sinfo["file"] = file.split("/")[-1]

    time = Time(header["DATE-OBS"], format='isot', scale='utc')
    time += TimeDelta(header["EXPTIME"], format='sec')

    return sinfo, time.mjd


def save_to_ascii(wl, flx, flx_std, mjd, trow):
    dir = r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR"
    outtablefile = r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR.csv"
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
    outdata = np.stack((wl, flx, flx_std), axis=-1)
    np.savetxt(fname, outdata, fmt='%1.4f')


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


# You should only need to modify
if __name__ == "__main__":
    print("Starting data reduction...")
    catalogue = pd.read_csv("object_catalogue.csv")
    allfiles = sorted(os.listdir(r"/home/fabian/PycharmProjects/auxillary/spectra_raw/SOAR"))

    print("Searching files...")
    flat_list = []  # Flats
    shifted_flat_list = []  # Flats created with a small camera tilt to get rid of the Littrow ghost
    for file in allfiles:
        if "quartz" in file and "test" not in file and "bias" not in file and "shifted" not in file:
            flat_list.append(r"/home/fabian/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)
        elif "quartz" in file and "test" not in file and "bias" not in file and "shifted" in file:
            shifted_flat_list.append(r"/home/fabian/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

    bias_list = []
    for file in allfiles:
        if "bias" in file and "test" not in file:
            bias_list.append(r"/home/fabian/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file)

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
    if os.path.isfile(r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR.csv"):
        os.remove(r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR.csv")
    for file in allfiles:
        file = r"/home/fabian/PycharmProjects/auxillary/spectra_raw/SOAR" + "/" + file
        if "bias" not in file and "quartz" not in file and "test" not in file and "FeAr" not in file and ".txt" not in file and "RED" not in file:
            compfiles = []  # Complamp list for this file

            if os.name == "nt":
                int_file_index = int(file.split("\\")[-1][:4])
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

            trow, mjd = get_star_info(file, catalogue)  # You probably need to write your own function. Trow needs to be a dict with "ra" and "dec" keys. Mjd is self-explanatory
            print(f'Working on index {int_file_index}, GAIA DR3 {trow["source_id"]}...')
            soardf = pd.concat([soardf, trow])

            cerropachon = EarthLocation.of_site('Cerro Pachon')  # Location of SOAR
            wl, flx, flx_std = extract_spectrum(
                file,
                master_bias,
                master_flat,
                crop,
                master_comp,
                mjd,
                cerropachon,
                trow["ra"],
                trow["dec"])
            save_to_ascii(wl, flx, flx_std, mjd, trow)  # You probably need to write your own function for saving the wl and flx

    # You can ignore everything below, this is only for Coadding spectra.
    if len(COADD_SIDS) > 0:
        directory = r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR"
        labeltable = pd.read_csv(r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR.csv")

        notincoaddtable = labeltable[~labeltable["source_id"].isin(COADD_SIDS)]
        notincoaddtable.to_csv(r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR.csv", index=False)

        for sid in COADD_SIDS:
            thissidlist = labeltable[labeltable["source_id"] == sid]
            filelist = thissidlist["file"].to_numpy()
            for_coadd = split_given_size(filelist, N_COADD)
            for coadd_list in for_coadd:
                n_file = coadd_list[0].replace(".fits", "_01.txt")
                trow, _ = get_star_info(r"/home/fabian/PycharmProjects/auxillary/spectra_processed/SOAR" + "/" + coadd_list[0])
                mjds = []
                for c in coadd_list:
                    with open(directory + "/" + c.replace(".fits", "_mjd.txt")) as mjdfile:
                        mjds.append(float(mjdfile.read()))  #
                mean_mjd = np.mean(mjds)

                allflx = []
                allwl = []
                all_flx_std = []
                for f in coadd_list:
                    wl, flx, t, flx_std = load_spectrum(directory + "/" + f.replace(".fits", "_01.txt"))
                    allwl.append(wl)
                    allflx.append(flx)
                    all_flx_std.append(flx_std)

                allwl = np.vstack(allwl)
                allflx = np.vstack(allflx)
                all_flx_std = np.vstack(all_flx_std)

                allwl = np.mean(allwl, axis=0)
                allflx = np.sum(allflx, axis=0) / len(allflx)
                all_flx_std = np.sum(all_flx_std, axis=0) / len(all_flx_std)

                save_to_ascii(allwl, allflx, all_flx_std, mean_mjd, trow)
