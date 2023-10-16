import ast
import os

import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from astropy.timeseries import LombScargle
from astropy.visualization import astropy_mpl_style, quantity_support

from analyse_results import vrad_pvalue
from fit_rv_curve import phasefold_tiny
from local_TESS_LS import do_tess_stuff

quantity_support()
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.coordinates import get_body, get_sun

# Big words: catalogue, study, indeterminate, irrelevant
# Prefixes, Suffixes: OG, HV, RV, WD, phot, spec, pulsation
paper_associations = {
    "2022A&A...662A..40C": "catalogue",
    "2019A&A...621A..38G": "catalogue",
    "2020A&A...635A.193G": "catalogue",
    "2017A&A...600A..50G": "catalogue",
    "2021ApJS..256...28L": "catalogue_RV",
    "2015MNRAS.448.2260G": "WD_catalogue",
    "2011MNRAS.417.1210G": "WD_catalogue",
    "2019MNRAS.482.4570G": "WD_catalogue",
    "2021MNRAS.508.3877G": "WD_catalogue",
    "2019ApJ...881....7L": "catalogue_RV",
    "2019MNRAS.486.2169K": "WD_catalogue",
    "2015MNRAS.446.4078K": "WD_catalogue",
    "2017ApJ...845..171B": "pulsation_catalogue",
    "2006ApJS..167...40E": "OG_WD_catalogue",
    "1986ApJS...61..305G": "OG_catalogue",
    "2017MNRAS.469.2102A": "WD_catalogue_RV",
    "2013ApJS..204....5K": "WD_catalogue",
    "1988SAAOC..12....1K": "OG_catalogue",
    "2019ApJ...881..135L": "catalogue",
    "2021A&A...650A.205V": "phot_study",
    "2020MNRAS.491.2280S": "irrelevant",
    "2016yCat....1.2035M": "WD_catalogue",
    "2020ApJ...901...93B": "irrelevant",
    "2016MNRAS.455.3413K": "catalogue",
    "2018ApJ...868...70L": "catalogue_RV",
    "2003AJ....126.1455S": "OG_spec_study",
    "2023ApJ...942..109L": "spec_study",
    "2004ApJ...607..426K": "OG_WD_catalogue",
    "2009yCat....1.2023S": "irrelevant",
    "2015MNRAS.452..765G": "WD_catalogue",
    "2008AJ....136..946M": "OG_variability_study",
    "2004A&A...426..367M": "OG_variability_study",
    "2020ApJ...889..117L": "catalogue",
    "2020ApJ...898...64L": "catalogue_RV",
    "2016ApJ...818..202L": "catalogue",
    "2022A&A...661A.113G": "variability_study",
    "1992AJ....104..203W": "OG_catalogue",
    "2016MNRAS.457.3396P": "spec_catalogue",
    "2015A&A...577A..26G": "variability_study",
    "2010A&A...513A...6O": "pulsation_study",
    "2021A&A...654A.107C": "catalogue",
    "2010MNRAS.407..681M": "WD_catalogue_RV",
    "2007ApJ...660..311B": "OG_HV_study",
    "2011A&A...530A..28G": "variability_study",
    "1957BOTT....2p...3I": "OG_catalogue",
    "2015MNRAS.450.3514K": "variability_study",
    "2013A&A...551A..31D": "irrelevant",
    "2019MNRAS.482.5222T": "WD_catalogue",
    "2012MNRAS.427.2180N": "spec_catalogue",
    "1997A&A...317..689T": "OG_catalogue_RV",
    "2005RMxAA..41..155S": "OG_catalogue",
    "2012ApJ...751...55B": "HV_study_RV",
    "2000A&AS..147..169B": "irrelevant",
    "1977A&AS...28..123B": "OG_catalogue",
    "2010AJ....139...59B": "irrelevant",
    "2015A&A...576A..44K": "variability_study",
    "2008ApJ...684.1143X": "irrelevant",
    "2019MNRAS.490.3158C": "irrelevant",
    "2017MNRAS.472.4173R": "WD_catalogue_RV",
    "2018MNRAS.475.2480P": "catalogue_RV",
    "2019MNRAS.488.2892P": "WD_catalogue",
    "2013MNRAS.429.2143C": "variability_study",
    "1980AnTok..18...55N": "OG_catalogue",
    "1984AnTok..20..130K": "OG_catalogue",
    "2011ApJ...730..128T": "WD_spec_catalogue",
    "2019PASJ...71...41L": "spec_catalogue",
    "2016A&A...596A..49B": "catalogue",
    "1990A&AS...86...53M": "OG_catalogue",
    "2007ApJ...671.1708B": "OG_HV_study_RV",
}


def get_frame(date='2023-10-12 00:00:00', utc_offset=-4):
    # Location of the SOAR telescope
    location = EarthLocation.of_site("Cerro Pachon")

    utcoffset = utc_offset * u.hour
    midnight = Time(date) - utcoffset

    delta_midnight = np.linspace(-8, 8, 1000) * u.hour
    times_obsnight = midnight + delta_midnight
    frame_obsnight = AltAz(obstime=times_obsnight, location=location)
    sunaltazs_obsnight = get_sun(times_obsnight).transform_to(frame_obsnight)

    moon_obsnight = get_body("moon", times_obsnight)
    moonaltazs_obsnight = moon_obsnight.transform_to(frame_obsnight)

    return delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight


def get_visibility(frame, ra, dec):
    observed_obj = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    obj_altazs_obsnight = observed_obj.transform_to(frame)

    return obj_altazs_obsnight


def plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=None, date=None):
    plt.figure(figsize=(4.8 * 16 / 9, 4.8))
    plt.grid(color="lightgray", zorder=1)
    if date:
        plt.title(f"Visibility for {date}")
    plt.plot(delta_midnight, sunaltazs_obsnight.alt, color='r', label='Sun', zorder=3)
    plt.plot(delta_midnight, moonaltazs_obsnight.alt, color=[0.75] * 3, ls='--', label='Moon', zorder=3)
    plt.plot(delta_midnight, obj_altazs_obsnight.alt, label='Target', color='lime', zorder=3)
    plt.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                     sunaltazs_obsnight.alt < -0 * u.deg, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                     sunaltazs_obsnight.alt < -18 * u.deg, color='k', zorder=0)
    plt.legend(loc='upper left')
    plt.xlim(-8 * u.hour, 8 * u.hour)
    plt.xticks((np.arange(9) * 2 - 8) * u.hour)
    plt.ylim(0 * u.deg, 90 * u.deg)
    plt.xlabel('Hours from EDT Midnight')
    plt.ylabel('Altitude [deg]')
    plt.tight_layout()
    if saveloc:
        plt.savefig(saveloc)
        plt.close()
    else:
        plt.show()


def plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=None):
    plt.figure(figsize=(3, 3))
    plt.grid(color="lightgray", zorder=1)
    # plt.plot(delta_midnight, sunaltazs_obsnight.alt, color='r', label='Sun', zorder=3)
    # plt.plot(delta_midnight, moonaltazs_obsnight.alt, color=[0.75] * 3, ls='--', label='Moon', zorder=3)
    plt.plot(delta_midnight, obj_altazs_obsnight.alt, label='Target', color='lime', zorder=3)
    plt.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                     sunaltazs_obsnight.alt < -0 * u.deg, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                     sunaltazs_obsnight.alt < -18 * u.deg, color='k', zorder=0)
    plt.xlabel("")
    plt.ylabel("")

    plt.xlim(-8 * u.hour, 8 * u.hour)
    plt.ylim(0 * u.deg, 90 * u.deg)

    ax = plt.gca()
    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    plt.axhline(30, color="gold", linestyle="--")
    plt.axhline(20, color="red", linestyle="--")

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if saveloc:
        plt.savefig(saveloc)
        plt.close()
    else:
        plt.show()


def texfigure(figpath, width="0.9"):
    return [r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}\textwidth]{{{figpath}}}",
            r"\end{figure}", ]


def textable(nspec, deltarv, u_deltarv, rvavg, u_rvavg, ra, dec, gmag, sp_class, alias, interestingness, logp):
    table = [
        fr"Alias & {alias} \\",
        fr"Spectral Class & {sp_class} \\",
        fr"RA/DEC & {round(ra, 4)}/{round(dec, 4)} \\",
        fr"G$_{{\mathrm{{mag}}}}$ & {gmag} \\",
        fr"$N_{{\mathrm{{spec}}}}$ & {nspec} \\",
        fr"$\Delta$RV & {round(deltarv, 2)} \pm {round(u_deltarv, 2)} \\",
        fr"RV$_{{\mathrm{{avg}}}}$ & {round(rvavg, 2)} \pm {round(u_rvavg, 2)} \\",
        fr"$\log p$ & {round(logp, 3)} \\"
        fr"Interestingness & {round(interestingness, 3)} \\"
    ]

    return table


def gen_interesting_table(result_params, catalogue, date='2023-10-12 00:00:00', utc_offset=-4):
    interesting_params = pd.DataFrame(
        {
            "source_id": [],
            "alias": [],
            "ra": [],
            "dec": [],
            "gmag": [],
            "spec_class": [],
            "logp": [],
            "deltaRV": [],
            "deltaRV_err": [],
            "RVavg": [],
            "RVavg_err": [],
            "Nspec": [],
            "interestingness": []
        }
    )

    delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight = get_frame(date, utc_offset)

    for i, star in result_params.iterrows():
        print(f"Calculating for star {i + 1}/{len(result_params)}...")
        cat_dict = dict(catalogue.loc[catalogue["source_id"] == star["source_id"]].iloc[0])
        nspec = star["Nspec"]
        deltarv = star["deltaRV"]
        u_deltarv = star["deltaRV_err"]
        rvavg = star["RVavg"]
        u_rvavg = star["RVavg_err"]
        ra = star["ra"]
        dec = star["dec"]
        gmag = cat_dict["gmag"]
        sp_class = cat_dict["SPEC_CLASS"]
        alias = cat_dict["name"]
        sid = star["source_id"]
        logp = star["logp"]

        if nspec == 0:
            continue
        if nspec == 1 and -400 < rvavg < 400:
            continue
        if gmag > 19:
            continue

        if "+" in sp_class:
            continue

        obj_altazs_obsnight = get_visibility(frame_obsnight, ra, dec)

        max_alt = np.array(obj_altazs_obsnight.alt).max()
        time_of_max = delta_midnight[np.argmax(obj_altazs_obsnight.alt)]

        if time_of_max.value > 7. or time_of_max.value < -7.:
            continue

        if max_alt < 30:
            continue

        interestingness = 0

        if logp < -4:
            interestingness += 5
        elif logp < -1.3:
            interestingness += 2
        else:
            interestingness -= 10

        if deltarv > 100:
            interestingness += (deltarv - 100) / 10
        else:
            continue

        interestingness += (15 - gmag)

        interestingness += 5 - nspec

        if interestingness < 5:
            continue

        interesting_params = pd.concat([interesting_params, pd.DataFrame({
            "source_id": [str(sid)],
            "alias": [alias],
            "ra": [ra],
            "dec": [dec],
            "gmag": [gmag],
            "spec_class": [sp_class],
            "logp": [logp],
            "deltaRV": [deltarv],
            "deltaRV_err": [u_deltarv],
            "RVavg": [rvavg],
            "RVavg_err": [u_rvavg],
            "Nspec": [nspec],
            "interestingness": [interestingness]
        })])

        plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/visibility.pdf")
        plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/tiny_visibility.pdf")

    interesting_params = interesting_params.sort_values("interestingness", axis=0, ascending=False)

    preamble = [
        r"\documentclass[12pt]{scrartcl} %[,Schriftgröße]{DIN A4}",
        r"\usepackage[english]{babel} %",
        r"\usepackage[utf8]{inputenc} %",
        r"\usepackage{color,graphicx} %",
        r"\usepackage[T1]{fontenc} %",
        r"\usepackage{array,colortbl} %",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{ifsym} %",
        r"\usepackage[margin=1.0in]{geometry}",
        r"\usepackage[labelfont={sf,it,small,color=gray},textfont={sf,it,small,color=gray},figurewithin=section,tablewithin=section]{caption} %Beschriftungen: kleiner, ohne serifen, kursiv, in grau",
        r"\usepackage{amsmath} %align-Umgebung",
        r"\usepackage{booktabs} %",
        r"\usepackage{caption}",
        r"\usepackage{float} %",
        r"\usepackage{amssymb}",
        r"\usepackage{hyperref}%[colorlinks=true,linkcolor=blue,citecolor=blue]",
        r"\hypersetup{",
        r"     colorlinks=true,",
        r"     linkcolor=black,",
        r"     filecolor=blue,",
        r"     citecolor = black,",
        r"     urlcolor=blue,",
        r"     }",
        r"\usepackage{comment}%Mehrzeilige Kommentare",
        r"\usepackage{url}",
        r"\usepackage{enumitem}  ",
        r"\usepackage{wrapfig}",
        r"\usepackage{subcaption}",
        r"\usepackage{makecell}",
        r"\usepackage{tabularx}",
        r"",
        r"\setlength{\parindent}{0pt}",
        r"\newcommand*{\astrosun}{{\odot}}",
        r"",
        r"\usepackage{fancyhdr}",
        r"",
        r"",
        r"\fancyhead[L]{Overview over interesting hot subdwarf targets}",
        r"",
        rf"\fancyhead[R]{{For {date.split(' ')[0]}}}",
        r"\fancyfoot{}",
        r"\fancyfoot[R]{\textcolor{gray}{\thepage}}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"",
        r"",
        r"\begin{document}",
        r"\pagestyle{fancy}",
    ]

    tablestart = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabularx}{0.75\textwidth}{XX}",
        r"\textbf{Parameter} & \textbf{Value} \\ \hline"
    ]

    tableend = [
        r"\end{tabularx}",
        r"\end{table}",
    ]

    print("Generating pdf...")
    with open("interesting_doc.tex", "w") as outtex:
        for line in preamble:
            outtex.write(line + "\n")
        for i, star in interesting_params.iterrows():
            nspec = star["Nspec"]
            deltarv = star["deltaRV"]
            u_deltarv = star["deltaRV_err"]
            rvavg = star["RVavg"]
            u_rvavg = star["RVavg_err"]
            ra = star["ra"]
            dec = star["dec"]
            gmag = star["gmag"]
            sp_class = star["spec_class"]
            alias = star["alias"]
            sid = star["source_id"]
            logp = star["logp"]
            interestingness = star["interestingness"]

            for line in texfigure(f"output/{star['source_id']}/RV_variation_broken_axis.pdf"):
                outtex.write(line + "\n")
            for line in tablestart:
                outtex.write(line + "\n")

            for line in textable(nspec, deltarv, u_deltarv, rvavg, u_rvavg, ra, dec, gmag, sp_class, alias, interestingness, logp):
                outtex.write(line + "\n")

            for line in tableend:
                outtex.write(line + "\n")
            for line in texfigure(f"output/{star['source_id']}/visibility.pdf"):
                outtex.write(line + "\n")
            outtex.write(r"\newpage" + "\n")

        outtex.write("\end{document}" + "\n")

    os.system("lualatex interesting_doc.tex")


def quick_visibility(date, ra, dec, saveloc=None):
    delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight = get_frame(date)
    obj_altazs_obsnight = get_visibility(frame_obsnight, ra, dec)
    plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=saveloc)


def quick_tiny_visibility(date, ra, dec, custom_saveloc=None):
    delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight = get_frame(date)
    obj_altazs_obsnight = get_visibility(frame_obsnight, ra, dec)
    plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, custom_saveloc)


def get_paper_url(bibcode):
    base_url = "http://adsabs.harvard.edu/abs/"
    return base_url + bibcode


def colortext(text, color):
    return fr"\textcolor{{{color}}}{{{text}}}"


def flagtex(flags):
    outstring = r"\, "
    if "OG" in flags:
        outstring += colortext("\\ This star was mentioned in old papers", "BrickRed")
    if "HV" in flags:
        outstring += colortext("\\ This star was mentioned in a HV paper", "BrickRed")
    if "RV" in flags:
        outstring += colortext("\\ This star might have previous RV measurements", "BrickRed")
    if "WD" in flags:
        outstring += colortext("\\ This star might be a WD", "BrickRed")
    if "phot" in flags:
        outstring += colortext("\\ This star might have a photometrically found partner", "BrickRed")
    if "spec" in flags:
        outstring += colortext("\\ This star might have a spectroscopically found partner", "BrickRed")
    if "pulsation" in flags:
        outstring += colortext("\\ This star might be a pulsator", "OliveGreen")
    if "HV-detection" in flags:
        outstring += colortext("\\ This star has a high mean velocity", "OliveGreen")
    if "rvv-detection" in flags:
        outstring += colortext("\\ This star is RV variable", "OliveGreen")
    if "rvv-candidate" in flags:
        outstring += colortext("\\ This star might be RV variable", "OliveGreen")
    return outstring


def do_tess_stuff_smartly(tic, source_id, folder="lightcurves", plotit=False):
    if os.path.isfile(f"lightcurves/{tic}.log"):
        print("Old log found, reading...")
        with open(f"lightcurves/{tic}.log", "r", encoding="utf-8") as file:
            for line in file.readlines():
                if "Period" in line:
                    try:
                        period = float(line.split("hours")[0].split("=")[1].strip())
                        return period/2
                    except:
                        pass
    else:
        return do_tess_stuff(tic, source_id, folder="lightcurves", plotit=False)


def tiny_tess_plot(gaia_id, lc_datafile, period):
    plt.close()
    plt.cla()
    plt.clf()
    data = pd.read_csv(lc_datafile, delim_whitespace=True)

    p = data["Period[h]"].to_numpy()
    power = data["Power"].to_numpy()

    plt.figure(figsize=(6, 3))
    plt.title("Period = %5.2f h" % period)
    plt.plot(p, power, color='k')
    plt.xlim(min(p), max(p))
    plt.axvline(period, color='r', ls='--', zorder=0)
    plt.xscale('log')
    plt.xlabel('P [h]')
    plt.ylabel('Power')

    plt.savefig(f"output/{gaia_id}/tess_periodogram.pdf")


def placeholders(text, location):
    plt.close()
    plt.cla()
    plt.clf()
    plt.figure(figsize=(6, 3))

    ax1 = plt.gca()
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    p = plt.Rectangle((left, bottom), width, height, fill=False)
    p.set_transform(ax1.transAxes)
    p.set_clip_on(False)
    ax1.add_patch(p)

    ax1.text(0.5 * (left + right), 0.5 * (bottom + top), text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax1.transAxes,
                 color="red")


    plt.savefig(location)
    plt.cla()
    plt.clf()


def make_pdf(interesting_params, saveloc="interesting_doc.tex", get_lc=False, try_fold=False):
    preamble = [
        r"\documentclass[12pt]{scrartcl}",
        r"\usepackage[english]{babel}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{color, graphicx}",
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{array, colortbl} ",
        r"\usepackage{ifsym} ",
        r"\usepackage[margin = 0 in]{geometry}",
        r"\usepackage{float} ",
        r"\usepackage{hyperref}[colorlinks = true, linkcolor = blue, citecolor = blue]",
        r"\hypersetup{colorlinks = true,linkcolor = black,filecolor = blue,citecolor = black,urlcolor = blue,}",
        r"\usepackage{tabularx}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
    ]

    tablestart = [
        r"\begin{minipage}{.5\textwidth}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabularx}{0.9\textwidth}{XX}",
        r"\textbf{Parameter} & \textbf{Value} \\ \hline"
    ]

    tableend = [
        r"\end{tabularx}",
        r"\end{table}",
        r"\end{minipage}%"
    ]

    bibstart = [
        r"\begin{minipage}{.5\textwidth}"
        r"\begin{minipage}{.9\linewidth}"
        r"\raggedleft"
        r"\textbf{Papers this star was mentioned in:}"
    ]

    print("Generating pdf...")
    with open(saveloc, "w") as outtex:
        for line in preamble:
            outtex.write(line + "\n")
        for i, star in interesting_params.iterrows():
            gaia_id = star["source_id"]
            print("\n")
            print(f"======[{gaia_id}]======")
            print(f"Page {i + 1}/{len(interesting_params)}")
            bibcodes = star["bibcodes"]
            bibcodes = ast.literal_eval(bibcodes)
            tic = star["tic"]
            flags = star["flags"]
            flags = ast.literal_eval(flags)
            nspec = star["Nspec"]
            deltarv = star["deltaRV"]
            u_deltarv = star["deltaRV_err"]
            rvavg = star["RVavg"]
            u_rvavg = star["RVavg_err"]
            ra = star["ra"]
            dec = star["dec"]
            gmag = star["gmag"]
            sp_class = star["spec_class"]
            alias = star["alias"]
            sid = star["source_id"]
            logp = star["logp"]
            interestingness = star["interestingness"]

            period = None
            rv_success = True

            if nspec > 4 and logp < -1.3 and try_fold:
                if not os.path.isfile(f"output/{gaia_id}/tinyphfold_rv.pdf"):
                    rvtable = pd.read_csv(f"output/{gaia_id}/" + "/RV_variation.csv")
                    vels = rvtable["culum_fit_RV"].to_numpy()
                    verrs = rvtable["u_culum_fit_RV"].to_numpy()
                    times = rvtable["mjd"].to_numpy()

                    max_period = 5
                    min_period = 0.5 / 24

                    min_frequency = 1 / max_period
                    max_frequency = 1 / min_period

                    # Calculate the Lomb-Scargle periodogram
                    try:
                        print("Calculating Periodogram...")
                        frequency, power = LombScargle(times, vels, verrs, nterms=2).autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)

                        # Convert frequency to period
                        periods = 1 / frequency

                        period = periods[np.argmax(power)]

                        phasefold_tiny(vels, verrs, times, period, gaia_id,
                                       predetermined=False,
                                       custom_saveloc=f"output/{gaia_id}/tinyphfold_rv.pdf",
                                       custom_title="RV curve phasefolded for RV curve period")
                    except np.linalg.LinAlgError:
                        rv_success = False

            else:
                print(f"Not enough samples! ({nspec})")

                if os.path.isfile(f"output/{gaia_id}/tinyphfold_rv.pdf"):
                    os.remove(f"output/{gaia_id}/tinyphfold_rv.pdf")

                rv_success = False

            lc_success = True
            if get_lc and nspec > 4 and logp < -1.3:
                if not os.path.isfile(f"output/{gaia_id}/tinyphfold_tess.pdf"):
                    try:
                        period = do_tess_stuff_smartly(tic, gaia_id, plotit=False)
                        tiny_tess_plot(gaia_id, f"output/{gaia_id}/ls.dat", period)
                        period *= 2
                        if period:
                            rvtable = pd.read_csv(f"output/{gaia_id}/" + "/RV_variation.csv")
                            vels = rvtable["culum_fit_RV"].to_numpy()
                            verrs = rvtable["u_culum_fit_RV"].to_numpy()
                            times = rvtable["mjd"].to_numpy()
                            phasefold_tiny(vels, verrs, times, period/24, gaia_id,
                                           predetermined=False,
                                           custom_saveloc=f"output/{gaia_id}/tinyphfold_tess.pdf",
                                           custom_title="RV curve phasefolded for 2x TESS period")
                    except Exception as e:
                        print("Problem encountered!")
                        print(e)
                        lc_success = False
            else:
                if os.path.isfile(f"output/{gaia_id}/tinyphfold_tess.pdf"):
                    os.remove(f"output/{gaia_id}/tinyphfold_tess.pdf")
                lc_success = False


            outtex.write(r"\pagestyle{empty}")
            outtex.write(flagtex(flags))
            for line in texfigure(f"output/{star['source_id']}/RV_variation_broken_axis.pdf"):
                outtex.write(line + "\n")
            for line in tablestart:
                outtex.write(line + "\n")

            for line in textable(nspec, deltarv, u_deltarv, rvavg, u_rvavg, ra, dec, gmag, sp_class, alias, interestingness, logp):
                outtex.write(line + "\n")

            for line in tableend:
                outtex.write(line + "\n")
            for line in bibstart:
                outtex.write(line + "\n")

            for i, bibc in enumerate(bibcodes):
                linksafe = bibc.replace('&', r'\&')
                if i < 10:
                    outtex.write(f"\\href{{https://ui.adsabs.harvard.edu/abs/{bibc}}}{{{linksafe}}}\\\\\n")
                else:
                    if i == 10:
                        outtex.write(f"and {len(bibcodes)-10} more")
                        outtex.write(f"\\href{{https://ui.adsabs.harvard.edu/abs/{bibc}}}{{.}}\n")
                    else:
                        outtex.write(f"\\href{{https://ui.adsabs.harvard.edu/abs/{bibc}}}{{.}}\n")
            outtex.write("\n")
            outtex.write("\\end{minipage}\n\\end{minipage}\n")
            outtex.write("\\begin{minipage}{0.3333\\textwidth}\n")
            outtex.write("\\centering\n")
            outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{output/{star['source_id']}/tiny_visibility.pdf}}\n")
            outtex.write("\\end{minipage}%\n")
            outtex.write("\\begin{minipage}{0.6666\\textwidth}\n")
            outtex.write("\\centering\n")
            if lc_success:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{output/{star['source_id']}/tess_periodogram.pdf}}\n")
            else:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{no_tess_placeholder.png}}\n")
            outtex.write("\\end{minipage}\\\\\n")
            outtex.write("\\begin{minipage}{0.5\\textwidth}\n")
            outtex.write("\\centering\n")
            if lc_success:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{output/{star['source_id']}/tinyphfold_tess.pdf}}\n")
            else:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{no_tess_placeholder.png}}\n")
            outtex.write("\\end{minipage}%\n")
            outtex.write("\\begin{minipage}{0.5\\textwidth}\n")
            outtex.write("\\centering\n")
            if rv_success:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{output/{star['source_id']}/tinyphfold_rv.pdf}}\n")
            else:
                outtex.write(f"\\includegraphics[width=0.9\linewidth,height=6.5cm,keepaspectratio]{{too_few_RV_placeholder.png}}\n")
            outtex.write("\\end{minipage}\n")
            outtex.write(r"\newpage" + "\n")

        outtex.write("\end{document}" + "\n")

    os.system(f"lualatex -interaction=nonstopmode {saveloc}")


def create_one_stop_show(catalogue, result_params, date='2023-10-3 00:00:00', utc_offset=-4):
    if not os.path.isfile("interesting_params.csv"):
        interesting_params = pd.DataFrame(
            {
                "source_id": [],
                "alias": [],
                "ra": [],
                "dec": [],
                "gmag": [],
                "spec_class": [],
                "logp": [],
                "deltaRV": [],
                "deltaRV_err": [],
                "RVavg": [],
                "RVavg_err": [],
                "Nspec": [],
                "interestingness": [],
                "bibcodes": [],
                "tic": [],
                "known_category": [],
                "flags": [],
                "maxtime": []
            }
        )

        delta_midnight, frame_obsnight, sunaltazs_obsnight, moonaltazs_obsnight = get_frame(date, utc_offset)

        for i, star in result_params.iterrows():
            print(f"Calculating for star {i + 1}/{len(result_params)}...")
            cat_dict = dict(catalogue.loc[catalogue["source_id"] == star["source_id"]].iloc[0])
            nspec = star["Nspec"]
            deltarv = star["deltaRV"]
            u_deltarv = star["deltaRV_err"]
            rvavg = star["RVavg"]
            u_rvavg = star["RVavg_err"]
            ra = star["ra"]
            dec = star["dec"]
            gmag = cat_dict["gmag"]
            sp_class = cat_dict["SPEC_CLASS"]
            alias = cat_dict["name"]
            sid = star["source_id"]
            logp = star["logp"]
            bibcodes = star["bibcodes"]
            tic = star["tic"]

            flags = []
            known_category = "unknown"
            interestingness = 0

            if nspec == 0:
                known_category = "indeterminate"
                interestingness -= 5000
            if nspec == 1 and -250 < rvavg < 250:
                known_category = "indeterminate"
                interestingness -= 1000

            if np.abs(rvavg) > 250:
                flags.append("HV-detection")

            if gmag > 19:
                pass

            obj_altazs_obsnight = get_visibility(frame_obsnight, ra, dec)

            max_alt = np.array(obj_altazs_obsnight.alt).max()
            time_of_max = delta_midnight[np.argmax(obj_altazs_obsnight.alt)]

            # if time_of_max.value > 7. or time_of_max.value < -7.:
            #     interestingness -= 1000
            #
            # if max_alt < 30:
            #     interestingness -= 1000

            if logp < -4:
                flags.append("rvv-detection")
                interestingness += 5
            elif logp < -1.3:
                flags.append("rvv-candidate")
                interestingness += 2
            else:
                interestingness -= 10

            if deltarv > 100:
                interestingness += (deltarv - 100) / 10
            else:
                interestingness -= 5

            interestingness += (15 - gmag)

            # interestingness += 5 - nspec

            bibcodes = bibcodes.split(";")

            if len(bibcodes) >= 10:
                known_category = "likely_known"

            # Big words: catalogue, study, indeterminate, irrelevant
            # Prefixes, Suffixes: OG, HV, RV, WD, phot, spec, pulsation

            bibcode_associations = []

            for b in bibcodes:
                try:
                    bibcode_associations.append(paper_associations[b])
                except KeyError:
                    known_category = "indeterminate"

            for association in bibcode_associations:
                if association == "variability_study":
                    known_category = "known"
                    break
                if "study" in association:
                    known_category = "likely_known"
                for flg in ["OG", "HV", "RV", "WD", "phot", "spec", "pulsation"]:
                    if flg in association.split("_"):
                        known_category = "likely_known"
                        if flg not in flags:
                            flags.append(flg)

            if "+" in sp_class:
                known_category = "known"

            interesting_params = pd.concat([interesting_params, pd.DataFrame({
                "source_id": [str(sid)],
                "alias": [alias],
                "ra": [ra],
                "dec": [dec],
                "gmag": [gmag],
                "spec_class": [sp_class],
                "logp": [logp],
                "deltaRV": [deltarv],
                "deltaRV_err": [u_deltarv],
                "RVavg": [rvavg],
                "RVavg_err": [u_rvavg],
                "Nspec": [nspec],
                "interestingness": [interestingness],
                "bibcodes": [bibcodes],
                "tic": [tic],
                "known_category": [known_category],
                "flags": [flags],
                "maxtime": [time_of_max]
            })])

            plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/visibility.pdf", date=date)
            # plot_visibility_tiny(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=f"./output/{sid}/tiny_visibility.pdf")

        interesting_params = interesting_params.sort_values("interestingness", axis=0, ascending=False)

        interesting_params.to_csv("interesting_params.csv", index=False)

    else:
        interesting_params = pd.read_csv("interesting_params.csv")

    known_params = interesting_params[interesting_params["known_category"] == "known"].reset_index()
    unknown_params = interesting_params[interesting_params["known_category"] == "unknown"].reset_index()
    indeterminate_params = interesting_params[interesting_params["known_category"] == "indeterminate"].reset_index()
    lk_params = interesting_params[interesting_params["known_category"] == "likely_known"].reset_index()

    make_pdf(lk_params, "likelyknown_stars.tex", False, False)
    make_pdf(unknown_params, "unknown_stars.tex", True, True)
    make_pdf(indeterminate_params, "indeterminate_stars.tex", True, True)
    make_pdf(known_params, "known_stars.tex", True, True)





if __name__ == "__main__":
    placeholders("Sorry, no TESS data found!", "no_tess_placeholder.png")
    placeholders("Too few datapoints!", "too_few_RV_placeholder.png")
    result_params = pd.read_csv("result_parameters.csv")
    catalogue = pd.read_csv("all_objects_withlamost.csv")

    create_one_stop_show(catalogue, result_params, date='2023-10-3 00:00:00', utc_offset=-4)
    # quick_tiny_visibility('2023-10-3 00:00:00', 230.46649284665, -0.39995045926, "tinytest.png")

    # rp = pd.read_csv("result_parameters.csv")
    # cat = pd.read_csv("all_objects_withlamost.csv")
    #
    # gen_interesting_table(rp, cat)
