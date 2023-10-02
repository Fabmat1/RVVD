import os

import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from astropy.visualization import astropy_mpl_style, quantity_support

from analyse_results import vrad_pvalue

quantity_support()
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.coordinates import get_body, get_sun


def get_frame(date='2023-10-3 00:00:00', utc_offset=-4):
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



def plot_visibility(delta_midnight, sunaltazs_obsnight, moonaltazs_obsnight, obj_altazs_obsnight, saveloc=None):
    plt.figure(figsize=(4.8 * 16 / 9, 4.8))
    plt.grid(color="lightgray", zorder=1)
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


def texfigure(figpath):
    return [r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.9\textwidth]{{{figpath}}}",
            r"\end{figure}", ]


def textable(nspec, deltarv, u_deltarv, rvavg, u_rvavg, ra , dec , gmag, sp_class,  alias, interestingness, logp):
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


def gen_interesting_table(result_params, catalogue, date='2023-10-3 00:00:00', utc_offset=-4):
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
        # if gmag > 19:
        #     continue

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

    interesting_params = interesting_params.sort_values("interestingness", axis=0, ascending=False)

    preamble = [
        r"\documentclass[xcolor=dvipsnames, 12pt]{scrartcl} %[,Schriftgröße]{DIN A4}",
        r"\usepackage[english]{babel} %",
        r"\usepackage[utf8]{inputenc} %",
        r"\usepackage{color,graphicx} %",
        r"\usepackage[T1]{fontenc} %",
        r"\usepackage{array,colortbl,xcolor} %",
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
        r"\fancyhead[L]{Overview over interesting subdwarf targets}",
        r"",
        r"\fancyhead[R]{For 10/03/2023}",
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


if __name__ == "__main__":
    rp = pd.read_csv("result_parameters.csv")
    cat = pd.read_csv("all_objects_withlamost.csv")

    gen_interesting_table(rp, cat)
