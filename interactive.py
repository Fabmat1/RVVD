import ast
import os.path
import shutil
import traceback

import sys

from ztfquery import lightcurve
import astroquery.exceptions
import itertools
import json
from multiprocessing import cpu_count, Manager, active_children
import subprocess
import threading
import tkinter as tk
import webbrowser
from idlelib.tooltip import Hovertip
from queue import Empty
from shutil import which
from tkinter import ttk, filedialog

import _thread
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astroquery.exceptions import InvalidQueryError
from gatspy import periodic
import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from astroquery.vizier import Vizier
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from tksheet import Sheet

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from analyse_results import vrad_pvalue
from data_reduction import *
from get_interesting_stars import quick_visibility
from main import general_config, fit_config, plot_config, interactive_main, plot_rvcurve_brokenaxis, open_spec_files, load_spectrum
from plot_spectra import plot_system_from_ind, plot_system_from_file
from preprocessing import *
import matplotlib
import screeninfo


def get_monitor_from_coord(x, y):
    monitors = screeninfo.get_monitors()

    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return [m.width, m.height]

    return [monitors[0].width, monitors[0].height]


if os.name != "nt":
    matplotlib.use('QtAgg')


def callback(url):
    webbrowser.open_new(url)


configs = [general_config, fit_config, plot_config]

general_tooltips = {
    "SPECTRUM_FILE_SEPARATOR": ["Column Separator", "Separator between columns in the ASCII file"],
    "CATALOGUE": ["Catalogue Location", "The location of the catalogue"],
    "FILE_LOC": ["Spectra Directory", "Directory that holds the spectrum files"],
    "OUTPUT_DIR": ["Output Directory", "Directory where outputs will be saved"],
    "VERBOSE": ["Verbose Output", "Enable/Disable verbose output"],
    "NO_NEGATIVE_FLUX": ["No Negative Flux Check", "Check for negative flux values"],
    "SORT_OUT_NEG_FLX": ["Negative Flux Filter", "Filter out spectra files with significant portions of negative flux"],
    "SUBDWARF_SPECIFIC_ADJUSTMENTS": ["Subdwarf Adjustments", "Apply some tweaks for the script to be optimized to hot subdwarfs"],
    "GET_TICS": ["Get TIC IDs", "Get TIC IDs via query. This will be slow the first time it is run."],
    "GET_VISIBILITY": ["Get Visibility", "Whether to get the visibility of the objects for a certain night and location."],
    "FOR_DATE": ["Visibility Date", "Date for which to get the visibility"],
    "post_progress": ["IGNORE THIS SETTING", "Debug setting please ignore :)"],
    "TAG_KNOWN": ["Tag Known", "Tag systems whose RV variability is known"]
}

fit_tooltips = {
    "OUTLIER_MAX_SIGMA": ["Outlier Maximum Sigma", "Sigma value above which a line from the individual gets rejected as a fit to a wrong line. Outliers do not get used in the cumulative fit."],
    "ALLOW_SINGLE_DATAPOINT_PEAKS": ["Allow Single Datapoint Peaks", "Whether to accept lines that are made up by only one datapoint."],
    "MAX_ERR": ["Maximum Error", "Maximum allowed error above which a RV gets rejected as bad [m/s]"],
    "CUT_MARGIN": ["Cut Margin", "Margin used for cutting out disturbing lines, if their standard deviation was not yet determined [Å]"],
    "MARGIN": ["Window Margin", "Window margin around lines used in determining fits [Å]"],
    "AUTO_REMOVE_OUTLIERS": ["Auto Remove Outliers", "Whether an input from the user is required to remove outliers from being used in the cumulative fit"],
    "MIN_ALLOWED_SNR": ["Minimum SNR", "Minimum allowed SNR to include a line in the cumulative fit"],
    "SNR_PEAK_RANGE": ["SNR Peak Range", "Width of the peak that is considered the \"signal\" [Multiples of the FWHM]"],
    "COSMIC_RAY_DETECTION_LIM": ["Cosmic Ray Alert", "Minimum times peak height/flux std required to detect cr, minimum times diff std required to detect cr"],
    "USE_LINE_AVERAGES": ["Use Line Averages", "Determine guessed FWHM by examining previously fitted lines, not recommended when using multiprocessing!"],
}

plot_tooltips = {
    "FIG_DPI": ["Plot DPI", "DPI value of plots that are created, if they are not pdf files"],
    "PLOT_FMT": ["Plot Format", "File format of plots (.pdf is recommended due to smaller file sizes)"],
    "SHOW_PLOTS": ["Show Plots", "Show matplotlib plotting window for each plot"],
    "PLOTOVERVIEW": ["Plot Overview", "Plot overview of entire subspectrum"],
    "SAVE_SINGLE_IMGS": ["Save Single Images", "Save individual plots of fits as images in the respective folders !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!"],
    "REDO_IMAGES": ["Redo Images", "Redo images already present in folders"],
    "SAVE_COMPOSITE_IMG": ["Save Composite Image", "Save RV-Curve plot"],
    "REDO_STARS": ["Redo Stars Calculation", "Whether to redo stars for which RVs have already be determined"],
    "PLOT_LABELS_FONT_SIZE": ["Plot Labels Font Size", "Label font size"],
    "PLOT_TITLE_FONT_SIZE": ["Plot Title Font Size", "Title font size"],
    "CREATE_PDF": ["Create PDF", "Group all RV-plots into one big .pdf at the end of the calculations !MAY CREATE VERY LARGE FILES FOR BIG DATASETS!"],
}


# from terminedia import ColorGradient


def save_preferences(pref_dict):
    with open(".RVVDprefs", "w") as preffile:
        json.dump(pref_dict, preffile)


def load_preferences():
    if not os.path.isfile(".RVVDprefs"):
        with open(".RVVDprefs", "w") as preffile:
            json.dump({
                "isisdir": "/ISIS_models"
            }, preffile)
    with open(".RVVDprefs", "r") as preffile:
        preferences = json.load(preffile)
    return preferences


# red_green_gradient = ColorGradient([(0, (255, 65, 34)), (0.5, (255, 255, 255)), (1, (92, 237, 115))])


def open_settings(window, queue):
    global configs

    def save_settings():
        for cat in [mod_gc, mod_fc, mod_pc]:
            for key, val in cat.items():
                cat[key] = val.get()

        queue.put(["update_configs", [mod_gc, mod_fc, mod_pc]])
        settings_window.destroy()

    # Open the settings dialogue
    settings_window = tk.Toplevel(window)
    settings_window.protocol("WM_DELETE_WINDOW", save_settings)
    try:
        if os.name == "nt":
            settings_window.iconbitmap("favicon.ico")
        else:
            imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
            settings_window.tk.call('wm', 'iconphoto', settings_window._w, imgicon)
    except:
        pass
    settings_window.title("Settings")
    settings_window.geometry("800x600+0+0")

    # Create the notebook
    notebook = ttk.Notebook(settings_window)

    # Create the first tab
    general = tk.Frame(notebook)
    notebook.add(general, text="General Settings")
    fitting = tk.Frame(notebook)
    notebook.add(fitting, text="Fitting")
    plotting = tk.Frame(notebook)
    notebook.add(plotting, text="Plotting")
    notebook.pack(fill="x", pady=1)

    # Add widgets to the settings dialogue
    mod_gc = {}
    mod_fc = {}
    mod_pc = {}

    for cat, tab, mod_cat, l_and_tt in zip(configs, [general, fitting, plotting], [mod_gc, mod_fc, mod_pc], [general_tooltips, fit_tooltips, plot_tooltips]):
        for key, val in cat.items():
            labeltext, tooltip = l_and_tt[key]
            if isinstance(val, str):
                smolframe = tk.Frame(tab)
                tk_stringval = tk.StringVar(value=val)
                strval = tk.Entry(smolframe, textvariable=tk_stringval)
                l = tk.Label(smolframe, text=labeltext)
                Hovertip(smolframe, tooltip, 250)
                l.pack(side=tk.LEFT)
                strval.pack(side=tk.LEFT)
                smolframe.pack(anchor="w")
                mod_cat[key] = tk_stringval
            elif isinstance(val, bool):
                tk_bool = tk.BooleanVar()
                boolval = tk.Checkbutton(tab, text=labeltext, variable=tk_bool)
                if val:
                    boolval.select()
                Hovertip(boolval, tooltip, 250)
                boolval.pack(anchor="w")
                mod_cat[key] = tk_bool
            elif isinstance(val, float) or isinstance(val, int):
                smolframe = tk.Frame(tab)
                tk_doub = tk.DoubleVar()
                doubval = tk.Entry(smolframe, textvariable=tk_doub)
                l = tk.Label(smolframe, text=labeltext)
                l.pack(side=tk.LEFT)
                Hovertip(smolframe, tooltip, 250)
                doubval.pack(side=tk.LEFT)
                smolframe.pack(anchor="w")
                tk_doub.set(val)
                mod_cat[key] = tk_doub

    save_button = tk.Button(settings_window, text="Save Settings", command=save_settings)
    save_button.pack()


def start_proc(queue):
    print("Starting...")
    if os.path.isfile("object_catalogue.csv"):
        queue.put(["start_process"])
    else:
        messagebox.showwarning("No preprocessed spectra found", "No pre-processed spectra were detected in /spectra_processed.\nPlease preprocess your spectra.")


def cell_colors(sheet, row_indices_full, row_indices_part, col_ind, colors):
    detectioncell_positions = [(r, col_ind) for r in row_indices_full]
    candidatecell_positions = [(r, col_ind) for r in row_indices_part]

    if len(detectioncell_positions) > 0:
        sheet.highlight_cells(cells=detectioncell_positions, bg=colors[0])
    if len(candidatecell_positions) > 0:
        sheet.highlight_cells(cells=candidatecell_positions, bg=colors[1])


def load_or_create_image(window, file_name, size, queue, creation_func=None, remake=False, *args, **kwargs):
    if not os.path.isfile(file_name):
        returnqueue = man.Queue()
        queue.put(["execute_function_with_return", creation_func, (returnqueue, *args), kwargs])
        finished = returnqueue.get()

    # transformation matrix we can apply on pages
    if ".pdf" in file_name:
        doc = fitz.open(file_name)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=mat)
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        im = Image.open(file_name)

    img_tk = ImageTk.PhotoImage(im.resize(size))
    label = tk.Label(window, image=img_tk)
    label.image = img_tk  # Save reference to image
    return label


def construct_table(master, params):
    labels = ["Alias", "Spectral Class", "RA/DEC", "G mag", "N spec", "Delta RV", "RV avg", "log p", "pmra/pmdec", "Parallax", "Spatial Velocity"]

    tframe = tk.Frame(master)

    def copyfnt(event):
        siblings = list(event.widget.master.children.values())
        thisindex = siblings.index(event.widget)
        if thisindex % 2 == 0:
            copyval = siblings[thisindex + 1].cget("text")
        else:
            copyval = siblings[thisindex].cget("text")
        master.clipboard_clear()
        master.clipboard_append(copyval)

    for i, p in enumerate(params):
        label = tk.Label(tframe, text=labels[i])
        if isinstance(params[i], tuple):
            if labels[i] == "RA/DEC" or labels[i] == "pmra/pmdec":
                value = tk.Label(tframe, text=f"{round(p[0], 4)}/{round(p[1], 4)}")
            else:
                value = tk.Label(tframe, text=f"{round(p[0], 2)}±{round(p[1], 2)}")
        else:
            if isinstance(p, float):
                value = tk.Label(tframe, text=f"{round(p, 4)}")
            else:
                value = tk.Label(tframe, text=f"{p}")

        label.bind("<Button-1>", copyfnt)
        value.bind("<Button-1>", copyfnt)
        label.grid(row=i + 1, column=1)
        value.grid(row=i + 1, column=2)

    return tframe


def get_interpolation_function(arr_x, arr_y):
    interpolations = [interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1])) for x, y in zip(arr_x, arr_y)]

    def interpolated_median(x):
        # Evaluate the interpolations at x
        evaluated = [interp(x) for interp in interpolations]

        # Return the median
        return np.median(evaluated, axis=0)

    return interpolated_median


def calcsed(gaia_id, parameter_list, griddirs, isisdir):
    parameter_list = [("c*_xi", "0", 1), ("c*_z", "0", 1)] + parameter_list
    n_list = []
    for a, b, c in parameter_list:
        if b != "":
            n_list.append((a, b, c))
    parameter_list = n_list
    starline = f'variable star = "GAIA DR3 {gaia_id}";'
    paramstring = f'''variable par = struct{{name = ["{'","'.join([p[0] for p in parameter_list])}"],
    value = [{",".join([p[1] for p in parameter_list])}],
    freeze = [{",".join([str(p[2]) for p in parameter_list])}]}};'''
    double_quoted_list = ', '.join([f'"{item}"' for item in griddirs])
    griddirectories = "variable griddirectories, bpaths;\ngriddirectories = [" + double_quoted_list + "];"
    isispath = f'bpaths = ["{str(isisdir)}"];'

    if not os.path.isdir(f"SEDs/{gaia_id}"):
        os.mkdir(f"SEDs/{gaia_id}")
    else:
        os.remove(f"SEDs/{gaia_id}/photometry.sl")
    with open(f"SEDs/{gaia_id}/photometry.sl", "w") as script:
        script.write(starline + "\n")
        script.write(paramstring + "\n")
        script.write(griddirectories + "\n")
        script.write(isispath + "\n")
        with open(f"SEDs/sedscriptbase.sl", "r") as restofthefnowl:
            script.write(restofthefnowl.read())

    p = subprocess.Popen(f"isis photometry.sl", shell=True, cwd=f"SEDs/{gaia_id}")

    return p


def setCanvasSize(window, canvas):
    # Force tkinter to update the GUI, thus calculating the correct screen size
    window.update_idletasks()

    # Setting the canvas size according to frame size
    canvas.configure(width=window.winfo_width() * 0.75, height=window.winfo_height() * 0.9)


def showimages_sed(gaia_id, frame, window):
    canvas = tk.Canvas(frame)
    scrollbar_y = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar_x = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    images = []

    for pdfpath in [f"SEDs/{gaia_id}/photometry_SED.pdf", f"SEDs/{gaia_id}/photometry_results.pdf"]:
        try:
            doc = fitz.open(pdfpath)
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=mat)
            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_tk = ImageTk.PhotoImage(im)
            image_label = tk.Label(scroll_frame, image=img_tk)
            image_label.image = img_tk
            image_label.pack()
            images.append(image_label)
        except fitz.fitz.FileNotFoundError:
            noresults = tk.Label(scroll_frame,
                                 font=('Segoe UI', 25),
                                 fg='#ff0000',
                                 text="No results to show yet!\nGenerate some!")
            noresults.pack(fill="both", expand=1)
            break

    scrollbar_y.pack(side=tk.RIGHT, fill="y")
    scrollbar_x.pack(side=tk.BOTTOM, fill="x")
    canvas.pack(side=tk.LEFT, fill="both", expand=True)

    def scrollHorizontally(event):
        if os.name == 'nt':
            canvas.xview_scroll(-1, "units")
        elif os.name == "posix":
            canvas.xview_scroll(-1, "units")

    def scrollVertically(event):
        if os.name == 'nt':
            canvas.yview_scroll(-1, "units")
        elif os.name == "posix":
            canvas.yview_scroll(-1, "units")

    def scrollHorizontallyInverse(event):
        if os.name == 'nt':
            canvas.xview_scroll(1, "units")
        elif os.name == "posix":
            canvas.xview_scroll(1, "units")

    def scrollVerticallyInverse(event):
        if os.name == 'nt':
            canvas.yview_scroll(1, "units")
        elif os.name == "posix":
            canvas.yview_scroll(1, "units")

    def bind_to_element(element):
        element.bind('<Up>', scrollVertically)
        element.bind('<Down>', scrollVerticallyInverse)
        element.bind('<Left>', scrollHorizontally)
        element.bind('<Right>', scrollHorizontallyInverse)
        element.bind('<Button-4>', scrollVertically)
        element.bind('<Button-5>', scrollVerticallyInverse)
        element.bind('<Shift-Button-4>', scrollHorizontally)
        element.bind('<Shift-Button-5>', scrollHorizontallyInverse)

    for i in images:
        bind_to_element(i)
    bind_to_element(canvas)
    setCanvasSize(frame, window)


def calcpgramsamples(x_ptp, min_p, max_p):
    n = np.ceil(x_ptp / min_p)
    R_p = (x_ptp / (n - 1) - x_ptp / n) / 5

    df = 1 / min_p - (1 / (min_p + R_p))
    return int(np.ceil((1 / min_p - 1 / max_p) / df))

def master_spectrum_window(window, queue, gaia_id, button=None, sheet=None):
    if sheet is not None:
        gaia_id = sheet.data[list(sheet.get_selected_cells())[0][0]][0]

    object_list = pd.read_csv("result_parameters.csv")

    assoc_files = object_list[object_list["source_id"] == gaia_id].iloc[0]["associated_files"].split(";")
    assoc_files = [f.replace(".fits", "") for f in assoc_files]

    create_master_window = tk.Toplevel(window)
    try:
        if os.name == "nt":
            create_master_window.iconbitmap("favicon.ico")
        else:
            imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
            create_master_window.tk.call('wm', 'iconphoto', create_master_window._w, imgicon)
    except:
        pass
    create_master_window.title(f"Create master spectrum for Gaia DR3 {gaia_id}")
    create_master_window.update_idletasks()  # This forces tkinter to update the window calculations.
    create_master_window.geometry("800x600+0+0")

    masspecframe = tk.Frame(create_master_window)

    custom_name_val = tk.StringVar(value="")
    custom_name_entry = tk.Entry(masspecframe, textvariable=custom_name_val)
    custom_name_l = tk.Label(masspecframe, text="Custom Name")
    custom_name_l.grid(row=1, column=1)
    custom_name_entry.grid(row=1, column=2)

    custom_sel_val = tk.StringVar(value="")
    custom_sel_entry = tk.Entry(masspecframe, textvariable=custom_sel_val)
    custom_sel_l = tk.Label(masspecframe, text="Custom Selection")
    custom_sel_l.grid(row=2, column=1)
    custom_sel_entry.grid(row=2, column=2)

    custom_rv_val = tk.StringVar(value="")
    custom_rv_entry = tk.Entry(masspecframe, textvariable=custom_rv_val)
    custom_rv_l = tk.Label(masspecframe, text="Custom RV val")
    custom_rv_l.grid(row=3, column=1)
    custom_rv_entry.grid(row=3, column=2)

    def do_master_thing():
        if custom_name_val.get() == "" and custom_sel_val.get() == "" and custom_rv_val.get() == "":
            create_master_spectrum(queue, gaia_id, assoc_files)
        else:
            create_master_spectrum(queue, gaia_id, assoc_files, custom_name_val.get(), float(custom_rv_val.get()), custom_sel_val.get())
        if button is not None:
            button["state"] = "normal"

    do_master_button = tk.Button(masspecframe, text="Create Master Spectrum", command=do_master_thing)
    do_master_button.grid(row=4, column=2)
    masspecframe.pack()


def showimages_model(gaia_id, frame, window):
    canvas = tk.Canvas(frame)
    scrollbar_y = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar_x = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    images = []

    for pdfpath in [f"models/{gaia_id}/spectroscopy_results.pdf", f"models/{gaia_id}/spectroscopy_spectrum_1.pdf"]:
        try:
            doc = fitz.open(pdfpath)
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=mat)
            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_tk = ImageTk.PhotoImage(im)
            image_label = tk.Label(scroll_frame, image=img_tk)
            image_label.image = img_tk
            image_label.pack()
            images.append(image_label)
        except:
            noresults = tk.Label(scroll_frame,
                                 font=('Segoe UI', 25),
                                 fg='#ff0000',
                                 text="No results to show yet!\nGenerate some!")
            noresults.pack(fill="both", expand=1)
            break

    scrollbar_y.pack(side=tk.RIGHT, fill="y")
    scrollbar_x.pack(side=tk.BOTTOM, fill="x")
    canvas.pack(side=tk.LEFT, fill="both", expand=True)

    def scrollHorizontally(event):
        if os.name == 'nt':
            canvas.xview_scroll(-1, "units")
        elif os.name == "posix":
            canvas.xview_scroll(-1, "units")

    def scrollVertically(event):
        if os.name == 'nt':
            canvas.yview_scroll(-1, "units")
        elif os.name == "posix":
            canvas.yview_scroll(-1, "units")

    def scrollHorizontallyInverse(event):
        if os.name == 'nt':
            canvas.xview_scroll(1, "units")
        elif os.name == "posix":
            canvas.xview_scroll(1, "units")

    def scrollVerticallyInverse(event):
        if os.name == 'nt':
            canvas.yview_scroll(1, "units")
        elif os.name == "posix":
            canvas.yview_scroll(1, "units")

    def bind_to_element(element):
        element.bind('<Up>', scrollVertically)
        element.bind('<Down>', scrollVerticallyInverse)
        element.bind('<Left>', scrollHorizontally)
        element.bind('<Right>', scrollHorizontallyInverse)
        element.bind('<Button-4>', scrollVertically)
        element.bind('<Button-5>', scrollVerticallyInverse)
        element.bind('<Shift-Button-4>', scrollHorizontally)
        element.bind('<Shift-Button-5>', scrollHorizontallyInverse)

    for i in images:
        bind_to_element(i)
    bind_to_element(canvas)
    setCanvasSize(frame, window)


def getgridoptions(isis_dir):
    if not isinstance(isis_dir, str):
        return ["ISIS directory not properly configured"]
    grid_fits_dirs = []
    start_depth = isis_dir.count(os.sep)

    for dir_path, dirs, files in os.walk(isis_dir):
        current_depth = dir_path.count(os.sep) - start_depth
        if current_depth > 3:
            del dirs[:]  # Clear the dirs list in-place to prevent further recursion down this branch.
            continue

        if "grid.fits" in files:
            grid_fits_dirs.append(dir_path)

    grid_fits_dirs = [g.replace(isis_dir, "") for g in grid_fits_dirs]

    if len(grid_fits_dirs) == 0:
        return ["ISIS directory not properly configured"]
    else:
        return grid_fits_dirs


def update_option_menu(opmenu, variable, options):
    menu = opmenu['menu']
    menu.delete(0, 'end')

    for string in options:
        menu.add_command(label=string,
                         command=lambda value=string: variable.set(value))


def fitsed(window, gaia_id):
    prefs = load_preferences()
    fitsedwindow = tk.Toplevel(window)
    fitsedwindow.title(f"Fit SED for Gaia DR3 {gaia_id}")
    fitsedwindow.geometry("800x600+0+0")

    isisexists = which("isis") is not None

    if not isisexists:
        isisdoesnotexist = tk.Label(fitsedwindow,
                                    font=('Segoe UI', 25),
                                    fg='#ff0000',
                                    text="No ISIS installation found!")
        isisdoesnotexist.pack()
        return

    if os.name == 'nt':
        fitsedwindow.state('zoomed')
    elif os.name == "posix":
        fitsedwindow.attributes('-zoomed', True)
    fitsedframe = tk.Frame(fitsedwindow)
    fitinputframe = tk.Frame(fitsedframe)
    fitoutputframe = tk.Frame(fitsedframe)

    convdict = {
        "Effective Temperature": "teff",
        "Surface Gravity": "logg",
        "Helium Abundance": "HE",
    }

    star_two_frame = ttk.LabelFrame(fitinputframe, text="Star Two")
    star_two_label_names = ["Effective Temperature", "Surface Gravity", "Helium Abundance"]
    c2vars = [(tk.StringVar(star_two_frame, value="c2_" + convdict[l]), tk.StringVar(star_two_frame), tk.IntVar(star_two_frame)) for l in star_two_label_names]
    star_two_inputs = [ttk.Entry(star_two_frame, textvariable=c2vars[i][1], state='disabled') for i, _ in enumerate(star_two_label_names)]
    star_two_freezeboxes = [tk.Checkbutton(star_two_frame, variable=c2vars[i][2], state='disabled') for i, _ in enumerate(star_two_label_names)]

    star_one_frame = ttk.LabelFrame(fitinputframe, text="Star One")
    star_one_label_names = ["Effective Temperature", "Surface Gravity", "Helium Abundance"]
    c1vars = [(tk.StringVar(star_one_frame, value="c1_" + convdict[l]), tk.StringVar(star_one_frame), tk.IntVar(star_one_frame)) for l in star_one_label_names]
    star_one_inputs = [ttk.Entry(star_one_frame, textvariable=c1vars[i][1]) for i, _ in enumerate(star_one_label_names)]
    star_one_freezeboxes = [tk.Checkbutton(star_one_frame, variable=c1vars[i][2]) for i, _ in enumerate(star_one_label_names)]

    gridoptions = getgridoptions(prefs["isisdir"])
    grid_one = tk.StringVar(star_one_frame, value="Select grid")
    gridlabel_one = tk.Label(star_one_frame, text="Select grid")
    grid_selector_one = tk.OptionMenu(star_one_frame, grid_one, *gridoptions)

    grid_two = tk.StringVar(star_two_frame, value="Select grid")
    grid_selector_two = tk.OptionMenu(star_two_frame, grid_two, *gridoptions)
    grid_selector_two.configure(state="disabled")
    gridlabel_two = tk.Label(star_two_frame, text="Select grid")

    def browse_sed_dir(isisdir):
        if isisdir is None:
            return
        fitsedwindow.wm_attributes('-topmost', 1)
        folder_selected = filedialog.askdirectory(parent=fitsedwindow)
        isisdir.set(folder_selected)
        prefs["isisdir"] = folder_selected
        save_preferences(prefs)
        gridoptions = getgridoptions(folder_selected)
        update_option_menu(grid_selector_one, grid_one, gridoptions)
        update_option_menu(grid_selector_two, grid_two, gridoptions)
        fitsedwindow.wm_attributes('-topmost', 0)

    miniframe = tk.Frame(fitinputframe)
    ttk.Label(miniframe, text="ISIS Model directory").pack(side=tk.LEFT, padx=10, pady=10)
    isis_dir = tk.StringVar(value=prefs["isisdir"])
    tk.Entry(miniframe, textvariable=isis_dir, state="disabled").pack(side=tk.LEFT, pady=10)
    tk.Button(miniframe, text="Browse Directories", command=lambda: browse_sed_dir(isis_dir)).pack(side=tk.LEFT, pady=10)
    miniframe.pack()

    for i, (label_name, entry, freezebox) in enumerate(zip(star_one_label_names, star_one_inputs, star_one_freezeboxes)):
        ttk.Label(star_one_frame, text=label_name).grid(row=i, column=0)
        entry.grid(row=i, column=1)
        ttk.Label(star_one_frame, text="Freeze?").grid(row=i, column=2)
        freezebox.grid(row=i, column=3)

    gridlabel_one.grid(row=i + 1, column=0)
    grid_selector_one.grid(row=i + 1, column=1)

    for i, (label_name, entry, freezebox) in enumerate(zip(star_two_label_names, star_two_inputs, star_two_freezeboxes)):
        ttk.Label(star_two_frame, text=label_name).grid(row=i, column=0)
        entry.grid(row=i, column=1)
        ttk.Label(star_two_frame, text="Freeze?").grid(row=i, column=2)
        freezebox.grid(row=i, column=3)

    gridlabel_two.grid(row=i + 1, column=0)
    grid_selector_two.grid(row=i + 1, column=1)

    def enable_star_two_inputs():
        if var.get():
            for input_box in star_two_inputs:
                input_box.config(state='normal')
            for freeze_box in star_two_freezeboxes:
                freeze_box.config(state='normal')
            grid_selector_two.config(state="normal")
        else:
            for input_box in star_two_inputs:
                input_box.config(state='disabled')
            for freeze_box in star_two_freezeboxes:
                freeze_box.config(state='disabled')
            grid_selector_two.config(state="disabled")

    var = tk.IntVar()
    checkbox = tk.Checkbutton(fitinputframe, text='Fit a composite SED?', variable=var, command=enable_star_two_inputs)

    star_one_frame.pack(anchor="nw", padx=10, pady=10, fill="x")
    star_two_frame.pack(anchor="nw", padx=10, pady=10, fill="x")
    checkbox.pack()

    noresults = None
    if os.path.isdir(f"SEDs/{gaia_id}"):
        showimages_sed(gaia_id, fitoutputframe, fitsedwindow)
    else:
        noresults = tk.Label(fitoutputframe,
                             font=('Segoe UI', 25),
                             fg='#ff0000',
                             text="No results to show yet!\nGenerate some!")
        noresults.pack(fill="both", expand=1)

    def calcsedwrapper(gaia_id, params, griddirs, isisdir):
        proc = calcsed(gaia_id, params, griddirs, isisdir)

        def execute_function_after_process():
            for child in fitoutputframe.winfo_children():
                child.destroy()
            progress_bar = ttk.Progressbar(fitoutputframe, mode='indeterminate', length=300)
            progress_bar.start()
            prog_label = tk.Label(fitoutputframe, text="Generating SED...")
            prog_label.pack()
            progress_bar.pack(pady=10)
            proc.communicate()  # Wait for the subprocess command to finish
            progress_bar.destroy()
            prog_label.destroy()
            showimages_sed(gaia_id, fitoutputframe, fitsedwindow)

        # Create a thread to execute the function after process completion
        thread = threading.Thread(target=execute_function_after_process)
        thread.start()

    dosedbtn = tk.Button(fitinputframe, text="Fit SED",
                         command=lambda: calcsedwrapper(gaia_id, [(a.get(), b.get(), c.get()) for a, b, c in c1vars + c2vars], [grid_one.get()[1:]] if grid_two.get() == "Select grid" else [grid_one.get()[1:], grid_two.get()[1:]], isis_dir.get() + "/"))
    dosedbtn.pack(anchor="se")

    fitinputframe.pack(side=tk.LEFT, anchor="nw", padx=10, pady=10)
    separator = ttk.Separator(fitsedframe, orient='vertical')
    separator.pack(side=tk.LEFT, fill="y")
    fitoutputframe.pack(side=tk.RIGHT, padx=10, pady=10, fill="both", expand=1)
    fitsedframe.pack(fill="both", expand=1)


def abs_mag(m_app, plx):
    d = 1 / (plx * 0.001)
    return m_app - 2.5 * np.log((d / 10))


def create_CMD_plot(gaia_id):
    objects = pd.read_csv("result_parameters.csv")
    target = objects[objects["source_id"] == gaia_id].iloc[0]
    bp_rp_target = target["bp_rp"]
    gmag_target = target["gmag"]
    plx_target = target["parallax"]
    plx_err_target = target["parallax_err"]

    objects = objects[0.2 * objects['parallax'] > objects['parallax_err']]
    bp_rps = objects["bp_rp"].to_numpy()
    plx = objects["parallax"].to_numpy()
    gmag = objects["gmag"].to_numpy()
    abs_mags = abs_mag(gmag, plx)
    plt.figure(figsize=(5, 5), dpi=200)
    plt.title(f"GAIA DR3 {gaia_id} on the CMD")
    plt.xlabel("BP-RP Color [mag]")
    plt.ylabel("Absolute G magnitude [mag]")
    plt.scatter(bp_rps, abs_mags, alpha=0.25, s=5, zorder=1, c="gray")
    plt.gca().invert_yaxis()
    if plx_target < 0:
        absmag_target = np.nan
        absmag_err_target = np.nan
        plt.title("Bad Parallax!")
    if plx_err_target < 0.2 * plx_target:
        absmag_target = abs_mag(gmag_target, plx_target)
        plt.scatter(bp_rp_target, absmag_target, zorder=5, c="darkred")
    else:
        absmag_target = abs_mag(gmag_target, plx_target + plx_err_target)
        plt.scatter(bp_rp_target, absmag_target, zorder=5, c="darkred")
        plt.errorbar(bp_rp_target, absmag_target, yerr=3, zorder=5, uplims=True, lolims=False, c="darkred")
    plt.tight_layout()
    plt.savefig(f"{general_config['OUTPUT_DIR']}/{gaia_id}/CMD.png")
    plt.close()


def show_CMD_window(gaia_id):
    cmd_window = tk.Toplevel()
    cmd_window.title(f"CMD for Gaia DR3 {gaia_id}")
    cmd_window.geometry("500x500+0+0")
    cmd_frame = tk.Frame(cmd_window)
    fpath = f"{general_config['OUTPUT_DIR']}/{gaia_id}/CMD.png"
    if not os.path.isfile(fpath):
        create_CMD_plot(gaia_id)
    im = Image.open(fpath)
    img_tk = ImageTk.PhotoImage(im.resize((500, 500)))
    label = tk.Label(cmd_frame, image=img_tk)
    label.image = img_tk  # Save reference to image
    label.pack()
    cmd_frame.pack(expand=1, fill="both")


def calcmodel(gaia_id, parameter_list, griddirs, isisdir, fileloc, resolution=3.2):
    parameter_list = [("c*_xi", "0", 1), ("c*_z", "0", 1)] + parameter_list
    n_list = []
    for a, b, c in parameter_list:
        if b != "":
            n_list.append((a, b, c))
    parameter_list = n_list
    firstline = 'require("stellar_isisscripts.sl");'
    double_quoted_list = ', '.join([f'"{item}"' for item in griddirs])
    modelline = f'variable modelgrid; modelgrid = [{double_quoted_list}];'
    paramstring = f'''variable initial_guess_params_values =  struct{{name = ["{'","'.join([p[0] for p in parameter_list])}"],
    value = [{",".join([p[1] for p in parameter_list])}],
    freeze = [{",".join([str(p[2]) for p in parameter_list])}]}};'''
    master_spec_line = f"""variable input =
  [
   struct{{
     filename = "{fileloc}",
     spectype = "ASCII_with_2_columns",
     ignore = [{{3932,3935}},{{3967,3970}},{{4610,4655}},{{5888,5892}},{{5894,5898}}],
     cspline_anchorpoints = [[3000:3850:50],[3850:4050:100],[4050:4550:100],[4550:15050:200]],
     res_offset = 0., % R = res_offset + res_slope*lambda
     res_slope = 1/{resolution},
   }}
  ];"""
    isispath = f'variable bpaths = ["{str(isisdir)}"];'

    if not os.path.isdir(f"models/{gaia_id}"):
        os.mkdir(f"models/{gaia_id}")
    else:
        os.remove(f"models/{gaia_id}/spectroscopy.sl")
    with open(f"models/{gaia_id}/spectroscopy.sl", "w") as script:
        script.write(firstline + "\n")
        script.write(modelline + "\n")
        script.write(paramstring + "\n")
        script.write(master_spec_line + "\n")
        script.write(isispath + "\n")
        with open(f"models/modelscriptbase.sl", "r") as restofthefnowl:
            script.write(restofthefnowl.read())

    p = subprocess.Popen(f"isis spectroscopy.sl", shell=True, cwd=f"models/{gaia_id}")

    return p


def fitmodel(window, gaia_id):
    prefs = load_preferences()
    fitmodelwindow = tk.Toplevel(window)
    fitmodelwindow.title(f"Fit Model for Gaia DR3 {gaia_id}")
    fitmodelwindow.geometry("800x600+0+0")

    isisexists = which("isis") is not None

    if not isisexists:
        isisdoesnotexist = tk.Label(fitmodelwindow,
                                    font=('Segoe UI', 25),
                                    fg='#ff0000',
                                    text="No ISIS installation found!")
        isisdoesnotexist.pack()
        return

    if os.name == 'nt':
        fitmodelwindow.state('zoomed')
    elif os.name == "posix":
        fitmodelwindow.attributes('-zoomed', True)
    fitmodelframe = tk.Frame(fitmodelwindow)
    fitinputframe = tk.Frame(fitmodelframe)
    fitoutputframe = tk.Frame(fitmodelframe)

    convdict = {
        "Rotational Velocity": "vsini",
        "Radial Velocity": "vrad",
        "Effective Temperature": "teff",
        "Surface Gravity": "logg",
        "Helium Abundance": "HE",
    }

    star_one_frame = ttk.LabelFrame(fitinputframe, text="Stellar Parameters")
    star_one_label_names = ["Radial Velocity", "Rotational Velocity", "Effective Temperature", "Surface Gravity", "Helium Abundance"]
    c1vars = [(tk.StringVar(star_one_frame, value="c1_" + convdict[l]), tk.StringVar(star_one_frame), tk.IntVar(star_one_frame)) for l in star_one_label_names]
    star_one_inputs = [ttk.Entry(star_one_frame, textvariable=c1vars[i][1]) for i, _ in enumerate(star_one_label_names)]
    star_one_freezeboxes = [tk.Checkbutton(star_one_frame, variable=c1vars[i][2]) for i, _ in enumerate(star_one_label_names)]

    gridoptions = getgridoptions(prefs["isisdir"])
    grid_one = tk.StringVar(star_one_frame, value="Select grid")
    gridlabel_one = tk.Label(star_one_frame, text="Select grid")
    grid_selector_one = tk.OptionMenu(star_one_frame, grid_one, *gridoptions)

    for i, (label_name, entry, freezebox) in enumerate(zip(star_one_label_names, star_one_inputs, star_one_freezeboxes)):
        ttk.Label(star_one_frame, text=label_name).grid(row=i, column=0)
        entry.grid(row=i, column=1)
        ttk.Label(star_one_frame, text="Freeze?").grid(row=i, column=2)
        freezebox.grid(row=i, column=3)

    gridlabel_one.grid(row=i + 1, column=0)
    grid_selector_one.grid(row=i + 1, column=1)

    star_one_frame.pack(anchor="nw", padx=10, pady=10, fill="x")

    metaoptions = ttk.LabelFrame(fitinputframe, text="Advanced Options")

    def browse_model_dir(isisdir):
        if isisdir is None:
            return
        fitmodelwindow.wm_attributes('-topmost', 1)
        folder_selected = filedialog.askdirectory(parent=fitmodelwindow)
        isisdir.set(folder_selected)
        prefs["isisdir"] = folder_selected
        save_preferences(prefs)
        gridoptions = getgridoptions(folder_selected)
        update_option_menu(grid_selector_one, grid_one, gridoptions)
        fitmodelwindow.wm_attributes('-topmost', 0)

    miniframe = tk.Frame(metaoptions)
    ttk.Label(miniframe, text="ISIS Model directory").pack(side=tk.LEFT, padx=5)
    isis_dir = tk.StringVar(value=prefs["isisdir"])
    tk.Entry(miniframe, textvariable=isis_dir, state="disabled").pack(side=tk.LEFT, padx=5)
    tk.Button(miniframe, text="Browse Directories", command=lambda: browse_model_dir(isis_dir)).pack(side=tk.LEFT, padx=5)
    miniframe.grid(row=0, column=0, columnspan=3)

    resvar = tk.StringVar(value="3.2")
    reslabel = tk.Label(metaoptions, text="Wavelength resolution Δλ")
    resinput = ttk.Entry(metaoptions, textvariable=resvar, width=5)
    reslabel.grid(row=1, column=0, sticky="w")
    resinput.grid(row=1, column=1, sticky="w")

    metaoptions.pack(anchor="nw", padx=10, pady=10, fill="x")

    if os.path.isdir(f"models/{gaia_id}"):
        showimages_model(gaia_id, fitoutputframe, fitmodelwindow)
    else:
        noresults = tk.Label(fitoutputframe,
                             font=('Segoe UI', 25),
                             fg='#ff0000',
                             text="No results to show yet!\nGenerate some!")
        noresults.pack(fill="both", expand=1)

    def calcmodelwrapper(gaia_id, params, griddir, isisdir, fileloc, resolution):
        proc = calcmodel(gaia_id, params, griddir, isisdir, fileloc, resolution)

        def execute_function_after_process():
            for child in fitoutputframe.winfo_children():
                child.destroy()
            progress_bar = ttk.Progressbar(fitoutputframe, mode='indeterminate', length=300)
            progress_bar.start()
            prog_label = tk.Label(fitoutputframe, text="Fitting Model...")
            prog_label.pack()
            progress_bar.pack(pady=10)
            proc.communicate()  # Wait for the subprocess command to finish
            progress_bar.destroy()
            prog_label.destroy()
            showimages_model(gaia_id, fitoutputframe, fitmodelwindow)

        # Create a thread to execute the function after process completion
        thread = threading.Thread(target=execute_function_after_process)
        thread.start()

    # TODO: fix up these variable names

    buttonsframe = tk.Frame(fitinputframe)
    domodelbtn = tk.Button(buttonsframe, text="Fit Model",
                           command=lambda: calcmodelwrapper(gaia_id, [(a.get(), b.get(), c.get()) for a, b, c in c1vars], [grid_one.get()[1:]], isis_dir.get() + "/", f"/home/fabian/PycharmProjects/RVVD_plus_LAMOST/master_spectra/{gaia_id}_stacked.txt",
                                                            resvar.get()))

    create_master_spec_btn = tk.Button(buttonsframe, text="Create master spectrum", command=lambda: master_spectrum_window(window, queue, gaia_id, button=domodelbtn))

    if not os.path.isfile(f"master_spectra/{gaia_id}_stacked.txt"):
        domodelbtn["state"] = "disabled"

    domodelbtn.pack(side=tk.RIGHT)
    create_master_spec_btn.pack(side=tk.RIGHT)
    buttonsframe.pack(anchor="se")

    fitinputframe.pack(side=tk.LEFT, anchor="nw", padx=10, pady=10)
    separator = ttk.Separator(fitmodelframe, orient='vertical')
    separator.pack(side=tk.LEFT, fill="y")
    fitoutputframe.pack(side=tk.RIGHT, padx=10, pady=10, fill="both", expand=1)
    fitmodelframe.pack(fill="both", expand=1)


def plotgzflightcurve(resultqueue, gaia_id, pgramdata, lctype, bandnames, bandcolors, times, fluxes, flux_errors, nterms=1, nsamp=None, min_p=None, max_p=None, noiseceil=None):
    # the figure that will contain the plot
    fig, axs = plt.subplots(2, 1, figsize=(4.8 * 16 / 9, 4.8), dpi=90 * 9 / 4.8)

    # adding the subplot
    plot1 = axs[0]
    periodogram = axs[1]

    normflxs = []
    normflxerrs = []
    for ind, [f, fe] in enumerate(zip(fluxes, flux_errors)):
        nfe = fe / np.median(f)
        nf = f / np.median(f)
        if noiseceil is not None:
            mask = nf < noiseceil
            times[ind] = times[ind][mask]
            fluxes[ind] = fluxes[ind][mask]
            flux_errors[ind] = flux_errors[ind][mask]
            nfe = nfe[mask]
            nf = nf[mask]

        normflxs.append(nf)
        normflxerrs.append(nfe)

    mashedtimes = np.concatenate(times)

    if pgramdata is None:
        model = periodic.LombScargleMultibandFast(Nterms=nterms)
        model.fit(mashedtimes,
                  np.concatenate(fluxes),
                  np.concatenate(flux_errors),
                  np.concatenate([np.full((len(fluxes[0]), 1), bandnames[0]),
                                  np.full((len(fluxes[1]), 1), bandnames[1]),
                                  np.full((len(fluxes[2]), 1), bandnames[2])]))
        tdiffs = np.diff(mashedtimes)

        if min_p is None:
            min_p = np.max([np.min(tdiffs[tdiffs > 0]), 0.01])
        elif max_p is None:
            max_p = np.ptp(mashedtimes) / 2

        if nsamp is None:
            nsamp = calcpgramsamples(np.ptp(mashedtimes), min_p, max_p)

        pgram = model.score_frequency_grid(1 / max_p, (1 / min_p - 1 / max_p) / nsamp, nsamp)

        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        ps = 1 / freqs
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        np.savetxt(f"lightcurves/{gaia_id}/{lctype.lower()}_lc_periodogram.txt", np.vstack([ps, pgram]).T, delimiter=",")
    else:
        ps = pgramdata[:, 0]
        pgram = pgramdata[:, 1]

    periodogram.set_xlabel("Period [d]")
    periodogram.set_ylabel("Relative Power [arb. Unit]")
    periodogram.set_xscale("log")
    periodogram.plot(ps, pgram, color="navy", zorder=5)

    maxp = ps[ps < 2][np.argmax(pgram[ps < 2])]
    periodogram.axvline(maxp, color="darkred", linestyle="--", linewidth=1, zorder=4)
    periodogram.annotate(f'Period peak: {round(maxp, 7)}d', xy=(1, 1), xytext=(-15, -15),
                         xycoords='axes fraction', textcoords='offset points',
                         ha='right', va='top', color='red')

    for f, fe, t, c, l in zip(normflxs, normflxerrs, times, bandcolors, [f"{bandnames[0]} band Flux", f"{bandnames[1]} band Flux", f"{bandnames[2]} band Flux"]):
        # plotting the graph
        plot1.scatter(t, f, color=c, zorder=5, label=l)
        plot1.errorbar(t, f, yerr=fe, linestyle='', color=c, capsize=3, zorder=4, label="_nolabel_")

    plot1.legend(loc='upper right')
    plot1.grid(color="lightgrey", linestyle='--')
    plot1.set_xlabel("Time [d]")
    plot1.set_ylabel(f"Relative {lctype} G Flux\n [arb. Unit]")
    plt.tight_layout()

    resultqueue.put([fig, axs])


def plottesslightcurve(resultqueue, gaia_id, pgramdata, times, flux, flux_error, nterms=1, nsamp=None, min_p=None, max_p=None, noiseceil=None):
    # the figure that will contain the plot
    fig, axs = plt.subplots(2, 1, figsize=(4.8 * 16 / 9, 4.8), dpi=90 * 9 / 4.8)

    # adding the subplot
    plot1 = axs[0]
    periodogram = axs[1]

    nfe = flux_error / np.median(flux)
    nf = flux / np.median(flux)
    if noiseceil is not None:
        mask = nf < noiseceil
        times = times[mask]
        flux = flux[mask]
        flux_error = flux_error[mask]
        nfe = nfe[mask]
        nf = nf[mask]

    if pgramdata is None:
        model = periodic.LombScargleFast(Nterms=nterms)
        model.fit(times,
                  flux,
                  flux_error)

        tdiffs = np.diff(times)

        if min_p is None:
            min_p = np.max([np.min(tdiffs[tdiffs > 0]), 0.01])
        elif max_p is None:
            max_p = np.ptp(times) / 2

        if nsamp is None:
            nsamp = calcpgramsamples(np.ptp(times), min_p, max_p)

        pgram = model.score_frequency_grid(1 / max_p, (1 / min_p - 1 / max_p) / nsamp, nsamp)

        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        ps = 1 / freqs
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        np.savetxt(f"lightcurves/{gaia_id}/tess_lc_periodogram.txt", np.vstack([ps, pgram]).T, delimiter=",")
    else:
        ps = pgramdata[:, 0]
        pgram = pgramdata[:, 1]

    periodogram.set_xlabel("Period [d]")
    periodogram.set_ylabel("Relative Power [arb. Unit]")
    periodogram.set_xscale("log")
    periodogram.plot(ps, pgram, color="navy", zorder=5)

    if np.any(ps < 2):
        maxp = ps[ps < 2][np.argmax(pgram[ps < 2])]
    else:
        maxp = ps[np.argmax(pgram)]
    periodogram.axvline(maxp, color="darkred", linestyle="--", linewidth=1, zorder=4)
    periodogram.annotate(f'Period peak: {round(maxp, 7)}d', xy=(1, 1), xytext=(-15, -15),
                         xycoords='axes fraction', textcoords='offset points',
                         ha='right', va='top', color='red')

    plot1.scatter(times, nf, color="darkred", zorder=5, s=5, label="TESS Flux")
    plot1.errorbar(times, nf, yerr=nfe, linestyle='', color="darkred", capsize=3, zorder=4, label="_nolabel_")
    plot1.legend(loc='upper right')
    plot1.grid(color="lightgrey", linestyle='--')
    plot1.set_xlabel("Time [d]")
    plot1.set_ylabel("Relative Gaia G Flux\n [arb. Unit]")
    plt.tight_layout()

    resultqueue.put([fig, axs])


def plotlc_async(queue, gaia_id, frame, pgramdata, lctype, *args, **kwargs):
    progress_bar = ttk.Progressbar(frame, mode='indeterminate', length=300)
    progress_bar.start()
    progress_bar.pack(pady=10)

    # If gaia lc
    if lctype == "GAIA":
        def plotlc():
            plt.close('all')
            resultqueue = man.Queue()
            queue.put(["execute_function", plotgzflightcurve, (resultqueue, gaia_id, pgramdata, lctype, ["G", "BP", "RP"], ["green", "navy", "darkred"], *args), kwargs])
            fig, axs = resultqueue.get()
            progress_bar.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas.get_tk_widget().pack()
            plt.close(fig)

    elif lctype == "ZTF":
        def plotlc():
            plt.close('all')
            resultqueue = man.Queue()
            queue.put(["execute_function", plotgzflightcurve, (resultqueue, gaia_id, pgramdata, lctype, ["G", "I", "R"], ["green", "darkred", "red"], *args), kwargs])
            fig, axs = resultqueue.get()
            progress_bar.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas.get_tk_widget().pack()
            plt.close(fig)

    elif lctype == "TESS":
        def plotlc():
            plt.close('all')
            resultqueue = man.Queue()
            queue.put(["execute_function", plottesslightcurve, (resultqueue, gaia_id, pgramdata, *args), kwargs])
            fig, axs = resultqueue.get()
            progress_bar.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas.get_tk_widget().pack()
            plt.close(fig)

    thread = threading.Thread(target=plotlc)
    thread.start()
    return thread


def getgaialc(gaia_id):
    photquery = Vizier(
        columns=["Source", "TimeG", "TimeBP", "TimeRP", "FG", "FBP", "FRP", "e_FG", "e_FBP", "e_FRP", "noisyFlag"]
    ).query_region(
        SkyCoord.from_name(f'GAIA DR3 {gaia_id}'),
        radius=1 * u.arcsec,
        catalog='I/355/epphot'  # I/355/epphot is the designation for the Gaia photometric catalogue on Vizier
    )

    if len(photquery) != 0:
        table = photquery[0].to_pandas()
        table = table[table["noisyFlag"] != 1]
        table = table.drop(columns=["noisyFlag"])
        table.columns = ["Source", "TimeG", "TimeBP", "TimeRP", "FG", "FBP", "FRP", "e_FG", "e_FBP", "e_FRP"]
        table = table[table["Source"] == gaia_id]
        table = table.drop(columns=["Source"])
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        table.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
        return True
    else:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/gaia_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
            return False


def drawlcplot(lcdata, gaia_id, plotframe, plotctl, plotwrapper, btntip, queue, lctype, pgramdata):
    if lctype == "GAIA":
        lcdata = lcdata[~np.isnan(lcdata).any(axis=-1), :]
        tg = lcdata[:, 0]
        g_flx = lcdata[:, 3]
        g_flx_err = lcdata[:, 6]
        tbp = lcdata[:, 1]
        bp_flx = lcdata[:, 4]
        bp_flx_err = lcdata[:, 7]
        trp = lcdata[:, 1]
        rp_flx = lcdata[:, 5]
        rp_flx_err = lcdata[:, 8]
        btntip.destroy()
        plotlc_async(queue, gaia_id, plotframe, pgramdata, lctype, [tg, tbp, trp], [g_flx, bp_flx, rp_flx], [g_flx_err, bp_flx_err, rp_flx_err])
    elif lctype == "TESS":
        print(lcdata)
        t = lcdata[:, 0]
        tess_flx = lcdata[:, 1]
        tess_flx_err = lcdata[:, 2]
        #        crowdsap = lcdata[3]
        btntip.destroy()
        plotlc_async(queue, gaia_id, plotframe, pgramdata, lctype, t, tess_flx, tess_flx_err)
    elif lctype == "ZTF":
        gmask = lcdata[:, -1] == 1.
        imask = lcdata[:, -1] == 2.
        rmask = lcdata[:, -1] == 3.

        times = []
        flxs = []
        flx_errs = []
        for mask in [gmask, imask, rmask]:
            times.append(lcdata[:, 0][mask])
            flxs.append(lcdata[:, 1][mask])
            flx_errs.append(lcdata[:, 2][mask])

        plotlc_async(queue, gaia_id, plotframe, pgramdata, lctype, times, flxs, flx_errs)

    filterframe = tk.Frame(plotctl)
    checknoisevar = tk.IntVar(value=0)
    noiselvlvar = tk.StringVar(value="0")

    filterlabel = tk.Label(filterframe, text="Ignore Noise above ", padx=5)
    filtercheckbtn = tk.Checkbutton(filterframe, variable=checknoisevar)
    filterentry = tk.Entry(filterframe, textvariable=noiselvlvar, width=4)
    filterlabel.grid(row=0, column=0)
    filterentry.grid(row=0, column=1)
    filtercheckbtn.grid(row=0, column=2)

    redofitframe = tk.Frame(plotctl)
    checkredovar = tk.IntVar(value=0)
    loperiodvar = tk.StringVar(value="0")
    hiperiodvar = tk.StringVar(value="0")

    periodlabel = tk.Label(redofitframe, text="Redraw periodogram from ", padx=5)
    periodinterlabel = tk.Label(redofitframe, text=" to ", padx=5)
    periodcheckbtn = tk.Checkbutton(redofitframe, variable=checkredovar)
    periodloentry = tk.Entry(redofitframe, textvariable=loperiodvar, width=4)
    periodhientry = tk.Entry(redofitframe, textvariable=hiperiodvar, width=4)
    periodlabel.grid(row=0, column=0)
    periodloentry.grid(row=0, column=1)
    periodinterlabel.grid(row=0, column=2)
    periodhientry.grid(row=0, column=3)
    periodcheckbtn.grid(row=0, column=4)

    nsamplesframe = tk.Frame(plotctl)
    checknsamplesvar = tk.IntVar(value=0)
    nsamplesvar = tk.StringVar(value="0")
    nsampleslabel1 = tk.Label(nsamplesframe, text="Use ", padx=5)
    nsampleslabel2 = tk.Label(nsamplesframe, text=" samples", padx=5)
    nsamplescheckbtn = tk.Checkbutton(nsamplesframe, variable=checknsamplesvar)
    nsampleshientry = tk.Entry(nsamplesframe, textvariable=nsamplesvar, width=8)
    nsampleslabel1.grid(row=0, column=0)
    nsampleshientry.grid(row=0, column=1)
    nsampleslabel2.grid(row=0, column=2)
    nsamplescheckbtn.grid(row=0, column=3)

    ntermsframe = tk.Frame(plotctl)
    checkntermsvar = tk.IntVar(value=0)
    ntermsvar = tk.StringVar(value="0")
    ntermslabel1 = tk.Label(ntermsframe, text="Use ", padx=5)
    ntermslabel2 = tk.Label(ntermsframe, text=" terms", padx=5)
    ntermscheckbtn = tk.Checkbutton(ntermsframe, variable=checkntermsvar)
    ntermsentry = tk.Entry(ntermsframe, textvariable=ntermsvar, width=4)
    ntermslabel1.grid(row=0, column=0)
    ntermsentry.grid(row=0, column=1)
    ntermslabel2.grid(row=0, column=2)
    ntermscheckbtn.grid(row=0, column=3)

    def drawplotwrapper():
        for widget in plotframe.winfo_children():
            widget.destroy()
        nonlocal lcdata
        if checkntermsvar.get() == 1:
            nterms = ntermsvar.get()
            try:
                nterms = int(nterms)
            except ValueError:
                nterms = 1
        else:
            nterms = 1
        if checknsamplesvar.get() == 1:
            nsamp = nsamplesvar.get()
            try:
                nsamp = int(nsamp)
            except ValueError:
                nsamp = 50000
        else:
            nsamp = 50000
        if checkredovar.get() == 1:
            min_p = loperiodvar.get()
            max_p = hiperiodvar.get()
            try:
                min_p = float(min_p)
                max_p = float(max_p)
            except ValueError:
                min_p = None
                max_p = None
        else:
            min_p = None
            max_p = None
        if checknoisevar.get() == 1:
            noiseceil = noiselvlvar.get()
            try:
                noiseceil = float(noiseceil)
            except ValueError:
                noiseceil = None
        else:
            noiseceil = None
        if lctype == "GAIA":
            lcdata = lcdata[~np.isnan(lcdata).any(axis=-1), :]
            tg = lcdata[:, 0]
            g_flx = lcdata[:, 3]
            g_flx_err = lcdata[:, 6]
            tbp = lcdata[:, 1]
            bp_flx = lcdata[:, 4]
            bp_flx_err = lcdata[:, 7]
            trp = lcdata[:, 1]
            rp_flx = lcdata[:, 5]
            rp_flx_err = lcdata[:, 8]
            plotlc_async(queue, gaia_id, plotframe, None, lctype, [tg, tbp, trp], [g_flx, bp_flx, rp_flx], [g_flx_err, bp_flx_err, rp_flx_err], nterms=nterms, nsamp=nsamp, min_p=min_p, max_p=max_p, noiseceil=noiseceil)
        elif lctype == "TESS":
            t = lcdata[:, 0]
            tess_flx = lcdata[:, 1]
            tess_flx_err = lcdata[:, 2]
            #            crowdsap = lcdata[3]
            btntip.destroy()
            plotlc_async(queue, gaia_id, plotframe, None, lctype, t, tess_flx, tess_flx_err, nterms=nterms, nsamp=nsamp, min_p=min_p, max_p=max_p, noiseceil=noiseceil)
        elif lctype == "ZTF":
            gmask = lcdata[:, -1] == 1.
            imask = lcdata[:, -1] == 2.
            rmask = lcdata[:, -1] == 3.

            times = []
            flxs = []
            flx_errs = []
            for mask in [gmask, imask, rmask]:
                times.append(lcdata[:, 0][mask])
                flxs.append(lcdata[:, 1][mask])
                flx_errs.append(lcdata[:, 2][mask])
            plotlc_async(queue, gaia_id, plotframe, None, lctype, times, flxs, flx_errs, nterms=nterms, nsamp=nsamp, min_p=min_p, max_p=max_p, noiseceil=noiseceil)

    redoperiodogrambtn = tk.Button(plotctl, text="Redo Periodogram", command=drawplotwrapper)

    filterframe.grid(row=0, column=0, sticky="w")
    redofitframe.grid(row=1, column=0, sticky="w")
    nsamplesframe.grid(row=2, column=0, sticky="w")
    ntermsframe.grid(row=3, column=0, sticky="w")
    redoperiodogrambtn.grid(row=4, column=0, sticky="se")
    plotframe.grid(row=0, column=0, pady=10, padx=10)
    plotctl.grid(row=0, column=1, sticky="n", pady=10, padx=10)
    plotwrapper.pack()


def getgaiawrapper(gaia_id, tiplabel, lcframe, ctlframe, plotwrapper, queue):
    progress_bar = ttk.Progressbar(plotwrapper, mode='indeterminate', length=300)
    progress_bar.pack()
    progress_bar.start()

    def startgaiathread():
        thread = threading.Thread(target=lambda: getgaialc(gaia_id))
        thread.start()
        thread.join()

        lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/gaia_lc.txt", delimiter=",")

        if lcdata.ndim == 1:
            tiplabel.destroy()
            notfoundlabel = tk.Label(lcframe, text="No GAIA Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
        else:
            progress_bar.destroy()
            drawlcplot(lcdata, gaia_id, lcframe, ctlframe, plotwrapper, tiplabel, queue, "GAIA", None)

    outerthread = threading.Thread(target=startgaiathread)
    outerthread.start()


def opentessfile(flist):
    crowdsap = []
    try:
        with fits.open(flist[0]) as TESSdata:
            data = TESSdata[1].data
            BJD = np.array(data['TIME'])
            flux = np.array(data['PDCSAP_FLUX'])
            err_flux = np.array(data['PDCSAP_FLUX_ERR'])
            err_flux = err_flux / np.nanmean(flux)
            flux = flux / np.nanmean(flux)
            header = TESSdata[1].header
            crowdsap.append(header['CROWDSAP'])

            if len(flist) > 1:
                for i in range(1, len(flist)):
                    with fits.open(flist[i]) as TESSdata:
                        data = TESSdata[1].data
                        BJD = np.append(BJD, np.array(data['TIME']))
                        f = np.array(data['PDCSAP_FLUX'])
                        ef = np.array(data['PDCSAP_FLUX_ERR'])
                        flux = np.append(flux, f / np.nanmean(f))
                        err_flux = np.append(err_flux, ef / np.nanmean(f))
                        header = TESSdata[1].header
                        crowdsap.append(header['CROWDSAP'])

        err_flux = err_flux / np.nanmean(flux)
        flux = flux / np.nanmean(flux)

        bjd = np.array(BJD)
        flux = np.array(flux)
        flux_err = np.array(err_flux)
        crowdsap = np.array(crowdsap)
        return bjd, flux, flux_err, crowdsap
    except OSError:
        return np.array([]), np.array([]), np.array([]), np.nan


def gettesslc(tic, gaia_id):
    print(f"[{gaia_id}] Getting MAST data...")
    obsTable = Observations.query_criteria(dataproduct_type="timeseries",
                                           project="TESS",
                                           target_name=tic)

    try:
        data = Observations.get_product_list(obsTable)
    except InvalidQueryError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")

    times = []
    fluxes = []
    flux_errors = []
    crowdsaps = []

    print(f"[{gaia_id}] Looking for long cadence data...")
    long_c_lc = Observations.download_products(data, productSubGroupDescription="LC")

    if long_c_lc is not None:
        long_c_lc = long_c_lc[0][:]
        t1, f1, ef1, cs1 = opentessfile(long_c_lc)
        times.append(t1)
        fluxes.append(f1)
        flux_errors.append(ef1)
        crowdsaps.append(cs1)

    print(f"[{gaia_id}] Looking for short cadence data...")
    short_c_lc = Observations.download_products(data, productSubGroupDescription="FAST-LC")

    if short_c_lc is not None:
        short_c_lc = short_c_lc[0][:]
        t2, f2, ef2, cs2 = opentessfile(short_c_lc)
        times.append(t2)
        fluxes.append(f2)
        flux_errors.append(ef2)
        crowdsaps.append(cs2)

    times = np.concatenate(times)
    flux = np.concatenate(fluxes)
    flux_error = np.concatenate(flux_errors)
    print(f"[{gaia_id}] Got {len(times)} datapoints")

    if len(times) > 0:
        mask = np.logical_and(np.logical_and(~np.isnan(times), ~np.isnan(flux)), ~np.isnan(flux_error))
        times = times[mask]
        flux = flux[mask]
        flux_error = flux_error[mask]

        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        flux = flux[sorted_indices]
        flux_error = flux_error[sorted_indices]
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        np.savetxt(f"lightcurves/{gaia_id}/tess_lc.txt", np.vstack((times, flux, flux_error)).T, delimiter=",")
    else:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")


def gettesswrapper(gaia_id, tic, btntip, plotframe, plotctl, plotwrapper, lcframe, queue):
    progress_bar = ttk.Progressbar(lcframe, mode='indeterminate', length=300)
    progress_bar.pack()
    progress_bar.start()
    print(f"[{gaia_id}] Getting TESS LC...")

    def starttessthread():
        thread = threading.Thread(target=lambda: gettesslc(tic, gaia_id))
        thread.start()
        thread.join()
        print(f"[{gaia_id}] Beginning Plotting...")
        try:
            lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/tess_lc.txt", delimiter=",")
        except FileNotFoundError:
            btntip.destroy()
            notfoundlabel = tk.Label(lcframe, text="No TESS Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
            return

        progress_bar.destroy()

        if lcdata.ndim != 1:
            drawlcplot(lcdata, gaia_id, plotframe, plotctl, plotwrapper, btntip, queue, "TESS", None)
        else:
            btntip.destroy()
            notfoundlabel = tk.Label(lcframe, text="No TESS Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()

    outerthread = threading.Thread(target=starttessthread)
    outerthread.start()


def magtoflux(mag):
    return 1 / (2.5 ** mag)


def magerr_to_fluxerr(mag, magerr):
    return (np.abs(1 / (2.5 ** mag) - 1 / (2.5 ** (mag - magerr))) + np.abs(1 / (2.5 ** mag) - 1 / (2.5 ** (mag + magerr)))) / 2


def getztflc(gaia_id):
    coord = SkyCoord.from_name(f'GAIA DR3 {gaia_id}')

    # try:
    lcq = lightcurve.LCQuery().from_position((coord.ra * u.deg).value, (coord.dec * u.deg).value, 1)
    mask = lcq.data["catflags"] == 0

    data = lcq.data[mask]
    dates = data["mjd"].to_numpy()

    if len(dates) == 0:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/ztf_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False

    mags = data["mag"].to_numpy()
    mag_err = data["magerr"].to_numpy()
    filters = data["filtercode"].to_numpy()

    flx = magtoflux(mags)
    flx_err = magerr_to_fluxerr(mags, mag_err)

    table = pd.DataFrame({"mjd": dates, "flx": flx, "flx_err": flx_err, "filter": filters})
    if not os.path.isdir(f"lightcurves/{gaia_id}"):
        os.mkdir(f"lightcurves/{gaia_id}")
    table.to_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", index=False, header=False)

    return True


def getztfwrapper(gaia_id, tiplabel, lcframe, ctlframe, plotwrapper, queue):
    progress_bar = ttk.Progressbar(plotwrapper, mode='indeterminate', length=300)
    progress_bar.pack()
    progress_bar.start()

    def startztfthread():
        thread = threading.Thread(target=lambda: getztflc(gaia_id))
        thread.start()
        thread.join()

        try:
            lcdata = pd.read_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", delimiter=",")
            numdata = lcdata[lcdata.columns[:3]].to_numpy()
            filter = lcdata[lcdata.columns[-1]].to_numpy()
            filter[filter == "zg"] = 1
            filter[filter == "zi"] = 2
            filter[filter == "zr"] = 3
            lcdata = np.column_stack([numdata, filter]).astype(float)
        except pd.errors.EmptyDataError:
            tiplabel.destroy()
            notfoundlabel = tk.Label(lcframe, text="No ZTF Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
            lcdata = np.zeros(5)

        if lcdata.ndim == 1:
            tiplabel.destroy()
            notfoundlabel = tk.Label(lcframe, text="No ZTF Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
        else:
            tiplabel.destroy()
            progress_bar.destroy()
            drawlcplot(lcdata, gaia_id, lcframe, ctlframe, plotwrapper, tiplabel, queue, "ZTF", None)

    outerthread = threading.Thread(target=startztfthread)
    outerthread.start()


def lcplottab(lcdata, lcframe, name, gaia_id, tic, queue, pgramdata):
    btntip = tk.Label(lcframe, text=f"Use the button to look for a {name} lightcurve!")
    btntip.pack()

    plotwrapper = tk.Frame(lcframe)
    plotframe = tk.Frame(plotwrapper)
    plotctl = tk.Frame(plotwrapper)

    if lcdata is not None:
        if name == "GAIA":
            if lcdata.ndim == 1:
                btntip.destroy()
                notfoundlabel = tk.Label(lcframe, text=f"No {name} Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
                notfoundlabel.pack()
            else:
                drawlcplot(lcdata, gaia_id, plotframe, plotctl, plotwrapper, btntip, queue, name, pgramdata)
        elif name == "ZTF":
            drawlcplot(lcdata, gaia_id, plotframe, plotctl, plotwrapper, btntip, queue, name, pgramdata)
        else:
            drawlcplot(lcdata, gaia_id, plotframe, plotctl, plotwrapper, btntip, queue, name, pgramdata)

    else:
        if name == "GAIA":
            getgaiabtn = tk.Button(lcframe, text="Get Gaia Data", command=lambda: getgaiawrapper(gaia_id, btntip, plotframe, plotctl, plotwrapper, queue))
            getgaiabtn.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)
        elif name == "TESS":
            gettessbtn = tk.Button(lcframe, text="Get TESS Data", command=lambda: gettesswrapper(gaia_id, tic, btntip, plotframe, plotctl, plotwrapper, lcframe, queue))
            gettessbtn.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)
            if tic is None:
                gettessbtn["state"] = "disabled"
                btntip.destroy()
                btntip = tk.Label(lcframe, text="No TIC ID was found for this Star!", fg="red", font=('Segoe UI', 25))
                btntip.pack()
        elif name == "ZTF":
            getgaiabtn = tk.Button(lcframe, text="Get ZTF Data", command=lambda: getztfwrapper(gaia_id, btntip, plotframe, plotctl, plotwrapper, queue))
            getgaiabtn.pack(side=tk.BOTTOM, anchor="e", padx=10, pady=10)


def viewlc(window, gaia_id, tic, queue):
    lcwindow = tk.Toplevel(window)
    lcwindow.title(f"View Lightcurve for Gaia DR3 {gaia_id}")
    lcwindow.geometry("800x600+0+0")
    if os.name == 'nt':
        lcwindow.state('zoomed')
    elif os.name == "posix":
        lcwindow.attributes('-zoomed', True)
    fitmodelframe = tk.Frame(lcwindow)
    lc_tabs = ttk.Notebook(fitmodelframe)

    tesslc = tk.Frame(lc_tabs)
    lc_tabs.add(tesslc, text="TESS LC")

    gaialc = tk.Frame(lc_tabs)
    lc_tabs.add(gaialc, text="GAIA LC")

    ztflc = tk.Frame(lc_tabs)
    lc_tabs.add(ztflc, text="ZTF LC")

    if os.path.isfile(f"lightcurves/{gaia_id}/tess_lc.txt"):
        try:
            lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/tess_lc.txt", delimiter=",")
            if lcdata.ndim == 2:
                # lcdata = [lcdata[:, 0], lcdata[:, 1], lcdata[:, 2]]
                pgramdata = np.genfromtxt(f"lightcurves/{gaia_id}/tess_lc_periodogram.txt", delimiter=",")
                lcplottab(lcdata, tesslc, "TESS", gaia_id, tic, queue, pgramdata)
            else:
                notfoundlabel = tk.Label(tesslc, text="No TESS Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
                notfoundlabel.pack()
        except FileNotFoundError:
            lcplottab(None, tesslc, "TESS", gaia_id, tic, queue, None)
    else:
        lcplottab(None, tesslc, "TESS", gaia_id, tic, queue, None)

    if os.path.isfile(f"lightcurves/{gaia_id}/gaia_lc.txt"):
        try:
            lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/gaia_lc.txt", delimiter=",")
            if lcdata.ndim == 2:
                pgramdata = np.genfromtxt(f"lightcurves/{gaia_id}/gaia_lc_periodogram.txt", delimiter=",")
                lcplottab(lcdata, gaialc, "GAIA", gaia_id, tic, queue, pgramdata)
            else:
                pgramdata = None
                notfoundlabel = tk.Label(gaialc, text="No GAIA Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
                notfoundlabel.pack()
        except FileNotFoundError:
            lcplottab(None, gaialc, "GAIA", gaia_id, tic, queue, None)
    else:
        lcplottab(None, gaialc, "GAIA", gaia_id, tic, queue, None)

    if os.path.isfile(f"lightcurves/{gaia_id}/ztf_lc.txt"):
        try:
            lcdata = pd.read_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", delimiter=",")
            numdata = lcdata[lcdata.columns[:3]].to_numpy()
            filter = lcdata[lcdata.columns[-1]].to_numpy()
            filter[filter == "zg"] = 1
            filter[filter == "zi"] = 2
            filter[filter == "zr"] = 3
            lcdata = np.column_stack([numdata, filter]).astype(float)
            if lcdata.ndim == 2:
                try:
                    pgramdata = np.genfromtxt(f"lightcurves/{gaia_id}/ztf_lc_periodogram.txt", delimiter=",")
                except FileNotFoundError:
                    pgramdata = None
            else:
                pgramdata = None
            lcplottab(lcdata, ztflc, "ZTF", gaia_id, tic, queue, pgramdata)
        except FileNotFoundError:
            notfoundlabel = tk.Label(ztflc, text="No ZTF Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
        except pd.errors.EmptyDataError:
            notfoundlabel = tk.Label(ztflc, text="No ZTF Lightcurve was found for this Star!", fg="red", font=('Segoe UI', 25))
            notfoundlabel.pack()
    else:
        lcplottab(None, ztflc, "ZTF", gaia_id, tic, queue, None)

    lc_tabs.pack(fill="both", expand=1)
    fitmodelframe.pack(fill="both", expand=1)


def savenote(gaia_id, text, window=None):
    with open(f"{general_config['OUTPUT_DIR']}/{gaia_id}/note.txt", "w") as notefile:
        notefile.write(text)
    if window is not None:
        window.destroy()


def viewnote(window, gaia_id):
    notewindow = tk.Toplevel(window)
    notewindow.title(f"Note for Gaia DR3 {gaia_id}")
    notewindow.geometry("800x600+0+0")

    noteframe = tk.Frame(notewindow)
    notepad = tk.Text(noteframe)
    if os.path.isfile(f"{general_config['OUTPUT_DIR']}/{gaia_id}/note.txt"):
        with open(f"{general_config['OUTPUT_DIR']}/{gaia_id}/note.txt", "r") as notefile:
            notepad.insert("1.0", notefile.read())
    savebtn = tk.Button(notewindow, text="Save Note", command=lambda: savenote(gaia_id, notepad.get("1.0", 'end-1c')))
    notewindow.protocol("WM_DELETE_WINDOW", lambda: savenote(gaia_id, notepad.get("1.0", 'end-1c'), notewindow))

    notepad.pack(fill="both")
    noteframe.pack(fill="both")
    savebtn.pack(side=tk.RIGHT)


def polynomial(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


def potentialfctn(x, points):
    return -np.sum(1 / (np.abs(x - points) ** 2 + 0.1))


def pointpotentials(sample_arrays, min_val, max_val):
    potentialsample = np.linspace(min_val, max_val, len(sample_arrays[0]) * 100)
    potential = np.zeros(len(potentialsample))
    for sample in sample_arrays:
        potential += np.array([potentialfctn(x, sample) for x in potentialsample])

    interpolated_potential = interp1d(potentialsample, potential, bounds_error=False, fill_value='extrapolate')
    return potentialsample, potential, interpolated_potential


def generate_bins(sample_arrays):
    min_length = np.min([len(s) for s in sample_arrays])
    min_point = np.min([np.min(s) for s in sample_arrays])
    max_point = np.max([np.max(s) for s in sample_arrays])

    sample_arrays = np.stack([s[:min_length] for s in sample_arrays])
    bin_centers = np.median(sample_arrays, axis=0)

    pot_x, pot_y, pot_fn = pointpotentials(sample_arrays, min_point, max_point)

    def fitwrapper(bin_centers, a, b, c, d):
        new_bin_centers = bin_centers + polynomial(bin_centers, a, b, c, d)
        return pot_fn(new_bin_centers)

    params, errs = curve_fit(fitwrapper, bin_centers, np.full(len(bin_centers), pot_y.min()), p0=[0, 0, 0, 0], maxfev=1000000)

    bin_centers += polynomial(bin_centers, *params)

    return bin_centers, pot_x, pot_y


def truncate_and_align_arrays(wls, flxs, flx_stds):
    first_values = [s[0] for s in wls]
    lowest_value = np.max(first_values)
    arr_with_lowest_val = np.argmax(first_values)
    arr_with_lowest_val_stepsize = wls[arr_with_lowest_val][1] - wls[arr_with_lowest_val][0]

    new_wls = []
    new_flxs = []
    new_flx_stds = []
    for f, w, fstd in zip(flxs, wls, flx_stds):
        new_flxs.append(f[w > lowest_value - arr_with_lowest_val_stepsize / 2])
        new_wls.append(w[w > lowest_value - arr_with_lowest_val_stepsize / 2])
        new_flx_stds.append(fstd[w > lowest_value - arr_with_lowest_val_stepsize / 2])

    last_values = [s[-1] for s in wls]
    highest_value = np.min(last_values)
    arr_with_highest_val = np.argmin(last_values)
    arr_with_highest_val_stepsize = wls[arr_with_highest_val][-1] - wls[arr_with_highest_val][-2]

    out_wls = []
    out_flx = []
    out_flx_std = []
    for f, w, fstd in zip(new_flxs, new_wls, new_flx_stds):
        out_flx.append(f[w < highest_value - arr_with_highest_val_stepsize / 2])
        out_wls.append(w[w < highest_value - arr_with_highest_val_stepsize / 2])
        out_flx_std.append(fstd[w < highest_value - arr_with_highest_val_stepsize / 2])

    return out_wls, out_flx, out_flx_std


def simpleplot(x, y):
    plt.close('all')
    plt.plot(x, y)
    plt.tight_layout()
    plt.show()


def create_master_spectrum(queue, gaia_id, assoc_files, custom_name=None, custom_rv=None, custom_select=None):
    if custom_select == "":
        custom_select = None
    if not os.path.isdir("master_spectra"):
        os.mkdir("master_spectra")
    rv_table = pd.read_csv(f"{general_config['OUTPUT_DIR']}/{gaia_id}/RV_variation.csv")

    print("Loading spectra...")
    mjddict = {}
    wls = []
    flxs = []
    flx_stds = []

    for f in assoc_files:
        with open(f"spectra_processed/{f}_mjd.txt", "r") as infile:
            mjds = [float(l.strip()) for l in infile.readlines() if "#" not in l]
        flist = open_spec_files("spectra_processed", [f])
        for mjd, f in zip(mjds, flist):
            mjddict[round(mjd, 3)] = f

    if not custom_select:
        for ind, row in rv_table.iterrows():
            file = mjddict[round(row["mjd"], 3)]
            wl, flx, _, flx_std = load_spectrum(file)
            wl = wlshift(wl, -row["culum_fit_RV"])
            s = wl.argsort()
            wl = wl[s]
            flx = flx[s]
            flx_std = flx_std[s]

            flx_std /= np.nanmedian(flx)
            flx /= np.nanmedian(flx)

            wls.append(wl)
            flxs.append(flx)
            flx_stds.append(flx_std)

        # step 1: calculating wavelength ranges
        ranges = [np.ptp(x) for x in wls]

        # step 2: grouping similar ranges
        eps = 0.05
        groups = []
        for i, r1 in enumerate(ranges):
            thisgroup = []
            for j, r2 in enumerate(ranges):
                if i != j and abs((r1 - r2) / r1) <= eps:
                    thisgroup.append(sorted([i, j]))
            thisgroup = np.unique(list(itertools.chain.from_iterable(thisgroup))).tolist()
            groups.append(thisgroup)

        # sort and remove duplicate groups
        groups.sort()
        groups = list(groups for groups, _ in itertools.groupby(groups))

        # compute frequencies of the groups
        lengths = [len(group) for group in groups]

        # get the max frequency and group
        max_length = max(lengths)
        largest_group = groups[lengths.index(max_length)]
    else:
        rv_table["mjd"] = rv_table["mjd"].round(3)
        for m, f in mjddict.items():
            wl, flx, _, flx_std = load_spectrum(f)
            if custom_rv is not None:
                wl = wlshift(wl, -custom_rv)
            else:
                try:
                    rv = rv_table.loc[rv_table["mjd"] == m].iloc[0]
                except IndexError:
                    print("Some selected columns have no RVs!")
                    return
                wl = wlshift(wl, -rv)
            s = wl.argsort()
            wl = wl[s]
            flx = flx[s]
            flx_std = flx_std[s]

            wls.append(wl)
            flxs.append(flx)
            flx_stds.append(flx_std)
        largest_group = [int(i) for i in custom_select.split(",")]

    print("Grouping spectra...")
    wls = [wls[i] for i in largest_group]
    flxs = [flxs[i] for i in largest_group]
    flx_stds = [flx_stds[i] for i in largest_group]

    print("Truncating...")
    wls, flxs, flx_stds = truncate_and_align_arrays(wls, flxs, flx_stds)
    print("Generating bins...")
    bins, px, py = generate_bins(wls)

    global_wls = np.concatenate([np.array([0]), bins[:-1] + np.diff(bins) / 2])

    # Initialize the global flux array
    global_flxs = np.zeros(len(bins))
    global_flx_stds = np.zeros(len(bins))

    n_in_bin = np.zeros(global_flxs.shape)
    wls = np.concatenate(wls)
    flxs = np.concatenate(flxs)
    flx_stds = np.concatenate(flx_stds)

    print("Binning...")
    for i, bin in enumerate(bins):
        if i == 0:
            global_flxs[0] += np.sum(flxs[wls < bins[0]])
            global_flx_stds[0] += np.sqrt(np.sum(flx_stds[wls < bins[0]] ** 2))
        elif i == len(bins) - 1:
            global_flxs[-1] += np.sum(flxs[wls > bins[-1]])
            global_flx_stds[-1] += np.sqrt(np.sum(flx_stds[wls < bins[-1]] ** 2))
        else:
            mask = np.logical_and(wls >= bin, wls < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            flx_between = flxs[mask]
            flx_std_between = flx_stds[mask]
            wls_between = wls[mask]

            if np.sum(mask) > 1:
                if np.sum(mask) < 5:
                    np.delete(flx_between, flx_std_between.argmax())
                    np.delete(wls_between, flx_std_between.argmax())
                    np.delete(flx_std_between, flx_std_between.argmax())
                else:
                    threshhold = np.nanmedian(flx_between) + 3 * np.std(flx_between)
                    np.delete(flx_std_between, flx_between > threshhold)
                    np.delete(wls_between, flx_between > threshhold)
                    np.delete(flx_between, flx_between > threshhold)

            frac_to_next_bin = (wls_between - bin) / (bins[i + 1] - bin)
            global_flxs[i] += np.sum(flx_between * (1 - frac_to_next_bin)) / len(flx_between)
            global_flxs[i + 1] += np.sum(flx_between * frac_to_next_bin) / len(flx_between)
            global_flx_stds[i] += np.sqrt(np.sum((flx_std_between * (1 - frac_to_next_bin)) ** 2)) / len(flx_std_between)
            global_flx_stds[i + 1] += np.sqrt(np.sum((flx_std_between * frac_to_next_bin) ** 2)) / len(flx_std_between)
            n_in_bin[i] += len(flx_between)
            n_in_bin[i + 1] += len(flx_between)

    n_in_bin /= 2
    normal_n_count = float(np.argmax(np.bincount(n_in_bin.astype(int))))

    global_flxs = global_flxs[n_in_bin == normal_n_count]
    global_flx_stds = global_flx_stds[n_in_bin == normal_n_count]
    global_wls = global_wls[n_in_bin == normal_n_count]
    n_in_bin = n_in_bin[n_in_bin == normal_n_count]

    n_in_bin = n_in_bin[1:-1]
    global_flxs = global_flxs[1:-1]
    global_wls = global_wls[1:-1]
    global_flx_stds = global_flx_stds[1:-1]
    global_flxs /= n_in_bin
    global_flx_stds /= n_in_bin

    queue.put(["execute_function", simpleplot, (global_wls, global_flxs)])

    outdata = np.stack((global_wls, global_flxs, np.zeros(global_wls.shape)), axis=-1)
    if custom_name is None or custom_name == "":
        np.savetxt(f"master_spectra/{gaia_id}_stacked.txt", outdata, fmt='%1.4f')
    else:
        np.savetxt(f"master_spectra/{custom_name}.txt", outdata, fmt='%1.4f')


def viewplot(q, n, gaia_id):
    q.put(["execute_function",
           plot_system_from_ind,
           {"ind": str(gaia_id), "use_ind_as_sid": True, "normalized": n}])


def viewmasterplot(q, n, gaia_id):
    q.put(["execute_function",
           plot_system_from_file,
           {"source_id": str(gaia_id), "normalized": n}])


def phsin(x, period, offset, amplitude, shift):
    return offset + amplitude * np.sin((x - shift) * 2 * np.pi / period)


def make_subplot(ind, d, a, offset, amplitude, shift, nplot, data, tessbin, **kwargs):
    a.grid(True)
    if d["name"] == "RV":
        a.set_ylabel("Radial Velocity [km/s]")
        sinspace = np.linspace(-1, 1, 1000)
        a.plot(sinspace, phsin(sinspace, 1, offset, amplitude, 0), color="darkred", zorder=3)
    else:
        a.set_ylabel("Relative Flux [arb. unit]")
    if ind == nplot - 1:
        a.set_xlabel("Period")
    else:
        a.set_xticklabels([])
    if d["name"] == "TESS":
        mask = np.argsort(d["time"])

        d["time"] = d["time"][mask]
        d["flux"] = d["flux"][mask]
        d["flux_error"] = d["flux_error"][mask]

        if tessbin % 2 != 0:
            tessbin += 1

        # meanfilter = np.convolve(d["flux"], np.ones(tessbin) / tessbin, mode='valid')
        n = len(d["flux"] // tessbin)
        excess = len(d["flux"]) % tessbin
        if excess > 0:
            d["time"] = d["time"][:-excess]
            d["flux"] = d["flux"][:-excess]
            d["flux_error"] = d["flux_error"][:-excess]
        binned_time = np.mean(d["time"][:n * tessbin].reshape(-1, tessbin), axis=1)
        binned_flux = np.mean(d["flux"][:n * tessbin].reshape(-1, tessbin), axis=1)
        binned_error = np.sqrt(np.sum(d["flux_error"][:n * tessbin].reshape(-1, tessbin) ** 2, axis=1)) / tessbin

        a.scatter(binned_time, binned_flux, color="darkred", zorder=5)
        a.errorbar(binned_time, binned_flux, yerr=binned_error, linestyle='', color="darkred", capsize=3, zorder=4)

        # a.set_ylim(meanfilter.min() - 0.05 * np.ptp(meanfilter), meanfilter.max() + 0.05 * np.ptp(meanfilter))
        a.set_ylim(binned_flux.min() - 0.05 * np.ptp(binned_flux), binned_flux.max() + 0.05 * np.ptp(binned_flux))
    elif d["name"] == "RV":
        a.scatter(d["time"], d["flux"], color="navy", zorder=5)
        a.errorbar(d["time"], d["flux"], yerr=d["flux_error"], linestyle='', color="navy", capsize=3, zorder=4)
    elif d["name"] == "GAIA":
        for c, i in zip(["darkgreen", "navy", "darkred"], range(3)):
            a.scatter(d[f"time{i}"], d[f"flux{i}"], color=c, zorder=5)
            a.errorbar(d[f"time{i}"], d[f"flux{i}"], yerr=d[f"flux_error{i}"], linestyle='', color=c, capsize=3, zorder=4)
    elif d["name"] == "ZTF":
        for c, i in zip(["darkgreen", "red", "darkred"], range(3)):
            a.scatter(d[f"time{i}"], d[f"flux{i}"], color=c, zorder=5)
            a.errorbar(d[f"time{i}"], d[f"flux{i}"], yerr=d[f"flux_error{i}"], linestyle='', color=c, capsize=3, zorder=4)
    a.legend([d["name"]], loc="upper right")


def phasefoldplot(resultqueue, offset, amplitude, shift, nplot, data, tessbin=100, **kwargs):
    fig, axs = plt.subplots(nrows=nplot, ncols=1, figsize=(4.8 * 16 / 9, 4.8), dpi=90 * 9 / 4.8, sharex=True)

    if not isinstance(axs, plt.Axes):
        for ind, [d, a] in enumerate(zip(data, axs)):
            make_subplot(ind, d, a, offset, amplitude, shift, nplot, data, tessbin, **kwargs)
    else:
        make_subplot(0, data[0], axs, offset, amplitude, shift, nplot, data, tessbin, **kwargs)
    fig.subplots_adjust(wspace=0)
    resultqueue.put(fig)


def phasefoldplot_wrapper(plotframe, period, multiplier, shift, offset, amplitude, dataarray, textbox, *args, **kwargs):
    if period is None:
        period = 1
    elif isinstance(period, int):
        period = float(period) * multiplier
    elif not isinstance(period, float):
        period = period.get() * multiplier
    else:
        period *= multiplier

    for widget in plotframe.winfo_children():
        widget.destroy()

    datalist = []
    if kwargs["finetune"]:
        try:
            p_amt_whole_phase = 1 / (2 * np.max(np.diff(dataarray["RV"][0])))
            params, errs = curve_fit(phsin, dataarray["RV"][0], dataarray["RV"][1],
                                     [period, offset, amplitude, shift],
                                     dataarray["RV"][2], maxfev=100000,
                                     bounds=[
                                         [period - p_amt_whole_phase, offset * .5, amplitude * .5, 0],
                                         [period + p_amt_whole_phase, offset * 2, amplitude * 2, 1]
                                     ])
            period, offset, amplitude, shift = params
            shift = 1 - shift
            if textbox is not None:
                textbox.config(state=tk.NORMAL)
                textbox.delete(1.0, tk.END)
                textbox.insert(tk.END, "Current fit parameters:\n\n")
                textbox.insert(tk.END, f"period={period}\n")
                textbox.insert(tk.END, f"offset={offset}\n")
                textbox.insert(tk.END, f"amplitude={amplitude}\n")
                textbox.insert(tk.END, f"shift={shift}")
                textbox.config(state=tk.DISABLED)
        except RuntimeError as e:
            textbox.config(state=tk.NORMAL)
            textbox.delete(1.0, tk.END)
            textbox.insert(tk.END, "Current fit parameters:\n\n")
            textbox.insert(tk.END, f"period={period}\n")
            textbox.insert(tk.END, f"offset={offset}\n")
            textbox.insert(tk.END, f"amplitude={amplitude}\n")
            textbox.insert(tk.END, f"shift={shift}")
            textbox.config(state=tk.DISABLED)
            pass
    else:
        if textbox is not None:
            textbox.config(state=tk.NORMAL)
            textbox.delete(1.0, tk.END)
            textbox.insert(tk.END, "Current fit parameters:\n\n")
            textbox.insert(tk.END, f"period={period}\n")
            textbox.insert(tk.END, f"offset={offset}\n")
            textbox.insert(tk.END, f"amplitude={amplitude}\n")
            textbox.insert(tk.END, f"shift={shift}")
            textbox.config(state=tk.DISABLED)

    vtimes = (((dataarray["RV"][0] + (shift * period)) % period) / period)
    vtimes[vtimes > 1] -= 1
    datalist.append({
        "name": "RV",
        "time": np.concatenate([vtimes - 1, vtimes]),
        "flux": np.concatenate([dataarray["RV"][1], dataarray["RV"][1]]),
        "flux_error": np.concatenate([dataarray["RV"][2], dataarray["RV"][2]]),
    })

    if dataarray["GAIA"] is not None:
        gtimes = (((dataarray["GAIA"][0][0] + (shift * period)) % period) / period)
        gtimes[gtimes > 1] -= 1

        rptimes = (((dataarray["GAIA"][1][0] + (shift * period)) % period) / period)
        rptimes[rptimes > 1] -= 1

        bptimes = (((dataarray["GAIA"][2][0] + (shift * period)) % period) / period)
        bptimes[bptimes > 1] -= 1
        datalist.append({
            "name": "GAIA",
            "time0": np.concatenate([gtimes - 1, gtimes]),
            "flux0": np.concatenate([dataarray["GAIA"][0][1], dataarray["GAIA"][0][1]]),
            "flux_error0": np.concatenate([dataarray["GAIA"][0][2], dataarray["GAIA"][0][2]]),
            "time1": np.concatenate([bptimes - 1, bptimes]),
            "flux1": np.concatenate([dataarray["GAIA"][1][1], dataarray["GAIA"][1][1]]),
            "flux_error1": np.concatenate([dataarray["GAIA"][1][2], dataarray["GAIA"][1][2]]),
            "time2": np.concatenate([rptimes - 1, rptimes]),
            "flux2": np.concatenate([dataarray["GAIA"][2][1], dataarray["GAIA"][2][1]]),
            "flux_error2": np.concatenate([dataarray["GAIA"][2][2], dataarray["GAIA"][2][2]]),
        })

    if dataarray["ZTF"] is not None:
        gtimes = (((dataarray["ZTF"][0][0] + (shift * period)) % period) / period)
        gtimes[gtimes > 1] -= 1

        rtimes = (((dataarray["ZTF"][1][0] + (shift * period)) % period) / period)
        rtimes[rtimes > 1] -= 1

        itimes = (((dataarray["ZTF"][2][0] + (shift * period)) % period) / period)
        itimes[itimes > 1] -= 1
        datalist.append({
            "name": "ZTF",
            "time0": np.concatenate([gtimes - 1, gtimes]),
            "flux0": np.concatenate([dataarray["ZTF"][0][1], dataarray["ZTF"][0][1]]),
            "flux_error0": np.concatenate([dataarray["ZTF"][0][2], dataarray["ZTF"][0][2]]),
            "time1": np.concatenate([rtimes - 1, rtimes]),
            "flux1": np.concatenate([dataarray["ZTF"][1][1], dataarray["ZTF"][1][1]]),
            "flux_error1": np.concatenate([dataarray["ZTF"][1][2], dataarray["ZTF"][1][2]]),
            "time2": np.concatenate([itimes - 1, itimes]),
            "flux2": np.concatenate([dataarray["ZTF"][2][1], dataarray["ZTF"][2][1]]),
            "flux_error2": np.concatenate([dataarray["ZTF"][2][2], dataarray["ZTF"][2][2]]),
        })

    if dataarray["TESS"] is not None:
        ttimes = (((dataarray["TESS"][0] + (shift * period)) % period) / period)
        ttimes[ttimes > 1] -= 1

        datalist.append({
            "name": "TESS",
            "time": np.concatenate([ttimes - 1, ttimes]),
            "flux": np.concatenate([dataarray["TESS"][1], dataarray["TESS"][1]]),
            "flux_error": np.concatenate([dataarray["TESS"][2], dataarray["TESS"][2]]),
        })

    def plotpffold():
        plt.close('all')
        resultqueue = man.Queue()
        queue.put(["execute_function", phasefoldplot, (resultqueue, offset, amplitude, shift, len(datalist), datalist), kwargs])
        fig = resultqueue.get()
        canvas = FigureCanvasTkAgg(fig, master=plotframe)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, plotframe)
        toolbar.update()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    thread = threading.Thread(target=plotpffold)
    thread.start()


def varvalidation(var):
    try:
        return float(var.get())
    except:
        return 1.0


def extract_values(text_widget):
    # Get the content of the text widget
    content = text_widget.get("1.0", tk.END)

    # Split the content into lines
    lines = content.split("\n")

    # Initialize variables
    period = None
    offset = None
    amplitude = None
    shift = None

    # Parse each line to extract the values
    for line in lines:
        if line.startswith("period="):
            period = float(line.split("=")[1])
        elif line.startswith("offset="):
            offset = float(line.split("=")[1])
        elif line.startswith("amplitude="):
            amplitude = float(line.split("=")[1])
        elif line.startswith("shift="):
            shift = float(line.split("=")[1])

    return period, offset, amplitude, shift


def phasefold(analysis, gaia_id):
    pf_window = tk.Toplevel(analysis)
    try:
        if os.name == "nt":
            pf_window.iconbitmap("favicon.ico")
        else:
            imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
            pf_window.tk.call('wm', 'iconphoto', pf_window._w, imgicon)
    except:
        pass
    pf_window.title(f"Phasefolding for Gaia DR3 {gaia_id}")
    pf_window.update_idletasks()  # This forces tkinter to update the window calculations.
    pf_window.geometry("800x600+0+0")
    if os.name == 'nt':
        pf_window.state('zoomed')
    elif os.name == "posix":
        pf_window.attributes('-zoomed', True)

    plotwrapper = tk.Frame(pf_window)
    plotframe = tk.Frame(plotwrapper)
    plotctl = tk.Frame(plotwrapper)

    # Load RV, Gaia and TESS LC, if available.
    rvdata = pd.read_csv(f"output/{gaia_id}/RV_variation.csv")
    vels = rvdata["culum_fit_RV"].to_numpy()
    verrs = rvdata["u_culum_fit_RV"].to_numpy()
    mjd = rvdata["mjd"].to_numpy()
    dataarray = {"RV": [mjd, vels, verrs]}

    if os.path.isfile(f"lightcurves/{gaia_id}/gaia_lc.txt") and os.path.isfile(f"lightcurves/{gaia_id}/gaia_lc_periodogram.txt"):
        gaia_lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/gaia_lc.txt", delimiter=",")
        # "TimeBP", "FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"

        gaia_t = gaia_lcdata[:, 0]
        gaia_g_flx = gaia_lcdata[:, 3]
        gaia_g_flx_err = gaia_lcdata[:, 6]
        gaia_bp_t = gaia_lcdata[:, 1]
        gaia_bp_flx = gaia_lcdata[:, 4]
        gaia_bp_flx_err = gaia_lcdata[:, 7]
        gaia_rp_t = gaia_lcdata[:, 2]
        gaia_rp_flx = gaia_lcdata[:, 5]
        gaia_rp_flx_err = gaia_lcdata[:, 8]

        gaia_t = gaia_t[~np.isnan(gaia_t)]
        gaia_g_flx = gaia_g_flx[~np.isnan(gaia_g_flx)]
        gaia_g_flx_err = gaia_g_flx_err[~np.isnan(gaia_g_flx_err)]
        gaia_bp_t = gaia_bp_t[~np.isnan(gaia_bp_t)]
        gaia_bp_flx = gaia_bp_flx[~np.isnan(gaia_bp_flx)]
        gaia_bp_flx_err = gaia_bp_flx_err[~np.isnan(gaia_bp_flx_err)]
        gaia_rp_t = gaia_rp_t[~np.isnan(gaia_rp_t)]
        gaia_rp_flx = gaia_rp_flx[~np.isnan(gaia_rp_flx)]
        gaia_rp_flx_err = gaia_rp_flx_err[~np.isnan(gaia_rp_flx_err)]

        gaia_t += 2455197.5
        gaia_bp_t += 2455197.5
        gaia_rp_t += 2455197.5

        gaia_t = Time(gaia_t, format="jd").mjd
        gaia_bp_t = Time(gaia_bp_t, format="jd").mjd
        gaia_rp_t = Time(gaia_rp_t, format="jd").mjd

        gaia_g_flx_err /= np.nanmedian(gaia_g_flx)
        gaia_g_flx /= np.nanmedian(gaia_g_flx)
        gaia_bp_flx_err /= np.nanmedian(gaia_bp_flx)
        gaia_bp_flx /= np.nanmedian(gaia_bp_flx)
        gaia_rp_flx_err /= np.nanmedian(gaia_rp_flx)
        gaia_rp_flx /= np.nanmedian(gaia_rp_flx)

        gaia_pgram_data = np.genfromtxt(f"lightcurves/{gaia_id}/gaia_lc_periodogram.txt", delimiter=",")
        gaia_periods = gaia_pgram_data[:, 0]
        gaia_power = gaia_pgram_data[:, 1]

        dataarray["GAIA"] = [
            [gaia_t, gaia_g_flx, gaia_g_flx_err],
            [gaia_bp_t, gaia_bp_flx, gaia_bp_flx_err],
            [gaia_rp_t, gaia_rp_flx, gaia_rp_flx_err]
        ]

        gaia_maxp = gaia_periods[np.argmax(gaia_power)]
        del gaia_lcdata, gaia_pgram_data
    else:
        gaia_maxp = 1
        dataarray["GAIA"] = None

    if os.path.isfile(f"lightcurves/{gaia_id}/tess_lc.txt") and os.path.isfile(f"lightcurves/{gaia_id}/tess_lc_periodogram.txt"):
        tess_lcdata = np.genfromtxt(f"lightcurves/{gaia_id}/tess_lc.txt", delimiter=",")
        tess_t = Time(tess_lcdata[:, 0] + 2457000, format="jd").mjd  # TESS BJD to MJD
        tess_flx = tess_lcdata[:, 1]
        tess_flx_err = tess_lcdata[:, 2]

        tess_pgram_data = np.genfromtxt(f"lightcurves/{gaia_id}/tess_lc_periodogram.txt", delimiter=",")
        tess_periods = tess_pgram_data[:, 0]
        tess_power = tess_pgram_data[:, 1]
        tess_maxp = tess_periods[np.argmax(tess_power)]

        dataarray["TESS"] = [tess_t, tess_flx, tess_flx_err]

        del tess_lcdata, tess_pgram_data
    else:
        tess_t = None
        tess_maxp = 1
        dataarray["TESS"] = None

    if os.path.isfile(f"lightcurves/{gaia_id}/ztf_lc.txt") and os.path.isfile(f"lightcurves/{gaia_id}/ztf_lc_periodogram.txt"):
        ztf_lcdata = pd.read_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", delimiter=",")
        numdata = ztf_lcdata[ztf_lcdata.columns[:3]].to_numpy()
        ztf_t = numdata[:, 0]
        ztf_flx = numdata[:, 1]
        ztf_flx_err = numdata[:, 2]
        filter = ztf_lcdata[ztf_lcdata.columns[-1]].to_numpy()

        gmask = filter == "zg"
        imask = filter == "zi"
        rmask = filter == "zr"

        ztf_g_t = ztf_t[gmask]
        ztf_i_t = ztf_t[imask]
        ztf_r_t = ztf_t[rmask]
        ztf_g_flx = ztf_flx[gmask]
        ztf_i_flx = ztf_flx[imask]
        ztf_r_flx = ztf_flx[rmask]
        ztf_g_flx_err = ztf_flx_err[gmask]
        ztf_i_flx_err = ztf_flx_err[imask]
        ztf_r_flx_err = ztf_flx_err[rmask]

        ztf_g_flx_err /= np.nanmedian(ztf_g_flx)
        ztf_g_flx /= np.nanmedian(ztf_g_flx)
        ztf_i_flx_err /= np.nanmedian(ztf_i_flx)
        ztf_i_flx /= np.nanmedian(ztf_i_flx)
        ztf_r_flx_err /= np.nanmedian(ztf_r_flx)
        ztf_r_flx /= np.nanmedian(ztf_r_flx)

        ztf_pgram_data = np.genfromtxt(f"lightcurves/{gaia_id}/ztf_lc_periodogram.txt", delimiter=",")
        ztf_periods = ztf_pgram_data[:, 0]
        ztf_power = ztf_pgram_data[:, 1]
        ztf_maxp = ztf_periods[np.argmax(ztf_power)]

        dataarray["ZTF"] = [
            [ztf_g_t, ztf_g_flx, ztf_g_flx_err],
            [ztf_i_t, ztf_i_flx, ztf_i_flx_err],
            [ztf_r_t, ztf_r_flx, ztf_r_flx_err]
        ]

        del ztf_lcdata, ztf_pgram_data
    else:
        dataarray["ZTF"] = None
        ztf_maxp = None

    outputtext = tk.Text(plotctl, height=7, width=42)

    if os.path.isfile(f"phasefolds/{gaia_id}/orbit.txt"):
        with open(f"phasefolds/{gaia_id}/orbit.txt", "r") as file:
            lines = file.readlines()

            if len(lines) >= 2:
                data = lines[1].split(",")
                preload = True
                startperiod, half_amp, offset, phase = [float(d) for d in data]
            else:
                print("File does not have a second line")
                half_amp = np.ptp(vels) / 2
                offset = np.mean(vels)
                phase = 0
                startperiod = tess_maxp
                preload = False
    else:
        preload = False
        half_amp = np.ptp(vels) / 2
        offset = np.mean(vels)
        phase = 0
        startperiod = tess_maxp

    phasefoldplot_wrapper(plotframe, startperiod, 1, phase, offset, half_amp,
                          dataarray, outputtext, finetune=False, tessbin=int(np.sqrt(len(tess_t if tess_t is not None else [0]))))

    periods_dict = {"RV": 1, "GAIA": gaia_maxp, "TESS": tess_maxp, "ZTF": ztf_maxp}

    peaksource_var = tk.StringVar(value="TESS")
    peaksource_timesentry_var = tk.StringVar(value="1")
    p_amt_whole_phase = 5 / (2 * np.max(np.diff(mjd)))
    pslider_frame = tk.Frame(plotctl)
    pslider_label = tk.Label(pslider_frame, text="Modify Period:")
    pslider_slider = tk.Scale(pslider_frame,
                              from_=periods_dict[peaksource_var.get()] * varvalidation(peaksource_timesentry_var) + p_amt_whole_phase,
                              to=periods_dict[peaksource_var.get()] * varvalidation(peaksource_timesentry_var) - p_amt_whole_phase,
                              resolution=p_amt_whole_phase / 10000,
                              tickinterval=p_amt_whole_phase / 2,
                              length=300,
                              orient=tk.HORIZONTAL)
    pslider_slider.set(startperiod)
    pslider_label.grid(column=0, row=0)
    pslider_slider.grid(column=0, row=1)
    pslider_frame.grid(column=0, row=4, sticky="w")

    def updateperiodslider(source, force_low=None, force_high=None):
        print(source, force_low, force_high)
        if source is not None and source != "Slider":
            p_amt_whole_phase = 5 / (2 * np.max(np.diff(mjd)))
            pslider_slider.configure(from_=periods_dict[source] * varvalidation(peaksource_timesentry_var) + p_amt_whole_phase,
                                     to=periods_dict[source] * varvalidation(peaksource_timesentry_var) - p_amt_whole_phase,
                                     resolution=p_amt_whole_phase / 10000,
                                     tickinterval=p_amt_whole_phase / 2)
            pslider_slider.set(periods_dict[peaksource_var.get()] * varvalidation(peaksource_timesentry_var))
        else:
            if source == "Slider":
                return
            elif force_high != force_low and force_high > force_low:
                pslider_slider.configure(from_=force_low, to=force_high,
                                         tickinterval=(force_high - force_low) / 2,
                                         resolution=(force_high - force_low) / 10000)
                pslider_slider.set((force_low + force_high) / 2)
            else:
                pslider_slider.configure(from_=0.5, to=1.5,
                                         tickinterval=1 / 2,
                                         resolution=1 / 10000)
                pslider_slider.set((0.5 + 1.5) / 2)

    periodframe = tk.Frame(plotctl)
    loperiodvar = tk.StringVar(value="0.5")
    hiperiodvar = tk.StringVar(value="1.5")

    periodlabel = tk.Label(periodframe, text="Set slider range from ", padx=5)
    periodinterlabel = tk.Label(periodframe, text=" to ", padx=5)
    periodsetbtn = tk.Button(periodframe,
                             command=lambda: updateperiodslider(None,
                                                                force_low=varvalidation(loperiodvar),
                                                                force_high=varvalidation(hiperiodvar)),
                             text="Set")
    periodloentry = tk.Entry(periodframe, textvariable=loperiodvar, width=4)
    periodhientry = tk.Entry(periodframe, textvariable=hiperiodvar, width=4)
    periodlabel.grid(row=0, column=0)
    periodloentry.grid(row=0, column=1)
    periodinterlabel.grid(row=0, column=2)
    periodhientry.grid(row=0, column=3)
    periodsetbtn.grid(row=0, column=4)

    peaksource_selector_frame = tk.Frame(plotctl)
    peaksource_label = tk.Label(peaksource_selector_frame, text="Get Period from ")

    peaksource_selector = tk.OptionMenu(peaksource_selector_frame, peaksource_var, "RV", "TESS", "GAIA", "ZTF", "Slider",
                                        command=updateperiodslider)
    peaksource_label.pack(side=tk.LEFT)
    peaksource_selector.pack(side=tk.LEFT)
    peaksource_label_two = tk.Label(peaksource_selector_frame, text=" x")
    peaksource_label_two.pack(side=tk.LEFT)
    peaksource_timesentry = tk.Entry(peaksource_selector_frame, textvariable=peaksource_timesentry_var, width=4)
    peaksource_timesentry.pack(side=tk.LEFT)
    peaksource_selector_frame.grid(column=0, row=0, sticky="w")

    tessbin_frame = tk.Frame(plotctl)
    tessbin_label = tk.Label(tessbin_frame, text="TESS binning ")
    if tess_t is not None:
        binvar = str(int(np.sqrt(len(tess_t))))
    else:
        binvar = "0"
    tessbin_var = tk.StringVar(value=binvar)
    tessbin = tk.Entry(tessbin_frame, textvariable=tessbin_var)
    tessbin_label.pack(side=tk.LEFT)
    tessbin.pack(side=tk.LEFT)
    tessbin_frame.grid(column=0, row=1, sticky="w")

    ftvar = tk.IntVar(value=0)
    finetune = tk.Checkbutton(plotctl, variable=ftvar, text="Finetune RV fit")
    finetune.grid(column=0, row=2)
    periodframe.grid(column=0, row=3)

    ampslider_frame = tk.Frame(plotctl)
    ampslider_label = tk.Label(ampslider_frame, text="Modify Half-Amplitude:")
    ampslider_slider = tk.Scale(ampslider_frame,
                                from_=0,
                                to=500,
                                resolution=1,
                                tickinterval=100,
                                length=300,
                                orient=tk.HORIZONTAL)
    ampslider_slider.set(half_amp)
    ampslider_label.grid(column=0, row=0)
    ampslider_slider.grid(column=0, row=1)
    ampslider_frame.grid(column=0, row=5, sticky="w")

    offslider_frame = tk.Frame(plotctl)
    offslider_label = tk.Label(offslider_frame, text="Modify Offset:")
    offslider_slider = tk.Scale(offslider_frame,
                                from_=-250,
                                to=250,
                                resolution=1,
                                tickinterval=100,
                                length=300,
                                orient=tk.HORIZONTAL)
    offslider_slider.set(offset)
    offslider_label.grid(column=0, row=0)
    offslider_slider.grid(column=0, row=1)
    offslider_frame.grid(column=0, row=6, sticky="w")

    shiftslider_frame = tk.Frame(plotctl)
    shiftslider_label = tk.Label(shiftslider_frame, text="Modify Shift:")
    shiftslider_slider = tk.Scale(shiftslider_frame,
                                  from_=0,
                                  to=1,
                                  resolution=1 / 100,
                                  tickinterval=0.25,
                                  length=300,
                                  orient=tk.HORIZONTAL)
    shiftslider_label.grid(column=0, row=0)
    shiftslider_slider.grid(column=0, row=1)
    shiftslider_slider.set(phase)
    shiftslider_frame.grid(column=0, row=7, sticky="w")

    periods_dict["Slider"] = pslider_slider

    refold = tk.Button(plotctl, text="Redo Phasefold", command=lambda: phasefoldplot_wrapper(plotframe, periods_dict[peaksource_var.get()], varvalidation(peaksource_timesentry_var), shiftslider_slider.get(), offslider_slider.get(), ampslider_slider.get(),
                                                                                             dataarray, outputtext, tessbin=int(tessbin_var.get()),
                                                                                             finetune=bool(ftvar.get())))

    refold.grid(column=0, row=9, sticky="se")

    def savephfold():
        if not os.path.isdir(f"phasefolds/{gaia_id}"):
            os.mkdir(f"phasefolds/{gaia_id}")
        period, offset, K, phase = extract_values(outputtext)
        with open(f"phasefolds/{gaia_id}/orbit.txt", "w") as file:
            file.write(f"period,half_amplitude,offset,phase\n{period},{K},{offset},{phase}")

    outputtext.config(state=tk.DISABLED)
    outputtext.grid(column=0, row=11)
    saveparams = tk.Button(plotctl, text="Save Parameters", command=savephfold)
    saveparams.grid(column=0, row=12, sticky="se")

    for i in range(12):
        plotctl.grid_rowconfigure(i, minsize=25)
    plotframe.grid(row=0, column=0, pady=10, padx=10)
    plotctl.grid(row=0, column=1, sticky="n", pady=10, padx=10)
    plotwrapper.pack()


def galacticorbit(parameters, d):
    plt.close('all')
    ra = parameters["ra"]
    dec = parameters["dec"]

    pmra = parameters["pmra"]
    pmdec = parameters["pmdec"]

    rvavg = parameters["RVavg"]

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=d * u.pc,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 radial_velocity=rvavg * u.km / u.s)

    p = MWPotential2014
    o = Orbit(c)
    ts = np.linspace(0, 50, 5000)
    o.integrate(ts, p)

    plt.plot(o.R(ts), o.z(ts), color="darkgreen", zorder=5)
    plt.plot([o.R()], [o.z()], 'rx', zorder=6)
    plt.grid(True, linestyle="--", color="lightgray")
    plt.ylabel("z [kpc]")
    plt.xlabel("R [kpc]")
    plt.xlim(0, 15)
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.show()


def galacticorbit_wrapper(interesting_dataframe, w, gaia_id):
    parameters = interesting_dataframe[interesting_dataframe["source_id"] == gaia_id].iloc[0]
    plx = parameters["parallax"]
    plx_err = parameters["parallax_err"]

    if plx_err > 0.5 * plx and not plx_err >= plx:
        messagebox.showinfo("Uncertain Parralax", "The plot you are about to view was calculated with an unreliable parralax", parent=w)
        d = 1 / (np.abs(plx) * 0.001)
    elif plx_err >= plx:
        messagebox.showwarning("Bad Parralax", "The plot you are about to view was calculated with a bad parralax", parent=w)
        d = 1 / ((np.abs(plx) + plx_err) * 0.001)
    else:
        d = 1 / (plx * 0.001)
    queue.put(["execute_function", galacticorbit, (parameters, d)])


def analysis_tab(analysis, queue):
    frame = tk.Frame(analysis)
    try:
        interesting_dataframe = pd.read_csv("result_parameters.csv")
        sheet = Sheet(frame,
                      data=interesting_dataframe.values.tolist())
    except FileNotFoundError:
        no_results_yet = tk.Label(frame,
                                  font=('Segoe UI', 25),
                                  fg='#ff0000',
                                  text="No results to show yet!\nGenerate some using the processing tab!")
        no_results_yet.place(in_=frame, anchor="center", relx=.5, rely=.5)
        frame.pack(fill="both", expand=1)
        return

    tablesettings_frame = tk.Frame(analysis)

    current_dataframe = interesting_dataframe

    show_known = tk.IntVar(frame, value=1)
    show_unknown = tk.IntVar(frame, value=1)
    show_likely_known = tk.IntVar(frame, value=1)
    show_indeterminate = tk.IntVar(frame, value=1)
    highlight = tk.IntVar(frame, value=1)
    exclude_columns = tk.IntVar(frame, value=1)
    filter_kws = tk.StringVar(frame, value="")
    sortset = tk.StringVar(frame, value="logp")

    def highlight_cells(sheet, hlgt):
        nonlocal current_dataframe
        # Nspec, Deltarv, rvavg
        sheet.dehighlight_all()
        if hlgt == 1:
            columns = current_dataframe.columns.to_list()

            cell_colors(sheet,
                        current_dataframe["logp"][current_dataframe["logp"] < -4].index.tolist(),
                        current_dataframe["logp"][np.logical_and(current_dataframe["logp"] > -4, current_dataframe["logp"] < -1.3)].index.tolist(),
                        columns.index("logp"),
                        ("#5ced73", "#abf7b1"))

            cell_colors(sheet,
                        current_dataframe["deltaRV"][current_dataframe["deltaRV"] > 250].index.tolist(),
                        current_dataframe["deltaRV"][np.logical_and(current_dataframe["deltaRV"] < 250, current_dataframe["deltaRV"] > 150)].index.tolist(),
                        columns.index("deltaRV"),
                        ("#5ced73", "#abf7b1"))

            cell_colors(sheet,
                        current_dataframe["RVavg"][np.abs(current_dataframe["RVavg"]) > 250].index.tolist(),
                        current_dataframe["RVavg"][np.logical_and(np.abs(current_dataframe["RVavg"]) < 250, np.abs(current_dataframe["RVavg"]) > 250)].index.tolist(),
                        columns.index("RVavg"),
                        ("#5ced73", "#abf7b1"))

            cell_colors(sheet,
                        current_dataframe["gmag"][np.abs(current_dataframe["gmag"]) < 18].index.tolist(),
                        current_dataframe["gmag"][np.logical_and(current_dataframe["gmag"] < 19, current_dataframe["gmag"] > 18)].index.tolist(),
                        columns.index("gmag"),
                        ("#5ced73", "#abf7b1"))

            try:
                sids = pd.read_csv("observation_list.csv")["source_id"].to_list()
                cell_colors(sheet,
                            current_dataframe["source_id"][current_dataframe["source_id"].isin(sids)].index.tolist(),
                            current_dataframe["gmag"][np.logical_and(current_dataframe["gmag"] < 19, current_dataframe["gmag"] > 19)].index.tolist(),
                            columns.index("source_id"),
                            ("#5ced73", "#abf7b1"))
            except FileNotFoundError:
                pass

    highlight_cells(sheet, highlight.get())

    def prep_kstring(kstring):
        kstring = kstring.replace("in", "=")
        kstring = kstring.replace("<", "")
        kstring = kstring.replace(">", "")
        kstring = kstring.replace("!", "")
        sp = kstring.split("=")

        sp[0] = sp[0].strip()
        sp[1] = sp[1].strip()
        try:
            if "." in sp[1]:
                sp[1] = float(sp[1])
            else:
                sp[1] = int(sp[1])

        except:
            pass

        if sp[1] == "nan":
            sp[1] = np.nan

        return sp

    def update_sheet(*args):
        filters = []
        nonlocal current_dataframe
        if show_known.get() == 1:
            filters.append("known")
        if show_unknown.get() == 1:
            filters.append("unknown")
        if show_likely_known.get() == 1:
            filters.append("likely_known")
        if show_indeterminate.get() == 1:
            filters.append("indeterminate")
        if exclude_columns.get() == 1:
            excluded_cols = ["deltaRV_err", "RVavg_err", "bibcodes", "associated_files", "known_category", "flags", "pmra_err", "pmdec_err"]
        else:
            excluded_cols = []

        if "known_category" in interesting_dataframe.columns:
            current_dataframe = interesting_dataframe[interesting_dataframe["known_category"].isin(filters)].reset_index(drop=True)
        else:
            current_dataframe = interesting_dataframe.reset_index(drop=True)

        keywords = filter_kws.get().split(";")
        for k in keywords:
            if "=" in k and "<=" not in k and ">=" not in k and "!=" not in k:
                sp = prep_kstring(k)
                if not pd.isnull(sp[1]):
                    current_dataframe = current_dataframe[current_dataframe[sp[0]] == sp[1]].reset_index(drop=True)
                else:
                    current_dataframe = current_dataframe[pd.isnull(current_dataframe)].reset_index(drop=True)

            elif "<=" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[0]] <= sp[1]].reset_index(drop=True)
            elif ">=" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[0]] >= sp[1]].reset_index(drop=True)
            elif "!=" in k:
                sp = prep_kstring(k)
                if not pd.isnull(sp[1]):
                    current_dataframe = current_dataframe[current_dataframe[sp[0]] != sp[1]].reset_index(drop=True)
                else:
                    current_dataframe = current_dataframe[~pd.isnull(current_dataframe)].reset_index(drop=True)
            elif "in" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[1]].str.contains(sp[0])].reset_index(drop=True)
            elif "obslist" in k:
                sids = pd.read_csv("observation_list.csv")["source_id"].to_list()
                current_dataframe = current_dataframe[current_dataframe["source_id"].isin(sids)]
            elif "nobhb" in k:
                current_dataframe = current_dataframe[~current_dataframe["bibcodes"].str.contains("2021A&A...654A.107C")].reset_index(drop=True)

        colname = sortset.get()

        if colname in ["gmag", "logp"]:
            current_dataframe = current_dataframe.sort_values(by=[colname]).reset_index(drop=True)
        else:
            current_dataframe = current_dataframe.sort_values(by=[colname], ascending=False).reset_index(drop=True)

        for col in current_dataframe.columns:
            if col in excluded_cols:
                current_dataframe = current_dataframe.drop(columns=[col])

        sheet.headers(current_dataframe.columns)
        sheet.set_sheet_data(current_dataframe.values.tolist(), redraw=True)

        highlight_cells(sheet, highlight.get())

    def add_to_observation_list(gaia_id=None):
        if not gaia_id:
            gaia_id = sheet.data[list(sheet.get_selected_cells())[0][0]][0]
        if not os.path.isfile("observation_list.csv"):
            obs_list = pd.DataFrame()
        else:
            obs_list = pd.read_csv("observation_list.csv")
        if gaia_id not in obs_list["source_id"].to_list():
            obs_list = pd.concat([obs_list, interesting_dataframe[interesting_dataframe["source_id"] == gaia_id]])
            obs_list.to_csv("observation_list.csv", index=False)
        highlight_cells(sheet, highlight.get())

    def add_to_list_wrapper(gaia_id):
        def inner():
            return add_to_observation_list(gaia_id)

        return inner

    def remove_from_observation_list(gaia_id=None):
        if not gaia_id:
            gaia_id = sheet.data[list(sheet.get_selected_cells())[0][0]][0]
        if not os.path.isfile("observation_list.csv"):
            obs_list = pd.DataFrame()
        else:
            obs_list = pd.read_csv("observation_list.csv")
        obs_list = obs_list[obs_list["source_id"] != gaia_id]
        obs_list.to_csv("observation_list.csv", index=False)
        highlight_cells(sheet, highlight.get())

    def remove_from_list_wrapper(gaia_id):
        def inner():
            return remove_from_observation_list(gaia_id)

        return inner

    def flagging(master, flags):
        flagframe = tk.Frame(master)
        if "OG" in flags:
            og = tk.Label(flagframe, text="This star was mentioned in old papers", fg="#8B0000")
            og.pack(side=tk.BOTTOM)
        if "HV" in flags:
            hv = tk.Label(flagframe, text="This star was mentioned in a HV paper", fg="#8B0000")
            hv.pack(side=tk.BOTTOM)
        if "RV" in flags:
            rv = tk.Label(flagframe, text="This star might have\nprevious RV measurements", fg="#8B0000")
            rv.pack(side=tk.BOTTOM)
        if "WD" in flags:
            wd = tk.Label(flagframe, text="This star might be a WD", fg="#8B0000")
            wd.pack(side=tk.BOTTOM)
        if "phot" in flags:
            phot = tk.Label(flagframe, text="This star might have\na photometrically found partner", fg="#8B0000")
            phot.pack(side=tk.BOTTOM)
        if "spec" in flags:
            spec = tk.Label(flagframe, text="This star might have\na spectroscopically found partner", fg="#8B0000")
            spec.pack(side=tk.BOTTOM)
        if "pulsation" in flags:
            puls = tk.Label(flagframe, text="This star might be a pulsator", fg="#5ced73")
            puls.pack(side=tk.BOTTOM)
        if "HV-detection" in flags:
            hvde = tk.Label(flagframe, text="This star has a high mean velocity", fg="#5ced73")
            hvde.pack(side=tk.BOTTOM)
        if "rvv-detection" in flags:
            rvde = tk.Label(flagframe, text="This star is RV variable", fg="#5ced73")
            rvde.pack(side=tk.BOTTOM)
        if "rvv-candidate" in flags:
            rvca = tk.Label(flagframe, text="This star might be RV variable", fg="#5ced73")
            rvca.pack(side=tk.BOTTOM)

        return flagframe

    def reload_system():
        nonlocal interesting_dataframe
        gaia_id = sheet.data[list(sheet.get_selected_cells())[0][0]][0]

        rvtable = pd.read_csv(f"output/{gaia_id}/RV_variation.csv")
        vels = rvtable["culum_fit_RV"].to_numpy()
        verrs = rvtable["u_culum_fit_RV"].to_numpy()
        times = rvtable["mjd"].to_numpy()

        logp = vrad_pvalue(vels, verrs)
        deltarv = np.ptp(vels)

        updatedict = {
            "logp": logp,
            "deltaRV": deltarv,
            "deltaRV_err": np.sqrt(verrs[np.argmax(vels)] ** 2 + verrs[np.argmin(vels)] ** 2) / 2,
            "RVavg": np.mean(vels),
            "RVavg_err": np.sqrt(np.sum(np.square(verrs))) / len(verrs),
        }

        interesting_dataframe.loc[interesting_dataframe["source_id"] == gaia_id, list(updatedict.keys())] = list(updatedict.values())
        interesting_dataframe = interesting_dataframe.sort_values(by="logp", ascending=False)
        interesting_dataframe.to_csv("interesting_params.csv", index=False)

        plot_rvcurve_brokenaxis(vels, verrs, times, gaia_id, custom_saveloc=f"output/{gaia_id}/RV_variation_broken_axis.pdf")

        queue.put(["execute_function", plot_system_from_ind, {"ind": str(gaia_id), "savepath": f"output/{gaia_id}/spoverview.pdf", "use_ind_as_sid": True, "custom_xlim": (4000, 4500)}])

        update_sheet()

    def view_detail_window():
        gaia_id = sheet.data[list(sheet.get_selected_cells())[0][0]][0]
        star = interesting_dataframe[interesting_dataframe["source_id"] == gaia_id].to_dict('records')[0]
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
        logp = star["logp"]
        pmra = star["pmra"]
        pmdec = star["pmdec"]
        parallax = star["parallax"]
        parallax_error = star["parallax_err"]
        spatial_vel = star["spatial_vels"]
        tic = star["tic"]
        tic = str(tic)
        if "." in tic:
            tic = tic.split(".")[0]
        elif tic.lower() == "nan":
            tic = None

        try:
            flags = ast.literal_eval(star["flags"])
        except KeyError:
            flags = []

        detail_window = tk.Toplevel(analysis)
        try:
            if os.name == "nt":
                detail_window.iconbitmap("favicon.ico")
            else:
                imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
                detail_window.tk.call('wm', 'iconphoto', detail_window._w, imgicon)
        except:
            pass
        detail_window.title(f"Detail View for Gaia DR3 {gaia_id}")
        detail_window.update_idletasks()  # This forces tkinter to update the window calculations.
        detail_window.geometry("800x600+0+0")
        if os.name == 'nt':
            detail_window.state('zoomed')
        elif os.name == "posix":
            detail_window.attributes('-zoomed', True)

        main_frame = tk.Frame(detail_window)
        main_frame.pack(fill="both", expand=1)
        width, height = get_monitor_from_coord(detail_window.winfo_x(), detail_window.winfo_y())
        imsize = (int(width // 2.25), int(height // 2.25))

        if not os.path.isfile(f"output/{gaia_id}/RV_variation_broken_axis.pdf"):
            if os.path.isfile(f"output/{gaia_id}/RV_variation.csv"):
                rvdata = np.loadtxt(f"output/{gaia_id}/RV_variation.csv", delimiter=",", skiprows=1)
                rvs = rvdata[:, 0]
                rv_errs = rvdata[:, 1]
                mjds = rvdata[:, 2]
            else:
                rvs = []
                rv_errs = []
                mjds = []
        else:
            rvs = []
            rv_errs = []
            mjds = []

        rvplot = load_or_create_image(main_frame,
                                      f"output/{gaia_id}/RV_variation_broken_axis.pdf",
                                      imsize,
                                      queue,
                                      plot_rvcurve_brokenaxis,
                                      False,
                                      rvs,
                                      rv_errs,
                                      mjds,
                                      gaia_id,)

        rvplot.grid(row=1, column=1, sticky="news")

        visplot = load_or_create_image(main_frame, f"output/{gaia_id}/visibility.pdf",
                                       imsize,
                                       queue,
                                       quick_visibility,
                                       ra=ra,
                                       dec=dec,
                                       date=general_config["FOR_DATE"],
                                       saveloc=f"output/{gaia_id}/visibility.pdf")
        visplot.grid(row=1, column=2, sticky="news")

        spoverview = load_or_create_image(main_frame,
                                          f"output/{gaia_id}/spoverview.pdf",
                                          imsize,
                                          queue,
                                          plot_system_from_ind,
                                          ind=str(gaia_id),
                                          savepath=f"output/{gaia_id}/spoverview.pdf",
                                          use_ind_as_sid=True,
                                          custom_xlim=(4000, 4500))
        spoverview.grid(row=2, column=2, sticky="news")

        subframe = tk.Frame(main_frame)

        params = [alias, sp_class, (ra, dec), gmag, nspec, (deltarv, u_deltarv), (rvavg, u_rvavg), logp, (pmra, pmdec), (parallax, parallax_error), spatial_vel]
        table = construct_table(subframe, params)
        table.grid(row=1, column=1)

        bibframe = tk.Frame(subframe)
        if star["bibcodes"] != "-" and isinstance(star["bibcodes"], str):
            bibcodes = ast.literal_eval(star["bibcodes"])
        else:
            bibcodes = []

        def make_link(b):
            return lambda c: callback(f"https://ui.adsabs.harvard.edu/abs/{b}")

        bibtitle = tk.Label(bibframe, text="Bibcodes for this object")
        bibtitle.grid(row=1, column=1)
        for i, b in enumerate(bibcodes):
            if i > 9:
                break
            bibtext = tk.Label(bibframe, text=b, fg="blue", cursor="hand2")
            bibtext.bind("<Button-1>", make_link(b))
            bibtext.grid(row=i + 2, column=1)

        dots = tk.Frame(bibframe)
        for i, d in enumerate(bibcodes[9:]):
            dot = tk.Label(dots, text=".", fg="blue", cursor="hand2")
            dot.bind("<Button-1>", make_link(d))
            dot.grid(row=i // 20 + 1, column=i % 20)
        dots.grid(row=12, column=1)

        bibframe.grid(row=1, column=2)

        flags = flagging(subframe, flags)
        flags.grid(row=1, column=3)

        subframe.grid(row=2, column=1, sticky="news")

        buttonframe = tk.Frame(main_frame)

        addfunc = add_to_list_wrapper(gaia_id)
        addobs = tk.Button(buttonframe, text="Add to observation list", command=addfunc, width=20)
        addobs.grid(row=1, column=1)

        rmfunc = remove_from_list_wrapper(gaia_id)
        rmobs = tk.Button(buttonframe, text="Remove from obs. list", command=rmfunc, width=20)
        rmobs.grid(row=2, column=1)

        def simbad_link():
            return lambda: callback(f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+{gaia_id}&submit=submit+id")

        simbad_btn = tk.Button(buttonframe, text="SIMBAD", command=simbad_link(), width=20)
        simbad_btn.grid(row=4, column=1)

        normalize = tk.BooleanVar(value=True)
        normplot = tk.Checkbutton(buttonframe, text="Normalize", variable=normalize, width=20)
        normplot.select()
        normplot.grid(row=6, column=1)

        view_plot = tk.Button(buttonframe, text="View Plot", command=lambda: viewplot(queue, normalize.get(), gaia_id), width=20)
        view_plot.grid(row=7, column=1)

        view_plot = tk.Button(buttonframe, text="View Master Plot", command=lambda: viewmasterplot(queue, normalize.get(), gaia_id), width=20)
        view_plot.grid(row=8, column=1)

        fit_sed = tk.Button(buttonframe, text="Fit SED", command=lambda: fitsed(analysis, gaia_id), width=20)
        fit_sed.grid(row=10, column=1)

        fit_model = tk.Button(buttonframe, text="Fit Model", command=lambda: fitmodel(analysis, gaia_id), width=20)
        fit_model.grid(row=11, column=1)

        fit_model = tk.Button(buttonframe, text="Calculate Galactic\nOrbit", command=lambda: galacticorbit_wrapper(interesting_dataframe, detail_window, gaia_id), width=20)
        fit_model.grid(row=12, column=1)
        if np.isnan(rvavg) or np.isnan(parallax):
            fit_model["state"] = "disabled"

        view_lc = tk.Button(buttonframe, text="View Lightcurve", command=lambda: viewlc(analysis, gaia_id, tic, queue), width=20)
        view_lc.grid(row=14, column=1)

        view_lc = tk.Button(buttonframe, text="Phasefold", command=lambda: phasefold(analysis, gaia_id), width=20)
        view_lc.grid(row=15, column=1)

        view_cmd = tk.Button(buttonframe, text="View CMD", command=lambda: queue.put(["execute_function", show_CMD_window, (gaia_id,)]), width=20)
        view_cmd.grid(row=17, column=1)

        view_note = tk.Button(buttonframe, text="View Note", command=lambda: viewnote(analysis, gaia_id), width=20)
        view_note.grid(row=18, column=1)

        for i in range(16):
            buttonframe.grid_rowconfigure(i + 1, minsize=25)
        buttonframe.grid(row=1, column=3, sticky="news", rowspan=2, padx=10, pady=10)

    sheet.enable_bindings()
    sheet.headers(newheaders=interesting_dataframe.columns.tolist())
    sheet.disable_bindings("cut", "paste", "delete", "edit_cell", "edit_header", "edit_index")
    sheet.popup_menu_add_command("Add to observation list", add_to_observation_list)
    sheet.popup_menu_add_command("Remove from observation list", remove_from_observation_list)
    sheet.popup_menu_add_command("View detail window", view_detail_window)
    sheet.popup_menu_add_command("Create master spectrum", lambda: master_spectrum_window(analysis, queue, None, sheet=sheet))
    sheet.popup_menu_add_command("Reload System", reload_system)

    if "known_category" in interesting_dataframe.columns.tolist():
        k = tk.Checkbutton(tablesettings_frame, text="Show known", variable=show_known, command=update_sheet)
        k.grid(row=1, column=1)
        u = tk.Checkbutton(tablesettings_frame, text="Show unknown", variable=show_unknown, command=update_sheet)
        u.grid(row=1, column=2)
        lk = tk.Checkbutton(tablesettings_frame, text="Show likely known", variable=show_likely_known, command=update_sheet)
        lk.grid(row=1, column=3)
        nd = tk.Checkbutton(tablesettings_frame, text="Show indeterminate", variable=show_indeterminate, command=update_sheet)
        nd.grid(row=1, column=4)
    hg = tk.Checkbutton(tablesettings_frame, text="Highlight values", variable=highlight, command=update_sheet)
    hg.grid(row=1, column=5)
    excol = tk.Checkbutton(tablesettings_frame, text="Exclude Columns", variable=exclude_columns, command=update_sheet)
    excol.grid(row=1, column=6)
    droplabel = tk.Label(tablesettings_frame, text="Sort by: ")
    droplabel.grid(row=1, column=7)
    drop = tk.OptionMenu(tablesettings_frame, sortset, *interesting_dataframe.columns, command=update_sheet)
    drop.grid(row=1, column=8)
    filter_kw_label = tk.Label(tablesettings_frame, text="Filters: ")
    filter_kw_label.grid(row=1, column=9)
    filter_kw = tk.Entry(tablesettings_frame, textvariable=filter_kws)
    filter_kw.grid(row=1, column=10)
    filter_kw_btn = tk.Button(tablesettings_frame, text="Filter", command=update_sheet)
    filter_kw_btn.grid(row=1, column=11)

    tablesettings_frame.pack()
    frame.pack(fill="both", expand=1)
    sheet.pack(fill="both", expand=1)

    update_sheet()


def get_raw_files():
    outlist = []
    for f in os.listdir("spectra_raw"):
        if ".gitkeep" in f:
            continue
        else:
            outlist.append(f)

    return outlist


def preprocess(prep_tab, queue):
    prep_settings = {}
    prep_frame = tk.Frame(prep_tab)

    input_container = tk.Frame(prep_frame)

    filelist = get_raw_files()
    input_label = tk.Label(input_container,
                           text="Raw spectra",
                           font="SegoeUI 20")
    input_label.pack()
    file_handling_choices = tk.Frame(input_container)

    fileendings = []
    for f in filelist:
        fend = f.split(".")[-1]

        if fend not in fileendings:
            fileendings.append(fend)

    fenddict = {}
    iend = 0
    for i, fend in enumerate(fileendings):
        fendlabel = tk.Label(file_handling_choices, text=f".{fend}:")
        fendlabel.grid(row=1, column=i * 2)
        fend_ch = tk.StringVar(file_handling_choices, value="Generic ASCII")
        fend_drop = tk.OptionMenu(file_handling_choices,
                                  fend_ch,
                                  *["Generic ASCII", ])
        # "Generic FITS",
        #  "LAMOST Low Resolution",
        #  "LAMOST Medium Resolution"])
        fenddict[f"{fend}"] = fend_ch
        fend_drop.grid(row=1, column=i * 2 + 1)
        iend = i

    drpdwn_val = tk.StringVar(value="h,deg")
    drpdwn_label = tk.Label(file_handling_choices, text=f"RA/DEC units used:")
    drpdwn_label.grid(row=1, column=iend * 2 + 2)
    drpdwn = tk.OptionMenu(
        file_handling_choices,
        drpdwn_val,
        *["h,deg",
          "deg,deg"]
    )
    drpdwn.grid(row=1, column=iend * 2 + 3)

    file_handling_choices.pack()

    if len(filelist) == 0:
        input_label = tk.Label(input_container,
                               text="Please add your raw spectra files to\n./spectra_raw\nand restart this program",
                               fg="blue",
                               font="SegoeUI 12 italic")
        input_label.pack()
    else:
        input_sheet = Sheet(input_container,
                            data=[[f] for f in filelist],
                            show_top_left=False,
                            show_row_index=False,
                            show_header=False,
                            show_x_scrollbar=False,
                            show_y_scrollbar=True,
                            width=800,
                            height=800,
                            column_width=800
                            )
        input_sheet.enable_bindings()
        input_sheet.pack(fill="both", expand=1)
    input_container.grid(row=1, column=1, sticky='NEWS')

    intermediate_container = tk.Frame(prep_frame)
    subcontainer = tk.Frame(intermediate_container)
    process_arrow = ImageTk.PhotoImage(Image.open("Arrow.png"))
    parrlabel = tk.Label(subcontainer, image=process_arrow)
    parrlabel.image = process_arrow
    parrlabel.pack(padx=10, pady=10)

    final_container = tk.Frame(prep_frame)

    if os.path.isfile("object_catalogue.csv") == 0:
        output_label = tk.Label(final_container,
                                text="Pre-Processed spectra",
                                font="SegoeUI 20")
        output_label.pack(side=tk.TOP)
        output_descr = tk.Label(final_container,
                                text="Preprocessing has not been completed yet.",
                                fg="blue",
                                font="SegoeUI 12 italic")
        output_descr.pack(side=tk.TOP)
        output_sheet = Sheet(final_container,
                             data=[[]],
                             headers=[],
                             show_top_left=False,
                             show_row_index=False,
                             width=800,
                             height=800)
        output_sheet.enable_bindings()
    else:
        obj_cat = pd.read_csv("object_catalogue.csv")
        output_label = tk.Label(final_container,
                                text="Pre-Processed spectra",
                                font="SegoeUI 20")
        output_descr = tk.Label(final_container,
                                text="Preprocessing has not been completed yet.",
                                fg="blue",
                                font="SegoeUI 12 italic")
        output_label.pack(side=tk.TOP)
        output_sheet = Sheet(final_container,
                             data=obj_cat.values.tolist(),
                             headers=obj_cat.columns.tolist(),
                             show_top_left=False,
                             show_row_index=False,
                             width=800,
                             height=800)
        output_sheet.enable_bindings()
        output_sheet.pack(fill="both", expand=1)

    final_container.grid(row=1, column=3, sticky='NEWS')

    def prep_wrapper():
        if len(filelist) == 0:
            messagebox.showwarning("No raw files found", "No raw files found! Make sure that you placed your files in /spectra_raw and restarted this program.")
            return
        prep_settings["coordunit"] = drpdwn_val.get()
        preprocessing([(k, v.get()) for k, v in fenddict.items()], prep_settings)
        nonlocal output_descr
        if os.path.isfile("object_catalogue.csv") == 0:
            output_label = tk.Label(final_container,
                                    text="Pre-Processed spectra",
                                    font="SegoeUI 20")
            output_label.pack(side=tk.TOP)
            output_descr = tk.Label(final_container,
                                    text="Preprocessing has not been completed yet.",
                                    fg="blue",
                                    font="SegoeUI 12 italic")
            output_descr.pack(side=tk.TOP)
        else:
            obj_cat = pd.read_csv("object_catalogue.csv")
            output_sheet.set_sheet_data(obj_cat.values.tolist())
            output_sheet.headers(obj_cat.columns.tolist())
            output_sheet.pack(fill="both", expand=1)
            output_descr.pack_forget()

    process_btn = tk.Button(subcontainer, text="Preprocess", width=10, command=prep_wrapper)
    process_btn.pack()
    subcontainer.pack(side=tk.LEFT)
    intermediate_container.grid(row=1, column=2, sticky='NS')

    prep_frame.pack(fill="both", expand=1, padx=50, pady=50)


def download_catalogues():
    vizier = Vizier(columns=["**"], row_limit=999999)  # "*" retrieves all columns
    catalog_id = "J/A+A/662/A40/knownhsd"
    knownsd = vizier.get_catalogs(catalog_id)
    knownsd = knownsd[catalog_id]
    knownsd = knownsd.to_pandas()
    knownsd = knownsd.rename(columns=dict(zip(
        [
            "Name", "GaiaEDR3", "RA_ICRS", "DE_ICRS", "GLON", "GLAT", "SpClass", "SpClassS", "CSDSS", "CAPASS", "CPS1", "CSKYM", "Plx", "PlxZP", "e_Plx", "GMAG", "GGAIA", "e_GGAIA", "BPGAIA", "e_BPGAIA", "RPGAIA", "e_RPGAIA", "pmRAGAIA", "e_pmRAGAIA",
            "pmDEGAIA", "e_pmDEGAIA", "RVSDSS", "e_RVSDSS", "RVLAMOST", "e_RVLAMOST", "Teff", "e_Teff", "logg", "e_logg", "logY", "e_logY", "Ref", "E_B-V_", "e_E_B-V_", "AV", "FUVGALEX", "e_FUVGALEX", "NUVGALEX", "e_NUVGALEX", "VAPASS", "e_VAPASS",
            "BAPASS", "e_BAPASS", "gAPASS", "e_gAPASS", "rAPASS", "e_rAPASS", "iAPASS", "e_iAPASS", "uSDSS", "e_uSDSS", "gSDSS", "e_gSDSS", "rSDSS", "e_rSDSS", "iSDSS", "e_iSDSS", "zSDSS", "e_zSDSS", "uVST", "e_uVST", "gVST", "e_gVST", "rVST", "e_rVST",
            "iVST", "e_iVST", "zVST", "e_zVST", "uSKYM", "e_uSKYM", "vSKYM", "e_vSKYM", "gSKYM", "e_gSKYM", "rSKYM", "e_rSKYM", "iSKYM", "e_iSKYM", "zSKYM", "e_zSKYM", "gPS1", "e_gPS1", "rPS1", "e_rPS1", "iPS1", "e_iPS1", "zPS1", "e_zPS1", "yPS1", "e_yPS1",
            "J2MASS", "e_J2MASS", "H2MASS", "e_H2MASS", "K2MASS", "e_K2MASS", "YUKIDSS", "e_YUKIDSS", "JUKIDSS", "e_JUKIDSS", "HUKIDSS", "e_HUKIDSS", "KUKIDSS", "e_KUKIDSS", "ZVISTA", "e_ZVISTA", "YVISTA", "e_YVISTA", "JVISTA", "e_JVISTA", "HVISTA",
            "e_HVISTA", "KsVISTA", "e_KsVISTA", "W1", "e_W1", "W2", "e_W2", "W3", "e_W3", "W4", "e_W4"
        ],
        [
            "NAME", "GAIA_DESIG", "RA", "DEC", "GLON", "GLAT", "SPEC_CLASS", "SPEC_SIMBAD", "COLOUR_SDSS", "COLOUR_APASS", "COLOUR_PS1", "COLOUR_SKYM", "PLX", "PLX_ZP", "e_PLX", "M_G", "G_GAIA", "e_G_GAIA", "BP_GAIA", "e_BP_GAIA", "RP_GAIA", "e_RP_GAIA",
            "PMRA_GAIA", "e_PMRA_GAIA", "PMDEC_GAIA", "e_PMDEC_GAIA",
            "RV_SDSS", "e_RV_SDSS", "RV_LAMOST", "e_RV_LAMOST", "TEFF", "e_TEFF", "LOG_G", "e_LOG_G", "LOG_Y", "e_LOG_Y", "PARAMS_REF", "EB-V", "e_EB-V", "AV", "FUV_GALEX", "e_FUV_GALEX", "NUV_GALEX", "e_NUV_GALEX", "V_APASS", "e_V_APASS", "B_APASS",
            "e_B_APASS", "g_APASS", "e_g_APASS", "r_APASS",
            "e_r_APASS", "i_APASS", "e_i_APASS", "u_SDSS", "e_u_SDSS", "g_SDSS", "e_g_SDSS", "r_SDSS", "e_r_SDSS", "I_SDSS", "e_i_SDSS", "z_SDSS", "e_z_SDSS", "u_VST", "e_u_VST", "g_VST", "e_g_VST", "r_VST", "e_r_VST", "I_VST", "e_i_VST", "z_VST", "e_z_VST",
            "u_SKYM", "e_u_SKYM", "v_SKYM", "e_v_SKYM", "g_SKYM",
            "e_g_SKYM", "r_SKYM", "e_r_SKYM", "i_SKYM", "e_i_SKYM", "z_SKYM", "e_z_SKYM", "g_PS1", "e_g_PS1", "r_PS1", "e_r_PS1", "i_PS1", "e_i_PS1", "z_PS1", "e_z_PS1", "y_PS1", "e_y_PS1", "J_2MASS", "e_J_2MASS", "H_2MASS", "e_H_2MASS", "K_2MASS",
            "e_K_2MASS", "Y_UKIDSS", "e_Y_UKIDSS", "J_UKIDSS", "e_J_UKIDSS",
            "H_UKIDSS", "e_H_UKIDSS", "K_UKIDSS", "e_K_UKIDSS", "Z_VISTA", "e_Z_VISTA", "Y_VISTA", "e_Y_VISTA", "J_VISTA", "e_J_VISTA", "H_VISTA", "e_H_VISTA", "Ks_VISTA", "e_Ks_VISTA", "W1", "e_W1", "W2", "e_W2", "W3", "e_W3", "W4", "e_W4"
        ]
    )))
    knownsd["GAIA_DESIG"] = np.array(["GAIA EDR3 " + str(t) for t in knownsd["GAIA_DESIG"].to_list()])
    knownsd = knownsd.drop(columns=["recno", "_RA.icrs", "_DE.icrs"])
    knownsd.to_csv("catalogues/sd_catalogue_v56_pub.csv", index=False)

    vizier = Vizier(columns=["**"],
                    row_limit=999999)  # "*" retrieves all columns
    catalog_id = "J/A+A/662/A40/hotsd"
    candsd = vizier.get_catalogs(catalog_id)
    candsd = candsd[catalog_id]
    pandastb = candsd.to_pandas()
    pandastb = pandastb.drop(columns=["recno", "_RA.icrs", "_DE.icrs"])
    pandastb = pandastb.rename(columns=dict(zip(['GaiaEDR3', 'RA_ICRS', 'DE_ICRS', 'GLON', 'GLAT', 'Plx', 'e_Plx', 'GMAG', 'Gmag', 'BP-RP', 'E_BP_RP_c', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'pm', 'e_pm', 'RUWE', 'Rpm', 'e_EFlux', 'f_Plx', 'f_pm'],
                                                ["source_id", "ra", "dec", "l", "b", "parallax", "parallax_error", "abs_g_mag", "phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor_corrected", "pmra", "pmra_error", "pmdec", "pmdec_error", "pm", "ruwe",
                                                 "pm_error",
                                                 "reduced_proper_motion", "excess_flux_error", "parallax_selection_flag", "proper_motion_selection_flag"])))
    hdu = fits.BinTableHDU(Table.from_pandas(pandastb))
    hdu.writeto("catalogues/hotSD_gaia_edr3_catalogue.fits")

    messagebox.showinfo("Download Complete", "Download of catalogues is complete.")


def gui_window(queue, p_queue):
    if os.name == 'nt':
        ctypes.windll.shcore.SetProcessDpiAwareness(0)
    # Initialize the main window
    window = tk.Tk()
    window.title("RVVD")
    try:
        if os.name == "nt":
            window.iconbitmap("favicon.ico")
        else:
            imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
            window.tk.call('wm', 'iconphoto', window._w, imgicon)
    except Exception as e:
        print(e)
    window.geometry("800x600+0+0")
    if os.name == 'nt':
        window.state('zoomed')
    elif os.name == "posix":
        window.attributes('-zoomed', True)

    # Create the notebook
    main_tabs = ttk.Notebook(window)

    prep = tk.Frame(main_tabs)
    main_tabs.add(prep, text="Pre-Processing")
    processing = tk.Frame(main_tabs)
    main_tabs.add(processing, text="Processing")
    analysis = tk.Frame(main_tabs)
    main_tabs.add(analysis, text="Analysis")
    main_tabs.pack(expand=1, fill='both')

    # Create a progress bar to track the overall progress of the task
    overall = tk.DoubleVar()
    overall_progress = ttk.Progressbar(processing, variable=overall, orient="horizontal", length=1000, mode="determinate")
    overall_progress.pack(pady=2)
    suplabel = tk.Label(processing, text="Overall progress")
    suplabel.pack()

    line = tk.Frame(processing, height=2, bd=1, relief="sunken")
    line.pack(fill="x", padx=5, pady=5)

    # Create a frame to hold the individual progress bars
    frame = tk.Frame(processing)
    frame.pack()

    analysis_tab(analysis, queue)

    preprocess(prep, queue)

    # Calculate the number of rows needed to display the progress bars in a grid with two columns
    num_cores = cpu_count()

    # Create a progress bar for each subprocess
    subprocess_progress = [overall]
    ls = [suplabel]
    for i in range(num_cores):
        # Create a label for the progress bar
        label = tk.Label(frame, text="Subprocess {}".format(i + 1))
        label.grid(row=i // 2 * 2 + 1, column=i % 2, padx=10, pady=1)

        progress_var = tk.DoubleVar()

        # Create the progress bar
        progress = ttk.Progressbar(frame, variable=progress_var, orient="horizontal", length=400, mode="determinate")
        progress.grid(row=i // 2 * 2, column=i % 2, padx=10, pady=1)
        subprocess_progress.append(progress_var)
        ls.append(label)

    # Create the start button
    buttons_frame = tk.Frame()
    settings_button = tk.Button(buttons_frame, font=('Segoe UI', 12), text="Adjust Settings", command=lambda: open_settings(window, queue), height=1, width=15)
    settings_button.pack(side=tk.LEFT)
    start_button = tk.Button(buttons_frame, font=('Segoe UI', 12), text="Start", command=lambda: start_proc(queue), height=1, width=15, bg="#90EE90")
    start_button.pack(side=tk.LEFT)
    buttons_frame.place(in_=processing, anchor="se", relx=0.95, rely=0.95)

    if "sd_catalogue_v56_pub.csv" not in os.listdir("catalogues") or "hotSD_gaia_edr3_catalogue.fits" not in os.listdir("catalogues"):
        result = messagebox.askokcancel("Catalogue Download Required",
                                        "RVVD requires the Culpan(2022) catalogue of hot subdwarf stars to be downloaded in order to function correctly, do you wish to download the catalogue now?",
                                        icon="question")

        if result:
            download_catalogues()

    def update_progress(bars, labels):
        try:
            upd = p_queue.get(block=False)
        except Empty:
            window.update()
            window.after(10, lambda: update_progress(bars, labels))
            return
        if upd[2] == "progressbar":
            if upd[1] == 0:
                bars[upd[0] - 1].set(0)
            else:
                val = bars[upd[0] - 1].get() + upd[1]
                if val > 100:
                    val -= 100
                bars[upd[0] - 1].set(val)
        elif upd[2] == "text":
            labels[upd[0] - 1].configure(text=f"{upd[1]} - {round(bars[upd[0] - 1].get(), 2)}%")
        elif upd[0] == "done":
            for widget in analysis.winfo_children():
                widget.destroy()
            analysis_tab(analysis, queue)
        window.update()
        window.after(10, lambda: update_progress(bars, labels))

    def on_closing():
        if os.path.isdir("./mastDownload/TESS"):
            shutil.rmtree("./mastDownload/TESS")
        for prc in active_children():
            prc.terminate()
        os._exit(0)

    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the main loop to display the window
    update_progress(subprocess_progress, ls)
    window.mainloop()


if __name__ == "__main__":
    man = Manager()
    queue = man.Queue()
    progress_queue = man.Queue()
    gui_main = threading.Thread(None, gui_window, args=[queue, progress_queue])
    gui_main.start()

    while True:
        item = queue.get()
        if item[0] == "update_configs":
            configs = item[1]
        elif item[0] == "start_process":
            proc = threading.Thread(target=interactive_main, args=[configs, progress_queue])
            proc.start()
        elif item[0] == "execute_function":
            try:
                if isinstance(item[2], dict):
                    item[1](**item[2])
                elif len(item) == 3:
                    item[1](*item[2])
                else:
                    item[1](*item[2], **item[3])
            except Exception as e:
                print("Exception encountered!:", e)
                print(traceback.format_exc())
        elif item[0] == "execute_function_with_return":
            returnq = item[2][0]
            item[2] = item[2][1:]
            try:
                if isinstance(item[2], dict):
                    item[1](**item[2])
                elif len(item) == 3:
                    item[1](*item[2])
                else:
                    item[1](*item[2], **item[3])
            except Exception as e:
                print("Exception encountered:", e)
                print(traceback.format_exc())
            finally:
                returnq.put(True)
        queue.task_done()
