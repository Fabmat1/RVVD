import ast
import multiprocessing
import os
import time
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fitz
import webbrowser


def callback(url):
    webbrowser.open_new(url)


from PIL import Image, ImageTk

from get_interesting_stars import quick_visibility
from main import general_config, fit_config, plot_config, interactive_main
import threading
from queue import Empty
from tksheet import Sheet

from plot_spectra import plot_system_from_ind


# from terminedia import ColorGradient


def close_window():
    os._exit(1)


# red_green_gradient = ColorGradient([(0, (255, 65, 34)), (0.5, (255, 255, 255)), (1, (92, 237, 115))])


class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def open_settings(window, queue):
    def save_settings():
        for cat in [mod_gc, mod_fc, mod_pc]:
            for key, val in cat.items():
                cat[key] = val.get()

        queue.put(["update_configs", [mod_gc, mod_fc, mod_pc]])
        settings_window.destroy()

    # Open the settings dialogue
    settings_window = tk.Toplevel(window)
    settings_window.iconbitmap("favicon.ico")
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

    for cat, tab, mod_cat in zip([general_config, fit_config, plot_config], [general, fitting, plotting], [mod_gc, mod_fc, mod_pc]):
        for key, val in cat.items():
            if isinstance(val, str):
                tk_stringval = tk.StringVar(value=val)
                strval = tk.Entry(tab, textvariable=tk_stringval)
                l = tk.Label(tab, text=key)
                CreateToolTip(strval, "")
                l.pack()
                strval.pack()
                mod_cat[key] = tk_stringval
            elif isinstance(val, bool):
                tk_bool = tk.BooleanVar()
                boolval = tk.Checkbutton(tab, text=key, variable=tk_bool)
                if val:
                    boolval.select()
                CreateToolTip(boolval, "")
                boolval.pack()
                mod_cat[key] = tk_bool
            elif isinstance(val, float) or isinstance(val, int):
                tk_doub = tk.DoubleVar()
                doubval = tk.Entry(tab, textvariable=tk_doub)
                l = tk.Label(tab, text=key)
                l.pack()
                CreateToolTip(doubval, "")
                doubval.pack()
                tk_doub.set(val)
                mod_cat[key] = tk_doub

    save_button = tk.Button(settings_window, text="Save Settings", command=save_settings)
    save_button.pack()


def start_proc(queue):
    queue.put(["start_process"])


def cell_colors(sheet, row_indices_full, row_indices_part, col_ind, colors):
    detectioncell_positions = [(r, col_ind) for r in row_indices_full]
    candidatecell_positions = [(r, col_ind) for r in row_indices_part]

    if len(detectioncell_positions) > 0:
        sheet.highlight_cells(cells=detectioncell_positions, bg=colors[0])
    if len(candidatecell_positions) > 0:
        sheet.highlight_cells(cells=candidatecell_positions, bg=colors[1])


def load_or_create_image(window, file_name, size, creation_func=None, *args, **kwargs):
    if not os.path.isfile(file_name):
        creation_func(*args, **kwargs)
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
    labels = ["Alias", "Spectral Class", "RA/DEC", "G mag", "N spec", "Delta RV", "RV avg", "log p", "Interestingness", ]

    tframe = tk.Frame(master)
    for i, p in enumerate(params):
        label = tk.Label(tframe, text=labels[i])
        if isinstance(params[i], tuple):
            if labels[i] == "RA/DEC":
                value = tk.Label(tframe, text=f"{round(p[0], 4)}/{round(p[1], 4)}")
            else:
                value = tk.Label(tframe, text=f"{round(p[0], 2)}±{round(p[1], 2)}")
        else:
            if isinstance(p, float):
                value = tk.Label(tframe, text=f"{round(p, 2)}")
            else:
                value = tk.Label(tframe, text=f"{p}")

        label.grid(row=i + 1, column=1)
        value.grid(row=i + 1, column=2)

    return tframe


def analysis_tab(analysis):
    frame = tk.Frame(analysis)
    interesting_dataframe = pd.read_csv("interesting_params.csv")

    tablesettings_frame = tk.Frame(analysis)

    sheet = Sheet(frame,
                  data=interesting_dataframe.values.tolist())

    # global show_known
    # global show_unknown
    # global show_likely_known
    # global show_indeterminate
    # global highlight
    # global current_dataframe
    # global sortset
    current_dataframe = interesting_dataframe

    show_known = tk.IntVar(frame, value=1)
    show_unknown = tk.IntVar(frame, value=1)
    show_likely_known = tk.IntVar(frame, value=1)
    show_indeterminate = tk.IntVar(frame, value=1)
    highlight = tk.IntVar(frame, value=1)
    filter_kws = tk.StringVar(frame, value="")
    sortset = tk.StringVar(frame, value="interestingness")

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

            sids = pd.read_csv("observation_list.csv")["source_id"].to_list()
            cell_colors(sheet,
                        current_dataframe["source_id"][current_dataframe["source_id"].isin(sids)].index.tolist(),
                        current_dataframe["gmag"][np.logical_and(current_dataframe["gmag"] < 19, current_dataframe["gmag"] > 19)].index.tolist(),
                        columns.index("source_id"),
                        ("#5ced73", "#abf7b1"))

    highlight_cells(sheet, highlight.get())

    def prep_kstring(kstring):
        kstring = kstring.replace("in", "=")
        kstring = kstring.replace("<", "")
        kstring = kstring.replace(">", "")
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
        print(sp)
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

        current_dataframe = interesting_dataframe[interesting_dataframe["known_category"].isin(filters)].reset_index(drop=True)

        keywords = filter_kws.get().split(";")
        for k in keywords:
            if "=" in k and "<=" not in k and ">=" not in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[0]] == sp[1]].reset_index(drop=True)
            elif "<=" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[0]] <= sp[1]].reset_index(drop=True)
            elif ">=" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[0]] >= sp[1]].reset_index(drop=True)
            elif "in" in k:
                sp = prep_kstring(k)
                current_dataframe = current_dataframe[current_dataframe[sp[1]].str.contains(sp[0])].reset_index(drop=True)

        colname = sortset.get()

        if colname in ["gmag", "logp"]:
            current_dataframe = current_dataframe.sort_values(by=[colname]).reset_index(drop=True)
        else:
            current_dataframe = current_dataframe.sort_values(by=[colname], ascending=False).reset_index(drop=True)

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
        interestingness = star["interestingness"]
        flags = ast.literal_eval(star["flags"])

        detail_window = tk.Toplevel(analysis)
        detail_window.iconbitmap("favicon.ico")
        detail_window.title(f"Detail View for {gaia_id}")
        detail_window.geometry("800x600+0+0")
        detail_window.state('zoomed')

        main_frame = tk.Frame(detail_window)
        main_frame.pack(fill="both", expand=1)

        imsize = (int(detail_window.winfo_screenwidth() // 2.25), int(detail_window.winfo_screenheight() // 2.25))

        rvplot = load_or_create_image(main_frame,
                                      f"output/{gaia_id}/RV_variation_broken_axis.pdf",
                                      imsize)
        rvplot.grid(row=1, column=1, sticky="news")

        visplot = load_or_create_image(main_frame, f"output/{gaia_id}/visibility.pdf",
                                       imsize,
                                       quick_visibility, saveloc=f"output/{gaia_id}/visibility.pdf")
        visplot.grid(row=1, column=2, sticky="news")

        spoverview = load_or_create_image(main_frame,
                                          f"output/{gaia_id}/spoverview.pdf",
                                          imsize,
                                          plot_system_from_ind,
                                          ind=str(gaia_id),
                                          savepath=f"output/{gaia_id}/spoverview.pdf",
                                          use_ind_as_sid=True,
                                          custom_xlim=(4000, 4500))
        spoverview.grid(row=2, column=2, sticky="news")

        subframe = tk.Frame(main_frame)

        params = [alias, sp_class, (ra, dec), gmag, nspec, (deltarv, u_deltarv), (rvavg, u_rvavg), logp, interestingness]
        table = construct_table(subframe, params)
        table.grid(row=1, column=1)

        bibframe = tk.Frame(subframe)
        bibcodes = ast.literal_eval(star["bibcodes"])

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
        addobs = tk.Button(buttonframe, text="Add to observation list", command=addfunc)
        addobs.grid(row=1, column=1)

        rmfunc = remove_from_list_wrapper(gaia_id)
        rmobs = tk.Button(buttonframe, text="Remove from obs. list", command=rmfunc)
        rmobs.grid(row=2, column=1)

        def simbad_link():
            return lambda: callback(f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+{gaia_id}&submit=submit+id")

        simbad_btn = tk.Button(buttonframe, text="SIMBAD", command=simbad_link())
        simbad_btn.grid(row=3, column=1)

        def viewplot():
            return lambda: plot_system_from_ind(
                ind=str(gaia_id),
                use_ind_as_sid=True)

        viewplot = tk.Button(buttonframe, text="View Plot", command=viewplot())
        viewplot.grid(row=4, column=1)

        buttonframe.grid(row=1, column=3, sticky="news")

    sheet.enable_bindings()
    sheet.headers(newheaders=interesting_dataframe.columns.tolist())
    sheet.popup_menu_add_command("Add to observation list", add_to_observation_list)
    sheet.popup_menu_add_command("Remove from observation list", remove_from_observation_list)
    sheet.popup_menu_add_command("View detail window", view_detail_window)

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
    droplabel = tk.Label(tablesettings_frame, text="Sort by: ")
    droplabel.grid(row=1, column=6)
    drop = tk.OptionMenu(tablesettings_frame, sortset, *interesting_dataframe.columns, command=update_sheet)
    drop.grid(row=1, column=7)
    filter_kw_label = tk.Label(tablesettings_frame, text="Filters: ")
    filter_kw_label.grid(row=1, column=8)
    filter_kw = tk.Entry(tablesettings_frame, textvariable=filter_kws)
    filter_kw.grid(row=1, column=9)
    filter_kw_btn = tk.Button(tablesettings_frame, text="Filter", command=update_sheet)
    filter_kw_btn.grid(row=1, column=10)

    tablesettings_frame.pack()
    frame.pack(fill="both", expand=1)
    sheet.pack(fill="both", expand=1)


def gui_window(queue, p_queue):
    def update_progress(bars, labels):
        try:
            upd = p_queue.get(block=False)
        except Empty:
            upd = None
        if upd is not None:
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
        window.update()
        window.after(0, lambda: update_progress(bars, labels))

    # Initialize the main window
    window = tk.Tk()
    window.title("RVVD")
    window.iconbitmap("favicon.ico")
    window.geometry("800x600+0+0")
    window.state('zoomed')

    # Create the menu bar
    menu_bar = tk.Menu(window)

    # Create the File menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Settings", command=lambda: open_settings(window, queue))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=close_window)
    menu_bar.add_cascade(label="Menu", menu=file_menu)

    # Add the menu bar to the window
    window.config(menu=menu_bar)

    # Create the notebook
    main_tabs = ttk.Notebook(window)

    # Create the first tab
    processing = tk.Frame(main_tabs)
    main_tabs.add(processing, text="Processing")
    analysis = tk.Frame(main_tabs)
    main_tabs.add(analysis, text="Analysis")
    main_tabs.pack(expand=1, fill='both')

    # Create a progress bar to track the overall progress of the task
    overall = tk.DoubleVar()
    overall_progress = ttk.Progressbar(processing, variable=overall, orient="horizontal", length=500, mode="determinate")
    overall_progress.pack(pady=2)
    suplabel = tk.Label(processing, text="Overall progress")
    suplabel.pack()

    line = tk.Frame(processing, height=2, bd=1, relief="sunken")
    line.pack(fill="x", padx=5, pady=5)

    # Create a frame to hold the individual progress bars
    frame = tk.Frame(processing)
    frame.pack()

    analysis_tab(analysis)

    # Calculate the number of rows needed to display the progress bars in a grid with two columns
    num_cores = multiprocessing.cpu_count()

    # Create a progress bar for each subprocess
    subprocess_progress = [overall]
    ls = [suplabel]
    for i in range(num_cores):
        # Create a label for the progress bar
        label = tk.Label(frame, text="Subprocess {}".format(i + 1))
        label.grid(row=i // 2 * 2 + 1, column=i % 2, padx=10, pady=1)

        progress_var = tk.DoubleVar()

        # Create the progress bar
        progress = ttk.Progressbar(frame, variable=progress_var, orient="horizontal", length=200, mode="determinate")
        progress.grid(row=i // 2 * 2, column=i % 2, padx=10, pady=1)
        subprocess_progress.append(progress_var)
        ls.append(label)

    # Create the start button
    start_button = tk.Button(processing, text="Start", command=lambda: start_proc(queue))
    start_button.pack(side="right", padx=50, pady=10, ipadx=50)

    # Start the main loop to display the window
    update_progress(subprocess_progress, ls)
    window.mainloop()


if __name__ == "__main__":
    man = multiprocessing.Manager()
    configs = [general_config, fit_config, plot_config]
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
