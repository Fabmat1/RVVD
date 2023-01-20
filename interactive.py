import multiprocessing
import os
import time
import tkinter as tk
from tkinter import ttk
from main import general_config, fit_config, plot_config, interactive_main
import threading
from queue import Empty


def close_window():
    os._exit(1)


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


def gui_window(queue, p_queue):
    def update_progress(bars, labels):
        try:
            upd = p_queue.get(block=False)
        except Empty:
            upd = None
        if upd is not None:
            if upd[2] == "progressbar":
                if upd[1] == 0:
                    bars[upd[0]-1].set(0)
                else:
                    val = bars[upd[0] - 1].get() + upd[1]
                    if val > 100:
                        val -= 100
                    bars[upd[0]-1].set(val)
            elif upd[2] == "text":
                labels[upd[0]-1].configure(text=f"{upd[1]} - {round(bars[upd[0]-1].get(), 2)}%")
        window.update()
        window.after(0, lambda: update_progress(bars, labels))

    # Initialize the main window
    window = tk.Tk()
    window.title("RVVD")
    window.iconbitmap("favicon.ico")
    window.geometry("800x600+0+0")

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

    # Create a progress bar to track the overall progress of the task
    overall = tk.DoubleVar()
    overall_progress = ttk.Progressbar(window, variable=overall, orient="horizontal", length=500, mode="determinate")
    overall_progress.pack(pady=2)
    suplabel = tk.Label(window, text="Overall progress")
    suplabel.pack()

    line = tk.Frame(window, height=2, bd=1, relief="sunken")
    line.pack(fill="x", padx=5, pady=5)

    # Create a frame to hold the individual progress bars
    frame = tk.Frame(window)
    frame.pack()

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
    start_button = tk.Button(window, text="Start", command=lambda: start_proc(queue))
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
