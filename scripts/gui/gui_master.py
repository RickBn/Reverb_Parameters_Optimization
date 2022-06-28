import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Tuple

from scripts.gui.plot_handler import TkPyplot
from scripts.gui.audio_handler import TkAudioHandler
from scripts.gui.gui_plot_comparison import TkGuiPlotComparison
from scripts.gui.gui_utils import *
from functools import partial
from matplotlib import figure


class TkGuiHandler:

    def __init__(self,
                 root,
                 title: str,
                 tab_labels: Iterable[str],
                 sw_ratio: float = 0.85,
                 sh_ratio: float = 0.85):

        root.wm_title(title)

        screen_width = int(root.winfo_screenwidth() * sw_ratio)
        screen_height = int(root.winfo_screenheight() * sh_ratio)
        root.geometry(f'{screen_width}x{screen_height}')

        self.tabControl = ttk.Notebook(root, width=screen_width, height=screen_height)
        self.tabControl.grid(sticky="nsew")

        self.tab = list()
        for i, tab_name in enumerate(tab_labels):
            self.tab.append(tk.Frame(self.tabControl))
            self.tabControl.add(self.tab[i], text=tab_name)
            self.tab[i].grid(row=0, column=0, sticky="nsew")

            quit_btn = tk.Button(self.tab[i], text="Quit", command=partial(self.on_closing, root))
            quit_btn.grid(row=0, column=1, sticky="ne")

        TkGuiPlotComparison(self.tab[1], screen_width, screen_height)

        configure_grid_all(root)
        configure_grid_all(self.tabControl)

        for t in self.tab:
            configure_grid_all(t)

        self.window_raise(root)

    def window_raise(self, root):
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.attributes("-topmost", False)

    def on_closing(self, root):
        messagebox.askokcancel("Quit", "Do you want to quit?")
        root.destroy()


if __name__ == "__main__":
    tk_main = tk.Tk()

    window_title = "Reverb Parameters Optimizer"
    default_tabs = ("Parameters Optimizer", "Plots comparison")
    TkGuiHandler(root=tk_main, title=window_title, tab_labels=default_tabs)

    tk_main.mainloop()

