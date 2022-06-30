from tkinter import filedialog, messagebox, ttk
from typing import Iterable
from scripts.gui.gui_param_optimizer import TkGuiParamOptimizer
from scripts.gui.gui_plot_comparison import TkGuiPlotComparison
from scripts.gui.gui_utils import *
from functools import partial


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

        self.tab = list()
        for i, tab_name in enumerate(tab_labels):
            self.tab.append(tk.Frame(self.tabControl))
            self.tabControl.add(self.tab[i], text=tab_name)

        self.tabControl.grid(sticky="nsew")

        quit_btn = tk.Button(self.tabControl, text="Quit", command=partial(self.on_closing, root))
        quit_btn.grid(row=0, column=1, sticky="ne")

        rir_path = 'audio/input/chosen_rirs/'
        er_path = 'audio/trimmed_rirs/'
        result_path = 'audio/results/'
        input_path = 'audio/input/sounds/'

        TkGuiParamOptimizer(self.tab[0], rir_path, er_path, result_path, input_path)
        TkGuiPlotComparison(self.tab[1], screen_width, screen_height)

        configure_grid_all(root)
        configure_grid_all(self.tabControl)

        for t in self.tab:
            configure_grid_all(t)

        self.window_raise(root)

    def window_raise(self, root: tk.Tk):
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.attributes("-topmost", False)

    def on_closing(self, root: tk.Tk):
        messagebox.askokcancel("Quit", "Do you want to quit?")
        root.destroy()


if __name__ == "__main__":
    tk_main = tk.Tk()

    window_title = "Reverb Parameters Optimizer"
    default_tabs = ("Parameters Optimizer", "Plots comparison")
    TkGuiHandler(root=tk_main, title=window_title, tab_labels=default_tabs)

    tk_main.mainloop()

