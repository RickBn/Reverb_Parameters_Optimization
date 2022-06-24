import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scripts.gui.plot_handler import TkPyplot
from scripts.gui.audio_handler import TkAudioHandler
from functools import partial

class TkGuiHandler:

    def __init__(self,
                 root,
                 title: str = "Reverb Parameters Optimizer",
                 sw_ratio: float = 0.75,
                 sh_ratio: float = 0.75):

        #super().__init__()

        root.wm_title(title)

        screen_width = int(root.winfo_screenwidth() * sw_ratio)
        screen_height = int(root.winfo_screenheight() * sh_ratio)
        root.geometry(f'{screen_width}x{screen_height}')

        self.tabControl = ttk.Notebook(root)

        tab1 = tk.Frame(self.tabControl)
        tab2 = tk.Frame(self.tabControl)

        self.tabControl.add(tab1, text="Load Audio")
        self.tabControl.add(tab2, text="Parameters Optimizer")
        self.tabControl.pack(expand=1, fill="both")

        quit_button = tk.Button(tab1, text="Quit", command=partial(self.on_closing, root))
        quit_button.pack(side=tk.BOTTOM)

        init_rir_path = 'audio/input/chosen_rirs/'

        tk_plot = TkPyplot(tab1)
        canvas = tk_plot.canvas
        toolbar = tk_plot.toolbar

        audio_handler = TkAudioHandler(tab1, init_rir_path, tk_plot)

        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def on_closing(self, root):
        messagebox.askokcancel("Quit", "Do you want to quit?")
        root.destroy()


tk_main = tk.Tk()
TkGuiHandler(tk_main)
tk_main.mainloop()
