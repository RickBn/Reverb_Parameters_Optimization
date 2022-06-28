import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scripts.gui.plot_handler import TkPyplot
from scripts.gui.audio_handler import TkAudioHandler
from functools import partial
from matplotlib import figure

class TkGuiHandler:

    def __init__(self,
                 root,
                 title: str = "Reverb Parameters Optimizer",
                 sw_ratio: float = 0.85,
                 sh_ratio: float = 0.85):

        root.wm_title(title)

        screen_width = int(root.winfo_screenwidth() * sw_ratio)
        screen_height = int(root.winfo_screenheight() * sh_ratio)
        root.geometry(f'{screen_width}x{screen_height}')

        self.tabControl = ttk.Notebook(root, width=screen_width, height=screen_height)

        tab1 = tk.Frame(self.tabControl)

        tab2 = tk.Frame(self.tabControl)

        self.tabControl.add(tab1, text="Load Audio")
        self.tabControl.add(tab2, text="Parameters Optimizer")
        self.tabControl.grid(row=0, column=0)

        quit_button = tk.Button(tab1, text="Quit", command=partial(self.on_closing, root))
        quit_button.grid(row=0, column=1, sticky="ne") #.pack(side=tk.BOTTOM)

        init_rir_path = 'audio/input/chosen_rirs/'

        tk_plot1 = TkPyplot(tab1,
                            fig_w=int(screen_width*0.5),
                            fig_h=int(screen_height*0.75))

        canvas1 = tk_plot1.canvas
        toolbar1 = tk_plot1.toolbar

        tk_plot2 = TkPyplot(tab1,
                            fig_w=int(screen_width*0.5),
                            fig_h=int(screen_height*0.75),
                            shared_ax=tk_plot1.get_ax())

        canvas2 = tk_plot2.canvas
        toolbar2 = tk_plot2.toolbar

        audio_handler = TkAudioHandler(tab1, init_rir_path, tk_plot1, 0)
        audio_handler2 = TkAudioHandler(tab1, init_rir_path, tk_plot2, 1)

        toolbar1.grid(row=0, column=0, sticky='nw')
        toolbar2.grid(row=0, column=1, sticky='nw')
        canvas1.get_tk_widget().grid(row=1, column=0, sticky="w")
        canvas2.get_tk_widget().grid(row=1, column=1, sticky="e")

    def on_closing(self, root):
        messagebox.askokcancel("Quit", "Do you want to quit?")
        root.destroy()


tk_main = tk.Tk()
TkGuiHandler(tk_main)
tk_main.mainloop()
