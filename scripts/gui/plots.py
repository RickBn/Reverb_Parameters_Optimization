import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

import soundfile as sf

from typing import List

root = tk.Tk()
root.wm_title("Reverb Parameters Optimizer")

class TkPyplot(tk.Tk):

    def __init__(self, *args, fig_w: int = 5, fig_h: int = 4, fig_dpi: int = 100):
        #super().__init__()
        self.fig = Figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

        self.ax = self.fig.subplots(1, 2, sharex='all', sharey='all')

        #self.ax[0] = self.fig.add_subplot(1, 2, 1)
        self.line1, = self.ax[0].plot(0)
        self.ax[0].set_xlabel("t")
        self.ax[0].set_ylabel("A")
        self.ax[0].autoscale()

        #self.ax[1] = self.fig.add_subplot(1, 2, 2)
        self.line2, = self.ax[1].plot(0)
        self.ax[1].set_xlabel("t")
        self.ax[1].set_ylabel("A")
        self.ax[1].autoscale()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.update()

    def update_plot(self, audio: np.ndarray):

        x = np.arange(0, len(audio.T), 1)

        self.line1.set_data(x, audio)
        self.line2.set_data(x, audio)

        self.ax[0].relim()
        self.ax[0].autoscale_view()

        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.canvas.draw()


def load_audio_files(initdir: str, filetype: List[tuple], tk_plot: TkPyplot):
    audio_files = []

    file_paths = filedialog.askopenfilenames(initialdir=initdir, filetypes=filetype)

    for fp in file_paths:
        af = sf.read(fp)[0].T
        audio_files.append(af)

    # Plotting the first audio file of the list
    tk_plot.update_plot(audio_files[0][0])

    return audio_files


def show_next_audio_plot(audio_files: List[np.ndarray], tk_plot: TkPyplot, index: tk.IntVar):

    i = (index.get() + 1) % len(audio_files)
    tk_plot.update_plot(audio_files[i][0])

    counter.set(i)



def on_closing():
    messagebox.askokcancel("Quit", "Do you want to quit?")
    root.destroy()


tkplot = TkPyplot(root)
canvas = tkplot.canvas
toolbar = tkplot.toolbar


canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

rir_list = []
plugin_list = []

assign_rirs = lambda x=root: [rir_list.append(f) for f in
                              load_audio_files('audio/input/chosen_rirs/', [('Wav Files', '.wav')], tkplot)]


tk.Button(root, text="Load reference RIRs", command=assign_rirs).pack(side=tk.BOTTOM)


counter = tk.IntVar()

next_audio_plot = lambda x= root: [show_next_audio_plot(rir_list, tkplot, counter)]
tk.Button(root, text="Show next plot", command=next_audio_plot).pack(side=tk.BOTTOM)



button_quit = tk.Button(master=root, text="Quit", command=on_closing)



# slider_update = tk.Scale(root, from_=1, to=5, orient=tk.HORIZONTAL,
#                               command=update_frequency, label="Frequency [Hz]")


button_quit.pack(side=tk.BOTTOM)
#slider_update.pack(side=tk.BOTTOM)
toolbar.pack(side=tk.TOP, fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()

print(rir_list)
print(plugin_list)
