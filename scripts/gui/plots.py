import tkinter as tk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

import soundfile as sf

from typing import List

root = tk.Tk()
root.wm_title("Embedding in Tk")

class TkPyplot(tk.Tk):

    def __init__(self, *args, fig_w: int = 5, fig_h: int = 4, fig_dpi: int = 100):
        #super().__init__()
        self.fig = Figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.line1, = self.ax1.plot(np.arange(0, 44100, 1), np.arange(0, 44100, 1))#t, 2 * np.sin(2 * np.pi * t))
        self.ax1.set_xlabel("time [s]")
        self.ax1.set_ylabel("f(t)")
        #self.ax1.set_ylim(-1, 1)
        self.ax1.autoscale()

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.line2, = self.ax2.plot(np.arange(0, 44100, 1), np.arange(0, 44100, 1))
        self.ax2.set_xlabel("time [s]")
        self.ax2.set_ylabel("f(t)")
        #self.ax2.set_ylim(-1, 1)
        self.ax2.autoscale()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.update()

    def update_plot(self, audio: np.ndarray):

        x = np.arange(0, len(audio.T), 1)

        self.line1.set_data(x, audio)
        self.line2.set_data(x, audio)

        self.ax1.relim()
        self.ax1.autoscale_view()

        self.ax2.relim()
        self.ax2.autoscale_view()

        self.canvas.draw()


def open_file(initdir: str, filetype: List[tuple], tk_plot: TkPyplot):
    file = filedialog.askopenfilenames(initialdir=initdir, filetypes=filetype)

    rir, sr = sf.read(file[0])
    rir = rir.T

    tkplot.update_plot(rir[0])

    return rir


tkplot = TkPyplot(root)
canvas = tkplot.canvas
toolbar = tkplot.toolbar


canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tk.Button(master=root, text="Quit", command=root.quit)

rir_list = []
plugin_list = []

assign_rirs = lambda x=root: [rir_list.append(f) for f in
                              open_file('audio/input/chosen_rirs/', [('Wav Files', '.wav')], tkplot)]


tk.Button(root, text="RIRs", command=assign_rirs).pack(side=tk.BOTTOM)



#tkplot.update_plot(rir_list[0][0])

# slider_update = tk.Scale(root, from_=1, to=5, orient=tk.HORIZONTAL,
#                               command=update_frequency, label="Frequency [Hz]")

# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tk.BOTTOM)
#slider_update.pack(side=tk.BOTTOM)
toolbar.pack(side=tk.TOP, fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()

print(rir_list)
print(plugin_list)
