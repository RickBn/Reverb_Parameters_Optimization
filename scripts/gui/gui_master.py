from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from typing import List

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import sys

class RPORoot(ttk.Frame):
    def __init__(self, master=None):
        ttk.Frame.__init__(self,master)
        self.createWidgets()

    def createWidgets(self):
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        ax = fig.add_subplot()
        line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("f(t)")

        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()

    def plot(self,canvas,ax):
        c = ['r','b','g']  # plot marker colors
        ax.clear()         # clear axes from previous plot
        for i in range(3):
            theta = np.random.uniform(0,360,10)
            r = np.random.uniform(0,1,10)
            ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
            canvas.draw()


def open_file(initdir: str, filetype: List[tuple]):
    file = filedialog.askopenfilenames(initialdir=initdir, filetypes=filetype)
    return file


root = Tk()
root.title("Reverb Parameters Optimizer")

frm = ttk.Frame(root, padding=100, borderwidth=10)
frm.grid()
rir_list = []
plugin_list = []

assign_rirs = lambda x=root: [rir_list.append(f) for f in
                              open_file('audio/input/chosen_rirs/', [('Wav Files', '.wav')])]

assign_plugins = lambda x=root: [plugin_list.append(f) for f in
                                 open_file('vst3/', [('VST3', '*.vst3')])]

ttk.Button(frm, text="RIRs", command=assign_rirs).grid(column=0, row=0)
ttk.Button(frm, text="Plugins", command=assign_plugins).grid(column=1, row=0)
root.mainloop()
print(rir_list)
print(plugin_list)




button_quit = tkinter.Button(master=root, text="Quit", command=root.quit)


def update_frequency(new_val):
    # retrieve frequency
    f = float(new_val)

    # update data
    y = 2 * np.sin(2 * np.pi * f * t)
    line1.set_data(t, y)
    line2.set_data(t, y*2)

    # required to update canvas and attached toolbar!
    canvas.draw()


slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL,
                              command=update_frequency, label="Frequency [Hz]")

# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM)
toolbar.pack(side=tkinter.TOP, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

tkinter.mainloop()