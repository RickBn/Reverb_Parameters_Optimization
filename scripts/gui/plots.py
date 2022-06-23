import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib import figure

import numpy as np

class TkPyplot(tk.Tk):

    def __init__(self, root, fig_w: int = 12, fig_h: int = 7, fig_dpi: int = 100):
        self.fig = figure.Figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

        self.ax = self.fig.subplots(1, 2, sharex='all', sharey='all')

        self.line1, = self.ax[0].plot(0)
        self.ax[0].set_xlabel("t")
        self.ax[0].set_ylabel("A")
        self.ax[0].autoscale()

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
