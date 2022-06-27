import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import figure
import numpy as np


class TkPyplot(tk.Frame):

    def __init__(self, master, fig_w: int = 4, fig_h: int = 4, fig_dpi: int = 100):
        super().__init__(master)

        self.fig = figure.Figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

        self.ax = self.fig.subplots(1, 1)#self.fig.subplots(1, 2, sharex='all', sharey='all')

        self.line1, = self.ax.plot(0)
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("A")
        self.ax.autoscale()

        # self.line2, = self.ax[1].plot(0)
        # self.ax[1].set_xlabel("t")
        # self.ax[1].set_ylabel("A")
        # self.ax[1].autoscale()

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        self.toolbar.update()

    def update_plot(self, audio: np.ndarray, title: str):

        x = np.arange(0, len(audio.T), 1)

        self.line1.set_data(x, audio)

        self.ax.set_title(title)
        self.ax.relim()
        self.ax.autoscale_view()

        # self.ax[1].set_title(title)
        # self.ax[1].relim()
        # self.ax[1].autoscale_view()

        self.canvas.draw()
