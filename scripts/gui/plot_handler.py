import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import figure
import numpy as np


class TkPyplot(tk.Frame):

    def __init__(self, master, fig_w: int = 100, fig_h: int = 100, fig_dpi: int = 72, shared_ax=None):
        super().__init__()

        self.fig = figure.Figure(figsize=(fig_w/fig_dpi, fig_h/fig_dpi), dpi=fig_dpi)

        self.ax = self.fig.subplots()

        self.line, = self.ax.plot(0)
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("A")
        self.ax.legend(["Channel: " + str(0)])
        self.ax.autoscale()

        if shared_ax is not None:
            self.ax.get_shared_y_axes().join(self.ax, shared_ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        self.toolbar.update()

    def get_ax(self):
        return self.ax

    def update_plot(self, audio: np.ndarray, title: str, channel: int):

        x = np.arange(0, len(audio.T), 1)

        self.line.set_data(x, audio)
        self.ax.set_title(title)
        self.ax.legend(["Channel: " + str(channel)])

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()
