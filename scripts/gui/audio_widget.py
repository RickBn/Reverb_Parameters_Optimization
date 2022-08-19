import tkinter as tk
from typing import Iterable

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import figure
import numpy as np
from scripts.utils.dict_functions import *
from functools import partial


class TkAudioWidget:

    def __init__(self):
        self.audio_buffer = dict()

    def audio_buffer_update(self, new_buffer: Iterable):
        self.audio_buffer = new_buffer


class TkPyplot(TkAudioWidget):

    def __init__(self,
                 master,
                 fig_w: int = 100,
                 fig_h: int = 100,
                 fig_dpi: int = 72,
                 row: int = 0,
                 col: int = 0,
                 sticky: str = "nsew",
                 shared_ax=None):

        super().__init__()

        #self.audio_files = dict()
        self.index = tk.IntVar()
        self.channel = tk.IntVar()
        self.numCh = 2

        self.fig = figure.Figure(figsize=(fig_w/fig_dpi, fig_h/fig_dpi), dpi=fig_dpi)

        self.ax = self.fig.subplots()

        self.line, = self.ax.plot(0)
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("A")
        self.ax.legend(["Channel: " + str(0)])
        self.ax.autoscale()

        if shared_ax is not None:
            self.ax.get_shared_y_axes().join(self.ax, shared_ax)

        next_audio_plot = partial(self.switch_plot, asc=True)
        prev_audio_plot = partial(self.switch_plot, asc=False)
        switch_channel = partial(self.switch_channel)

        self.switch_channel_button = tk.Button(master, text="Switch channel", command=switch_channel, state="disabled")

        self.switch_plot_frame = tk.Frame(master)
        self.switch_plot_frame.grid(row=row, column=col, sticky=sticky)

        self.prev_plot_button = tk.Button(self.switch_plot_frame, text="Previous plot", command=prev_audio_plot,
                                          state="disabled", width=10)
        self.prev_plot_button.pack(side="left")

        self.next_plot_button = tk.Button(self.switch_plot_frame, text="Next plot", command=next_audio_plot,
                                          state="disabled", width=10)
        self.next_plot_button.pack(side="left")

        self.switch_channel_button.grid(row=row + 1, column=col, sticky=sticky)

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

    def switch_channel(self):
        self.channel.set((self.channel.get() + 1) % self.numCh)

        idx = self.index.get()
        ch = self.channel.get()
        self.update_plot(get_dict_idx_value(self.audio_buffer, idx)[ch],
                         get_dict_idx_key(self.audio_buffer, idx),
                         ch)

    def switch_plot(self, asc=True):

        if not self.audio_buffer:
            return

        i = (self.index.get() + (1 if asc else -1)) % len(self.audio_buffer)
        self.index.set(i)

        ch = self.channel.get()
        self.numCh = len(get_dict_idx_value(self.audio_buffer, i))

        audio = get_dict_idx_value(self.audio_buffer, i)[ch]
        title = get_dict_idx_key(self.audio_buffer, i)
        self.update_plot(audio, title, ch)

    def audio_buffer_update(self, new_buffer: Dict[str, np.ndarray]):
        self.audio_buffer = new_buffer

        if self.audio_buffer:
            self.update_plot(get_dict_idx_value(self.audio_buffer, 0)[0],
                             get_dict_idx_key(self.audio_buffer, 0),
                             0)
            self.numCh = len(get_dict_idx_value(self.audio_buffer, 0))

            if self.next_plot_button["state"] == "disabled":
                self.next_plot_button.config(state="normal")
                self.prev_plot_button.config(state="normal")
                self.switch_channel_button.config(state="normal")
