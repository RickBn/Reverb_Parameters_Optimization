import tkinter as tk
from tkinter import filedialog
from scripts.gui.plot_handler import TkPyplot
from scripts.utils.dict_functions import *
from functools import partial
import soundfile as sf
from typing import List


class TkAudioHandler(tk.Frame):

    def __init__(self, master, default_audio_path: str, tk_plot: TkPyplot, col: int):
        super().__init__()

        self.index = tk.IntVar()
        self.channel = tk.IntVar()
        self.audio_files = dict()

        load_rir_files = partial(self.load_audio_files,
                                 initdir=default_audio_path,
                                 filetype=[('Wav Files', '.wav')],
                                 tk_plot=tk_plot)

        next_audio_plot = partial(self.show_next_audio_plot, tk_plot=tk_plot)
        switch_channel = partial(self.switch_channel, tk_plot=tk_plot)

        self.switch_channel_button = tk.Button(master, text="Switch channel", command=switch_channel, state="disabled")
        self.next_plot_button = tk.Button(master, text="Show next plot", command=next_audio_plot, state="disabled")
        self.load_audio_button = tk.Button(master, text="Load reference RIRs", command=load_rir_files)

        self.load_audio_button.grid(row=2, column=col, sticky="n")
        self.next_plot_button.grid(row=3, column=col, sticky="n")
        self.switch_channel_button.grid(row=4, column=col, sticky="n")

    def get_audio_files(self):
        return self.audio_files

    def switch_channel(self, tk_plot: TkPyplot = None):
        self.channel.set((self.channel.get() + 1) % 2)

        if tk_plot is not None:
            idx = self.index.get()
            ch = self.channel.get()
            tk_plot.update_plot(get_dict_idx_value(self.audio_files, idx)[ch],
                                get_dict_idx_key(self.audio_files, idx),
                                ch)

    def load_audio_files(self, init_dir: str, filetype: List[tuple], tk_plot: TkPyplot):

        self.audio_files = dict()

        file_paths = filedialog.askopenfilenames(initialdir=init_dir, filetypes=filetype)

        if not file_paths:
            return

        for fp in file_paths:
            af = sf.read(fp)[0].T
            self.audio_files[fp.split('/')[-1]] = af

        if self.next_plot_button["state"] == "disabled":
            self.next_plot_button.config(state="normal")
            self.switch_channel_button.config(state="normal")

        # Plotting the first audio file of the list
        tk_plot.update_plot(get_dict_idx_value(self.audio_files, 0)[0],
                            get_dict_idx_key(self.audio_files, 0),
                            self.channel.get())

    def show_next_audio_plot(self, tk_plot: TkPyplot):

        if not self.audio_files:
            return

        i = (self.index.get() + 1) % len(self.audio_files)
        self.index.set(i)

        ch = self.channel.get()

        audio = get_dict_idx_value(self.audio_files, i)[ch]
        title = get_dict_idx_key(self.audio_files, i)
        tk_plot.update_plot(audio, title, ch)


