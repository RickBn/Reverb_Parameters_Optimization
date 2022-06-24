import tkinter as tk
from tkinter import filedialog
from scripts.gui.plot_handler import TkPyplot
from functools import partial
import soundfile as sf
from typing import List


class TkAudioHandler(tk.Frame):

    def __init__(self, master, default_audio_path: str, tk_plot: TkPyplot):
        super().__init__(master)

        self.index = tk.IntVar()
        self.channel = tk.IntVar()
        self.audio_files = list()

        load_rir_files = partial(self.load_audio_files,
                                 initdir=default_audio_path,
                                 filetype=[('Wav Files', '.wav')],
                                 tk_plot=tk_plot)

        next_audio_plot = partial(self.show_next_audio_plot, tk_plot=tk_plot)
        switch_channel = partial(self.switch_channel, tk_plot=tk_plot)

        self.switch_channel_button = tk.Button(master, text="Switch channel", command=switch_channel, state="disabled")
        self.next_plot_button = tk.Button(master, text="Show next plot", command=next_audio_plot, state="disabled")
        self.load_audio_button = tk.Button(master, text="Load reference RIRs", command=load_rir_files)

        self.switch_channel_button.pack(side=tk.BOTTOM)
        self.next_plot_button.pack(side=tk.BOTTOM)
        self.load_audio_button.pack(side=tk.BOTTOM, anchor=tk.SW, padx=200)

    def get_audio_files(self):
        return self.audio_files

    def switch_channel(self, tk_plot: TkPyplot = None):
        self.channel.set((self.channel.get() + 1) % 2)

        if tk_plot is not None:
            tk_plot.update_plot(self.audio_files[self.index.get()][self.channel.get()])


    def load_audio_files(self, initdir: str, filetype: List[tuple], tk_plot: TkPyplot):

        self.audio_files = list()

        file_paths = filedialog.askopenfilenames(initialdir=initdir, filetypes=filetype)

        if not file_paths:
            return

        for fp in file_paths:
            af = sf.read(fp)[0].T
            self.audio_files.append(af)

        # Plotting the first audio file of the list
        tk_plot.update_plot(self.audio_files[0][0])

        if self.next_plot_button["state"] == "disabled":
            self.next_plot_button.config(state="normal")
            self.switch_channel_button.config(state="normal")

        return self.audio_files

    def show_next_audio_plot(self, tk_plot: TkPyplot):

        if not self.audio_files:
            return

        i = (self.index.get() + 1) % len(self.audio_files)
        tk_plot.update_plot(self.audio_files[i][self.channel.get()])

        self.index.set(i)
