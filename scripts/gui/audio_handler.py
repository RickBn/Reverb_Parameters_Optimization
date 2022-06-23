import tkinter as tk
from tkinter import filedialog, messagebox
from scripts.gui.plots import TkPyplot

import numpy as np

import soundfile as sf

from typing import List


class TkAudioHandler:

    def __init__(self):
        self.index = tk.IntVar()
        self.audio_files = list()

    def get_audio_files(self):
        return self.audio_files

    def load_audio_files(self, initdir: str, filetype: List[tuple], tk_plot: TkPyplot):

        self.audio_files = list()

        file_paths = filedialog.askopenfilenames(initialdir=initdir, filetypes=filetype)

        for fp in file_paths:
            af = sf.read(fp)[0].T
            self.audio_files.append(af)

        # Plotting the first audio file of the list
        tk_plot.update_plot(self.audio_files[0][0])

        return self.audio_files

    def show_next_audio_plot(self, tk_plot: TkPyplot):

        i = (self.index.get() + 1) % len(self.audio_files)
        tk_plot.update_plot(self.audio_files[i][0])

        self.index.set(i)
