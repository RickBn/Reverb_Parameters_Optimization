import tkinter as tk
from tkinter import filedialog
from scripts.gui.audio_widget import *
from scripts.utils.dict_functions import *
from functools import partial
import soundfile as sf
from typing import List


class TkAudioHandler(tk.Frame):

    def __init__(self,
                 master,
                 default_audio_path: str,
                 row: int = 0,
                 col: int = 0,
                 sticky: str = "nsew",
                 slave_widget: TkAudioWidget = None):

        super().__init__()

        self.audio_files = dict()

        load_rir_files = partial(self.load_audio_files,
                                 init_dir=default_audio_path,
                                 filetype=[('Wav Files', '.wav')],
                                 slave_widget=slave_widget)

        self.load_audio_button = tk.Button(master, text="Load reference RIRs", command=load_rir_files)
        self.load_audio_button.grid(row=row, column=col, sticky=sticky)

    def get_audio_files(self):
        return self.audio_files

    def load_audio_files(self, init_dir: str, filetype: List[tuple], slave_widget: TkAudioWidget): #tk_plot: TkPyplot):

        self.audio_files = dict()

        file_paths = filedialog.askopenfilenames(initialdir=init_dir, filetypes=filetype)

        if not file_paths:
            return

        for fp in file_paths:
            af = sf.read(fp)[0].T
            self.audio_files[fp.split('/')[-1]] = af

        if slave_widget is not None:
            slave_widget.audio_buffer_update(self.audio_files)

