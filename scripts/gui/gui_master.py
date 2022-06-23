import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
from scripts.gui.plots import TkPyplot
from scripts.gui.audio_handler import TkAudioHandler

from functools import partial

from matplotlib.backend_bases import key_press_handler

import soundfile as sf

from typing import List

root = tk.Tk()
root.wm_title("Reverb Parameters Optimizer")


def on_closing():
    messagebox.askokcancel("Quit", "Do you want to quit?")
    root.destroy()


tk_plot = TkPyplot(root)
canvas = tk_plot.canvas
toolbar = tk_plot.toolbar

audio_handler = TkAudioHandler()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

plugin_list = []

init_rir_path = 'audio/input/chosen_rirs/'

load_rir_files = partial(audio_handler.load_audio_files,
                           initdir=init_rir_path,
                           filetype=[('Wav Files', '.wav')],
                           tk_plot=tk_plot)

next_audio_plot = partial(audio_handler.show_next_audio_plot, tk_plot=tk_plot)

tk.Button(root, text="Quit", command=on_closing).pack(side=tk.BOTTOM)
tk.Button(root, text="Show next plot", command=next_audio_plot).pack(side=tk.BOTTOM)
tk.Button(root, text="Load reference RIRs", command=load_rir_files).pack(side=tk.BOTTOM)

toolbar.pack(side=tk.TOP, fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()

rir_files = audio_handler.get_audio_files()

print(rir_files)
print(plugin_list)
