from typing import Dict
import numpy as np

from scripts.gui.audio_handler import TkAudioHandler
from scripts.gui.gui_utils import *
from scripts.gui.audio_widget import TkAudioWidget
from scripts.gui.gui_scrollable_canvas import TkScrollableCanvas
from functools import partial
from scripts.reverb_parameters_optimize import find_params_merged


class TkGuiParamOptimizer(tk.Frame, TkAudioWidget):
    def __init__(self,
                 master,
                 rir_path: str,
                 er_path: str,
                 result_path: str,
                 input_path: str,
                 generate_references: bool = False,
                 pre_norm: bool = True):

        super().__init__()

        self.optimize_button = tk.Button(master, text="OPTIMIZE", border=5)
        self.optimize_button.grid(row=5, column=2, sticky="nsew")

        self.scrollable_canvas = TkScrollableCanvas(master)
        self.canvas = self.scrollable_canvas.canvas
        self.canvas_frame = self.scrollable_canvas.canvas_frame

        audio_loader = TkAudioHandler(self.canvas_frame, default_audio_path=rir_path, row=0, col=0, sticky="nsew",
                                      slave_widget=self)

        optimize_label = tk.Label(self.canvas_frame, text="Optimize", borderwidth=2, relief="groove")
        optimize_label.grid(row=0, column=1, sticky="nsew")

    def audio_buffer_update(self, new_buffer: Dict[str, np.ndarray]):
        self.audio_buffer = new_buffer

        for idx, loaded_rir in enumerate(self.audio_buffer):
            label = tk.Label(self.canvas_frame, text=loaded_rir.replace('.wav', ''), border=2)
            checkbox = tk.Checkbutton(self.canvas_frame)
            label.grid(row=idx + 1, column=0, sticky="nsew")
            checkbox.grid(row=idx + 1, column=1, sticky="nsew")

        self.master.update()
        self.scrollable_canvas.configure_scroll_region()
