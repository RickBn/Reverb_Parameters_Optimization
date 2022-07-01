from typing import Dict
import numpy as np

from scripts.gui.audio_handler import TkAudioHandler
from scripts.gui.gui_utils import *
from scripts.gui.audio_widget import TkAudioWidget
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
        self.optimize_button.grid(row=4, column=1, sticky="nsew")

        self.canvas = tk.Canvas(master)
        self.canvas_frame = tk.Frame(self.canvas)
        v_scroll = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=0, sticky="nse")

        audio_loader = TkAudioHandler(self.canvas, default_audio_path=rir_path, row=4, col=0, sticky="nsew",
                                      slave_widget=self)
        # creates the label widgets
        # self._widgets = []
        # columns = 6
            # self._widgets.append(current_row)

        # for column in range(columns):
        #     self.grid_columnconfigure(column, weight=1)

        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor='nw')
        master.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas_frame.bind('<Enter>', self._bound_to_mousewheel)
        self.canvas_frame.bind('<Leave>', self._unbound_to_mousewheel)


    def _bound_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def audio_buffer_update(self, new_buffer: Dict[str, np.ndarray]):
        self.audio_buffer = new_buffer

        for idx, loaded_rir in enumerate(self.audio_buffer):
            current_row = []
            label = tk.Label(self.canvas_frame, text="Rir name", border=2)
            #label.config(bg="White", font=("Calibri bold", 20))
            label.grid(row=(idx + 2), column=0, sticky="nsew", padx=1, pady=1)
            current_row.append(label)



