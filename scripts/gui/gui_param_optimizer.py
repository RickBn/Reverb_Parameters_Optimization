from scripts.gui.audio_handler import TkAudioHandler
from scripts.gui.gui_utils import *
from functools import partial
from scripts.reverb_parameters_optimize import find_params_merged


class TkGuiParamOptimizer(tk.Frame):
    def __init__(self,
                 master,
                 rir_path: str,
                 er_path: str,
                 result_path: str,
                 input_path: str,
                 generate_references: bool = False,
                 pre_norm: bool = True):

        super().__init__()

        self.canvas = tk.Canvas(master)
        canvas_frame = tk.Frame(self.canvas)
        vscroll = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=0, sticky="nse")

        # creates the label widgets
        self._widgets = [];
        columns = 6
        for row in range(0, 80):
            current_row = []
            for column in range(columns):
                label = tk.Label(canvas_frame, text="Rir name", border=2)
                #label.config(bg="White", font=("Calibri bold", 20))
                label.grid(row=(row + 2), column=column, sticky="nsew", padx=1, pady=1)
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column, weight=1)

        self.canvas.create_window((0, 0), window=canvas_frame, anchor='nw')
        master.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        canvas_frame.bind('<Enter>', self._bound_to_mousewheel)
        canvas_frame.bind('<Leave>', self._unbound_to_mousewheel)

        self.load_audio_button = tk.Button(master, text="OPTIMIZE", border=5)#, command=load_rir_files)
        self.load_audio_button.grid(row=4, column=0, sticky="nsew")

    def _bound_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")



