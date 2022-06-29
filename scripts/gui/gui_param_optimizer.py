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

        ref_rirs = TkAudioHandler(master, default_audio_path=rir_path, row=0, col=0, sticky="nw")
        trimmed_rirs = TkAudioHandler(master, default_audio_path=er_path, row=1, col=0, sticky="nw")
        result_rirs = TkAudioHandler(master, default_audio_path=result_path, row=2, col=0, sticky="nw")
        input_sounds = TkAudioHandler(master, default_audio_path=input_path, row=3, col=0, sticky="nw")

        # load_rir_files = partial(find_params_merged,
        #                          init_dir=default_audio_path,
        #                          filetype=[('Wav Files', '.wav')],
        #                          tk_plot=tk_plot)
        #
        self.load_audio_button = tk.Button(master, text="OPTIMIZE", border=5)#, command=load_rir_files)
        self.load_audio_button.grid(row=4, column=0, sticky="nsew")



