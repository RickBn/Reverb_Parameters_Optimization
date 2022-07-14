import functools

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.parameters_learning import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.audio_metrics import *
from scripts.audio_functions.signal_generation import *
from scripts.utils.plot_functions import *
from scripts.direct_sound_eq import *

from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":
    # rir_path = 'audio/_SIM/'
    # er_path = 'audio/_SIM_TRIMMED/'

    rir_path = 'audio/input/chosen_rirs/'
    er_path = 'audio/trimmed_rirs/'
    result_path = 'audio/results/'
    input_path = 'audio/input/sounds/'

    find_params_merged(rir_path, er_path, result_path, input_path, generate_references=True, pre_norm=False)



