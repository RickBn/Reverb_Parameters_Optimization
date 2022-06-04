import os

import numpy as np
import pandas as pd

from scripts.parameters_learning import *
from scripts.utils.json_functions import *
from scripts.utils.plot_functions import *

from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.signal_generation import *
from scripts.audio_functions.reverb_features import *
from scripts.audio_functions.DSPfunc import *

rir_path = '../audio/input/chosen_rirs/'
rir_file = os.listdir(rir_path)
rir_folder = os.listdir('../audio/results')

rir, sr = sf.read(rir_path + rir_file[0])
rir = rir.T

test_sound = create_impulse(sr * 6)
test_sound = np.stack([test_sound, test_sound])

#rir_eq_coeffs = np.load('audio/armodels/rir_eq_coeffs_ms.npy', allow_pickle=True)[()]
rir_eq_coeffs = np.load('../audio/armodels/rir_eq_coeffs_kl_s.npy', allow_pickle=True)[()]

rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
rev_param_names_ex.pop('fdn_size_internal')
rev_param_ranges = rev_param_ranges_ex[:-1]