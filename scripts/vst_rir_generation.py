import os

import numpy as np
import pandas as pd

from scripts.parameters_learning import *
from scripts.utils.json_functions import *
from scripts.utils.plot_functions import *

from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.signal_generation import *
from scripts.audio_functions.pedalboard_process import *


def generate_vst_rir(params_path, input_audio, sr, rev_name='fv', rev_external=None, save_path=None):

	if rev_external is not None and rev_name == 'fv':
		raise Exception("Attention! Reverb name is the default native reverb one but you loaded an external reverb!")

	effect_params = rev_name

	for rir_idx, rir in enumerate(os.listdir(params_path)):

		current_param_path = params_path + rir + '/'
		model_path = current_param_path + effect_params + '/'

		for model in os.listdir(model_path):

			params = model_load(model_path + model)

			if rev_external is not None:
				reverb_norm = process_external_reverb(params, rev_external, sr, input_audio, hp_cutoff=20)

			else:
				reverb_norm = process_native_reverb(params, sr, input_audio, hp_cutoff=20)

			if save_path is not None:
					sf.write(save_path + rir + '/' + rir + '_' + effect_params + '.wav', reverb_norm.T, sr)

			# elif effect_params == 'fv':
			#
			# 	reverb_norm_native = process_native_reverb(params, sr, input_audio, hp_cutoff=20)
			#
			# 	if save_path is not None:
			# 		sf.write(save_path + rir + '/' + rir + '_' + effect_params + '.wav', reverb_norm_native.T, sr)


if __name__ == "__main__":
	rir_path = 'audio/trimmed_rirs/'
	rir_files = os.listdir(rir_path)

	sr = 44100

	impulse = create_impulse(sr * 6)
	impulse = np.stack([impulse, impulse])

	rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
	rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

	rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
	rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
	rev_param_names_ex.pop('fdn_size_internal')
	rev_param_ranges = rev_param_ranges_ex[:-1]

	params_path = 'audio/params/'
	params_folder = os.listdir(params_path)

	vst_rir_save_path = 'audio/vst_rirs/'

	generate_vst_rir(params_path, impulse, sr, 'fv', None, save_path=vst_rir_save_path)
	generate_vst_rir(params_path, impulse, sr, 'fdn', rev_external, save_path=vst_rir_save_path)
