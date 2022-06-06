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
from scripts.audio_functions.rir_functions import *


def generate_vst_rir(params_path, input_audio, sr, max_dict, rev_name='fv', rev_external=None, save_path=None):
	if rev_external is not None and rev_name == 'fv':
		raise Exception("Attention! Reverb name is the default native reverb one but you loaded an external reverb!")

	effect_params = rev_name

	for rir_idx, rir in enumerate(os.listdir(params_path)):

		current_param_path = params_path + rir + '/'
		model_path = current_param_path + effect_params + '/'

		dp_scale_factor = max_dict[rir + '.wav']
		#scaled_input = input_audio * dp_scale_factor

		for model in os.listdir(model_path):

			params = model_load(model_path + model)

			if rev_external is not None:
				reverb_norm = process_external_reverb(params, rev_external, sr, input_audio, hp_cutoff=20, norm=True)

			else:
				reverb_norm = process_native_reverb(params, sr, input_audio, hp_cutoff=20, norm=True)

			reverb_norm *= dp_scale_factor

			if save_path is not None:
				sf.write(save_path + rir + '/' + rir + '_' + effect_params + '.wav', reverb_norm.T, sr)

	# elif effect_params == 'fv':
	#
	# 	reverb_norm_native = process_native_reverb(params, sr, input_audio, hp_cutoff=20)
	#
	# 	if save_path is not None:
	# 		sf.write(save_path + rir + '/' + rir + '_' + effect_params + '.wav', reverb_norm_native.T, sr)


def merge_er_tail_rir(er_path, tail_path, fade_length=128, trim=None, save_path=None):
	er_files = os.listdir(er_path)
	tail_files = os.listdir(tail_path)

	for rir in tail_files:

		effect_path = tail_path + rir + '/'

		for effect_rir in os.listdir(effect_path):

			er_rir, er_sr = sf.read(er_path + rir + '.wav')
			er_rir = er_rir.T

			tail_rir, tail_sr = sf.read(effect_path + effect_rir)
			tail_rir = tail_rir.T

			if er_sr != tail_sr:
				raise Exception("Warning! ER and tail sampling rate doesn't match!")

			padded_er_rir = pad_signal(er_rir, len(er_rir), len(tail_rir.T) - 128)

			fade_in_tail = tail_rir * cosine_fade(len(tail_rir.T), fade_length, False)

			start_point = len(er_rir.T) - 128

			# print(start_point)
			#
			# print(len(padded_er_rir[:, start_point:].T))
			# print(len(fade_in_tail.T))

			padded_er_rir[:, start_point:] += fade_in_tail

			if trim is not None:
				padded_er_rir = padded_er_rir[:, :(trim * er_sr)]
				padded_er_rir *= cosine_fade(len(padded_er_rir.T), fade_length)

			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)

			ax.plot(padded_er_rir[0])

			if save_path is not None:
				sf.write(save_path + effect_rir, padded_er_rir.T, er_sr)

	# print(padded_er_rir.shape)


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

	rir_max = rir_maximum(rir_path)

	generate_vst_rir(params_path, impulse, sr, rir_max, 'fv', None, save_path=vst_rir_save_path)
	generate_vst_rir(params_path, impulse, sr, rir_max, 'fdn', rev_external, save_path=vst_rir_save_path)

	er_path = 'audio/trimmed_rirs/'
	tail_path = 'audio/vst_rirs/'
	merged_rirs_path = 'audio/merged_rirs/'

	fade_in = int(5 * sr * 0.001)

	merge_er_tail_rir(er_path, tail_path, fade_length=fade_in, trim=3, save_path=merged_rirs_path)

	input_sound_path = 'audio/input/sounds/'
	batch_input_sound = prepare_batch_input_stereo(input_sound_path)
	batch_convolution = prepare_batch_convolve(merged_rirs_path)

	merged_final_path = 'audio/merged_final/'
	batch_convolve(batch_input_sound, batch_convolution, input_sound_path, merged_rirs_path, 44100, scale_factor=0.70,
	               save_path=merged_final_path)
