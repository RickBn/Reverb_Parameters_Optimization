import numpy as np
import pedalboard
import pedalboard_native
import skopt
from pedalboard import Pedalboard, Reverb, load_plugin


def retrieve_external_vst3_params(vst3: pedalboard.pedalboard.VST3Plugin) -> (dict, list):

	params = vst3.parameters

	param_names = list(params.keys())
	param_names.remove('bypass')

	parameters = {}
	ranges = []

	for p in param_names:
		parameters[p] = vst3.__getattr__(p)
		ranges.append(skopt.space.space.Real(params[p].range[0], params[p].range[1], transform='identity'))
		#ranges.append((params[p].range[0], params[p].range[1]))

	return parameters, ranges


def external_vst3_set_params(params: dict, vst3: pedalboard.pedalboard.VST3Plugin) \
		-> pedalboard.pedalboard.VST3Plugin:

	for p in params:
		vst3.__setattr__(p, params[p])

	print(params)

	return


def external_vst3_fix_param_ranges(params: dict, vst3: pedalboard.pedalboard.VST3Plugin):

	for p in params:
		par = vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0]

		if par.approximate_step_size is not None:
			if par.approximate_step_size >= 0.5:
				step = 0.1
			elif par.approximate_step_size <= 0.05:
				step = 0.01

			new_range = np.arange(par.min_value, par.max_value + step, step).round(str(step)[::-1].find('.'))

			vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0]. \
				range = (par.min_value, par.max_value, step)

			vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0].\
				valid_values = new_range

			vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0]. \
				step_size = step

			vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0]. \
				approximate_step_size = None

			nnr = np.array([1 / len(new_range)] * len(new_range))
			nnr[0] = 0
			nnr = list(np.cumsum(nnr))
			nnr.append(1)

			nvtr = {}

			for i, v in enumerate(new_range):
				nvtr[v] = (nnr[i], nnr[i + 1])

			vst3.__getattr__(p)._AudioProcessorParameter__get_cpp_parameter().args[0].\
				_value_to_raw_value_ranges = nvtr

	return


def native_reverb_set_params(params: dict) -> pedalboard_native.Reverb:
	r = Reverb(freeze_mode=0)

	for p in params:
		r.__setattr__(p, params[p])

	print(r)

	return r


def plugin_process(vst3, audio, sr):
	effected = vst3(audio, sample_rate=sr)

	return effected


def board_process(board, audio):
	effected = board(audio)

	return effected

