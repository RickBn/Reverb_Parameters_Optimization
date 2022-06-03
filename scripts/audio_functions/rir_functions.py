import os
from kneed import KneeLocator

from scripts.utils.json_functions import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.audio_metrics import *


def rir_psd(rir_path, sr, frame_size=512, fade_factor=4, early_trim=500):

	arm_dict, psd_dict, lsd_dict = [{}] * 3
	fade_length = (frame_size // fade_factor)
	er = int((sr * 0.001) * early_trim)

	rir_files = os.listdir(rir_path)

	for idx, rir_file in enumerate(rir_files):

		a_a, p_a, l_a = [[]] * 3

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir_ms = ms_matrix(rir)

		for rir in rir_ms:
			a_ms, p_ms, l_ms = [[]] * 3

			rir_first_msec = rir[:er]

			xpad = pad_windowed_signal(rir_first_msec, frame_size)

			for chunk in range(0, len(xpad)-frame_size, frame_size)[1:]:

				xchunk = xpad[:chunk]
				xfaded = xchunk * cosine_fade(len(xchunk), fade_length)
				ar = armodel(xfaded, 2 * math.floor(2 + sr / 1000))
				a_ms.append(ar)

				rir_psd = ar2psd(ar, 2048)
				p_ms.append(rir_psd)

			a_a.append(a_ms)
			p_a.append(p_ms)

		arm_dict[rir_file] = a_a
		psd_dict[rir_file] = p_a

	return arm_dict, psd_dict

		# 	for psd_idx in range(1, len(p_ms), 1):
		# 		l_ms.append(log_spectral_distance(p_ms[psd_idx], p_ms[psd_idx-1]))
		#
		#
		# 	l_a.append(l_ms)
		#
		#
		# lsd_dict[rir_file] = l_a

	#
	# np.save('audio/armodels/arm_dict_ms.npy', arm_dict)
	# np.save('audio/armodels/psd_dict_ms.npy', psd_dict)
	# np.save('audio/armodels/lsd_dict_ms.npy', lsd_dict)


rir_path = 'audio/input/chosen_rirs/'
a_a, p_a = rir_psd(rir_path, 44100, 512, 4, 500)