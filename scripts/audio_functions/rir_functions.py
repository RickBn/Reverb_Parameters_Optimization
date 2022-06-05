import os
from typing import Dict, Optional, Any

from kneed import KneeLocator

from scripts.utils.json_functions import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.audio_metrics import *


def rir_psd_metrics(rir_path, sr, frame_size=512, fade_factor=4, early_trim=500, save_path=None):

	arm_dict = {}
	psd_dict = {}
	lsd_dict = {}

	fade_length = (frame_size // fade_factor)
	er = int((sr * 0.001) * early_trim)

	rir_files = os.listdir(rir_path)

	for idx, rir_file in enumerate(rir_files):

		a_a = []
		p_a = []
		l_a = []

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir_ms = ms_matrix(rir)

		for rir in rir_ms:
			a_ms = []
			p_ms = []
			l_ms = []

			rir_first_msec = rir[:er]

			xpad = pad_windowed_signal(rir_first_msec, frame_size)

			for chunk in range(0, len(xpad) - frame_size, frame_size)[1:]:
				xchunk = xpad[:chunk]
				xfaded = xchunk * cosine_fade(len(xchunk), fade_length)
				ar = armodel(xfaded, 2 * math.floor(2 + sr / 1000))
				a_ms.append(ar)

				rir_psd = ar2psd(ar, 2048)
				p_ms.append(rir_psd)

			for psd_idx in range(1, len(p_ms), 1):
				l_ms.append(log_spectral_distance(p_ms[psd_idx], p_ms[psd_idx - 1]))

			a_a.append(a_ms)
			p_a.append(p_ms)
			l_a.append(l_ms)

		arm_dict[rir_file] = a_a
		psd_dict[rir_file] = p_a
		lsd_dict[rir_file] = l_a

		if save_path is not None:
			print('Hi')
			#'audio/armodels/arm_dict_ms.npy'
			np.save(save_path + '/arm_dict_ms.npy', arm_dict)
			np.save(save_path + '/psd_dict_ms.npy', psd_dict)
			np.save(save_path + '/lsd_dict_ms.npy', lsd_dict)

	return arm_dict, psd_dict, lsd_dict


def rir_er_detection(rir_path, lsd_dict, early_trim=500, img_path=None, cut_dict_path=None):

	cut_dict = {}

	rir_files = os.listdir(rir_path)

	for rir_file in rir_files:

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir = ms_matrix(rir)[0]

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)

		er = int((sr * 0.001) * early_trim)
		ax.plot(normalize_audio(rir[:er]))

		x = []
		y = []
		cur_lsd = lsd_dict[rir_file][0]
		stride = int(er / len(cur_lsd))

		for i, lsd in enumerate(cur_lsd):
			x.append(((i + 1) * stride))
			y.append(lsd)  # / max(cur_lsd))

		kn = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing").knee
		cut_dict[rir_file] = float(kn)

		ax.plot(x, y / np.max(y), 'o-', color='darkorange')
		plt.axvline(kn, linestyle='--', color='red')

		if img_path is not None:
			# 'images/lsd/'
			fig.savefig(img_path + rir_file.replace('.wav', '.pdf'))

		if cut_dict_path is not None:
			# 'audio/armodels/
			model_store(cut_dict_path + 'cut_idx_kl.json', cut_dict)

	return cut_dict


def rir_trim(rir_path, cut_dict, fade_length=128, save_path=None):

	trimmed_rir_dict = {}

	rir_files = os.listdir(rir_path)

	for rir_file in rir_files:

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		cut_idx = int(cut_dict[rir_file])

		if rir.ndim > 1:
			trimmed_rir = rir[:, :cut_idx]

		else:
			trimmed_rir = rir[:cut_idx]

		print(len(trimmed_rir.T))

		trimmed_rir_faded = trimmed_rir * cosine_fade(len(trimmed_rir.T), fade_length)

		trimmed_rir_dict[rir_file] = trimmed_rir_faded

		# fig = plt.figure()
		# ax = fig.add_subplot(1, 1, 1)
		#
		# ax.plot(trimmed_rir[0])

		if save_path is not None:
			sf.write(save_path + rir_file, trimmed_rir_faded.T, sr)

	return trimmed_rir_dict


if __name__ == "__main__":
	frame_size = 512
	sr = 44100
	fade_factor = 4
	early_trim = 500

	rir_path = 'audio/input/chosen_rirs/'
	a_a, p_a, l_a = rir_psd_metrics(rir_path, sr, frame_size, fade_factor, early_trim)

	knee_save_path = 'images/lsd/'
	cut_dict_save_path = 'audio/armodels/'

	# arm_dict = np.load('audio/armodels/arm_dict_ms.npy', allow_pickle=True)[()]
	lsd_dict = np.load('audio/armodels/lsd_dict_ms.npy', allow_pickle=True)[()]

	cut_dict = rir_er_detection(rir_path, lsd_dict)

	trim_rir_save_path = 'audio/trimmed_rirs/'
	trim_rir_dict = rir_trim(rir_path, cut_dict, fade_length=128, save_path=trim_rir_save_path)