import os
from kneed import KneeLocator

from scripts.utils.json_functions import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.audio_metrics import *

#plt.switch_backend('agg')

def direct_sound_eq():
	rir_path = 'audio/input/chosen_rirs/'

	arm_dict = {}
	psd_dict = {}
	lsd_dict = {}

	frame_size = 512
	fade_length = (frame_size // 4)
	er = int((44100 * 0.001) * 500)

	rir_files = os.listdir(rir_path)

	for idx, rir_file in enumerate(rir_files):

		a_a = []
		p_a = []
		l_a = []

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir_ms = ms_matrix(rir)
		er = int((sr * 0.001) * 500)

		for rir in rir_ms:
			a_ms = []
			p_ms = []
			l_ms = []

			rir_first_ms = rir[:er]

			if len(rir_first_ms) % frame_size != 0:
				xpad = np.append(rir_first_ms, np.zeros(frame_size))
			else:
				xpad = rir_first_ms

			for chunk in range(0, len(xpad)-frame_size, frame_size)[1:]:

				xchunk = xpad[:chunk]
				xfaded = xchunk * cosine_fade(len(xchunk), fade_length)
				ar = armodel(xfaded, 2 * math.floor(2 + sr / 1000))
				a_ms.append(ar)

				rir_psd = ar2psd(ar, 2048)
				p_ms.append(rir_psd)

			for psd_idx in range(1, len(p_ms), 1):
				l_ms.append(log_spectral_distance(p_ms[psd_idx], p_ms[psd_idx-1]))

			a_a.append(a_ms)
			p_a.append(p_ms)
			l_a.append(l_ms)

		arm_dict[rir_file] = a_a
		psd_dict[rir_file] = p_a
		lsd_dict[rir_file] = l_a


	np.save('audio/armodels/arm_dict_ms.npy', arm_dict)
	np.save('audio/armodels/psd_dict_ms.npy', psd_dict)
	np.save('audio/armodels/lsd_dict_ms.npy', lsd_dict)

	#arm_dict = np.load('audio_functions/armodels/arm_dict.npy', allow_pickle=True)

	#lsd_dict = np.load('audio/armodels/lsd_dict_ms.npy', allow_pickle=True)[()]

	cut_dict = {}
	for rir_file in rir_files:

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir = ms_matrix(rir)[0]

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)

		ax.plot(rir[:er] / max(rir[:er]))

		x = []
		y = []
		cur_lsd = lsd_dict[rir_file][0]
		stride = int(er / len(cur_lsd))
		for i, lsd in enumerate(cur_lsd):
			x.append(((i+1)*stride))
			y.append(lsd) # / max(cur_lsd))

		kn = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing").knee
		cut_dict[rir_file] = float(kn)

		ax.plot(x, y / np.max(y), 'o-', color='darkorange')
		plt.axvline(kn, linestyle='--', color='red')

		#fig.savefig('images/lsd/' + rir_file.replace('.wav', '.png'))

		# idx = lsd_dict[rir_file][0].index(np.max(lsd_dict[rir_file][0][5:]))
		# cut_dict[rir_file] = (idx + 1) * frame_size
		# plt.plot(lsd_dict[rir_file][0] / np.max(lsd_dict[rir_file][0]))

	#model_store('audio/armodels/cut_idx_ms.json', cut_dict)
	model_store('audio/armodels/cut_idx_kl.json', cut_dict)

	#cut_dict = model_load('audio_functions/armodels/cut_idx_ms.json')

	rir_eq_coeffs = {}
	for rir_file in rir_files:
		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		rir_ms = ms_matrix(rir)
		cut_idx = int(cut_dict[rir_file])

		ar = []
		for rir in rir_ms:
			rir_cut = rir[:cut_idx]
			rir_faded = rir_cut * cosine_fade(len(rir_cut), fade_length)
			ar.append(armodel(rir_faded, 2 * math.floor(2 + sr / 1000)))

		rir_eq_coeffs[rir_file] = ar

	#np.save('audio/armodels/rir_eq_coeffs_ms.npy', rir_eq_coeffs)
	np.save('audio/armodels/rir_eq_coeffs_kl.npy', rir_eq_coeffs)
