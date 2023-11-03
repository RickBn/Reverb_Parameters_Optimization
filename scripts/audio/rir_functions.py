from kneed import KneeLocator
from scipy.signal import find_peaks

from scripts.audio.DSPfunc import *
from scripts.audio.audio_manipulation import *
from scripts.audio.audio_metrics import *
from scripts.utils.dict_functions import save_or_merge


plt.switch_backend('agg')

def beaforming_ambisonic(rir_ambisonic: np.array, wall_idx_ambisonic: int = 0):
	return rir_ambisonic[wall_idx_ambisonic:wall_idx_ambisonic+1,:]

def get_rir_wall_reflections_ambisonic(rir_ambisonic: np.array):
	return rir_ambisonic[0:7,:]


def rir_psd_metrics(rir_path: str,
                    sr: float,
                    frame_size: int = 512,
                    fade_factor: int = 4,
                    early_trim: int = 500,
                    direct_offset: bool = False,
                    ms_encoding: bool = False,
                    save_path: str = None):
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

		if ms_encoding:
			rir = ms_matrix(rir)

		for r_i in rir:
			a_r = []
			p_r = []
			l_r = []

			if direct_offset:
				offset = np.argmax(r_i > 0.0025)
			else:
				offset = 0

			rir_first_msec = r_i[offset:er]

			# fig = plt.figure()
			# ax = fig.add_subplot(1, 1, 1)
			# ax.plot(normalize_audio(r_i[:er]))
			# plt.axvline(offset, linestyle='--', color='red')

			xpad = pad_windowed_signal(rir_first_msec, frame_size)

			for chunk in range(0, len(xpad) - frame_size, frame_size)[1:]:
				xchunk = xpad[:chunk]
				xfaded = xchunk * cosine_fade(len(xchunk), fade_length)
				ar = armodel(xfaded, 2 * math.floor(2 + sr / 1000))
				a_r.append(ar)

				rir_psd = ar2psd(ar, 2048)
				p_r.append(rir_psd)

			for psd_idx in range(1, len(p_r), 1):
				l_r.append(log_spectral_distance(p_r[psd_idx], p_r[psd_idx - 1]))

			a_a.append(a_r)
			p_a.append(p_r)
			l_a.append(l_r)

		arm_dict[rir_file] = a_a
		psd_dict[rir_file] = p_a
		lsd_dict[rir_file] = l_a

		if save_path is not None:
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			save_or_merge(save_path + '/arm_dict.npy', arm_dict)
			save_or_merge(save_path + '/psd_dict.npy', psd_dict)
			save_or_merge(save_path + '/lsd_dict.npy', lsd_dict)

	return arm_dict, psd_dict, lsd_dict


def rir_er_detection(rir_path, lsd_dict, early_trim=500, ms_encoding=False, img_path=None, cut_dict_path=None):
	cut_dict = {}
	offset_dict = {}

	rir_files = os.listdir(rir_path)

	for rir_file in rir_files:
		cut_idx_list = []
		offset_list = []

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		if ms_encoding:
			rir = ms_matrix(rir)[0]

		for idx, r_i in enumerate(rir):
			er = int((sr * 0.001) * early_trim)

			offset = np.argmax(r_i) - 128
			offset_list.append(offset)

			print(str(offset))
			x = []
			y = []
			cur_lsd = lsd_dict[rir_file][idx]
			stride = 512#int(er / len(cur_lsd))

			for i, lsd in enumerate(cur_lsd):
				x.append(offset + ((i + 1) * stride))
				y.append(lsd)  # / max(cur_lsd))

			kn = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing").knee
			cut_idx_list.append(float(kn))

			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			ax.plot(r_i[:er])
			ax.plot(x, y, 'o-', color='darkorange')
			plt.axvline(kn, linestyle='--', color='red')
			plt.axvline(offset, linestyle='--', color='green')

			if img_path is not None:
				if not os.path.exists(img_path):
					os.makedirs(img_path)

				fig.savefig(img_path + rir_file.replace('.wav', '_') + str(idx) + '.pdf')
				plt.close(fig)

		cut_dict[rir_file] = cut_idx_list
		offset_dict[rir_file] = offset_list

		if cut_dict_path is not None:
			if not os.path.exists(cut_dict_path):
				os.makedirs(cut_dict_path)

			save_or_merge(cut_dict_path + 'cut_idx_kl.npy', cut_dict)
			save_or_merge(cut_dict_path + 'rir_offset.npy', offset_dict)

	return [cut_dict, offset_dict]


def rir_trim(rir_path, cut_dict, fade_length=256, save_path=None):
	trimmed_rir_dict = {}

	rir_files = os.listdir(rir_path)

	for rir_file in rir_files:

		rir, sr = sf.read(rir_path + rir_file)
		rir = rir.T

		cut_idx = int(np.max(cut_dict[rir_file]))

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
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			sf.write(save_path + rir_file, trimmed_rir_faded.T, sr)

	return trimmed_rir_dict


def remove_direct_from_rir(rir, fade_length=256):
	from scripts.audio.audio_manipulation import cosine_fade
	for ch in range(rir.shape[0]):
		peaks = find_peaks(rir[ch,:])

		direct_pos = peaks[0][0]

		print(f'Direct found at {direct_pos}')

		cos_fade = np.concatenate([np.zeros(direct_pos), cosine_fade(rir.shape[1] - direct_pos, fade_length, fade_out=False)])

		rir[ch, :] = rir[ch, :] * cos_fade

	return rir
