from kneed import KneeLocator
from scipy.signal import find_peaks
import dawdreamer as daw
import librosa

from scripts.audio.DSPfunc import *
from scripts.audio.audio_manipulation import *
from scripts.audio.audio_metrics import *
from scripts.utils.dict_functions import save_or_merge


plt.switch_backend('agg')

n_walls = 6

# Vector3f MathUtils::dirVector(Point3d& a, Point3d& b)
# {
# 	return Vector3f( a.x - b.x, a.y - b.y, a.z - b.z );
# }
#
# Point3d MathUtils::reflectionPoint(Point3d a, Point3d b, char reflAxis, float wallPosition)
# {
# 	Vector3f direction;
# 	float positionParam;
#
# 	switch (reflAxis)
# 	{
# 	case 'x':
# 		a.x = (2 * wallPosition) - a.x;
# 		direction = MathUtils::dirVector(a, b);
# 		positionParam = (wallPosition - a.x) / direction.x();
# 		return { wallPosition, a.y + direction.y() * positionParam, a.z + direction.z() * positionParam };
# 		break;
#
# 	case 'y':
# 		a.y = (2 * wallPosition) - a.y;
# 		direction = MathUtils::dirVector(a, b);
# 		positionParam = (wallPosition - a.y) / direction.y();
# 		return { a.x + direction.x() * positionParam, wallPosition, a.z + direction.z() * positionParam };
# 		break;
#
# 	case 'z':
# 		a.z = (2 * wallPosition) - a.z;
# 		direction = MathUtils::dirVector(a, b);
# 		positionParam = (wallPosition - a.z) / direction.z();
# 		return { a.x + direction.x() * positionParam, a.y + direction.y() * positionParam, wallPosition };
# 		break;
# 	}
#
# 	return { 0, 0, 0 };
#
# }


def get_azel_2points_3d(a_x, a_y, a_z, b_x, b_y, b_z):
	dir_vec = np.array([a_x - b_x, a_y - b_y, a_z - b_z])

	# Così per l'orientamento degli assi usato dal plugin
	az = np.degrees(np.arctan2(dir_vec[0], dir_vec[2]))
	el = np.degrees(np.arcsin((dir_vec[1]) / np.linalg.norm(dir_vec)))

	# Così se fosse orientato con gli assi normalmente
	# XsqPlusYsq = dir_vec[0] ** 2 + dir_vec[1] ** 2
	# az2 = np.degrees(np.arctan2(dir_vec[1], dir_vec[0]))
	# el2 = np.degrees(np.arctan2(dir_vec[2], np.sqrt(XsqPlusYsq)))

	return az, el


def get_refl_angle(fixed_params, wall_idx_ambisonic, wall_order):
	wall = wall_order[wall_idx_ambisonic]
	wall = wall.split('_')
	refl_axis = wall[0]
	wall_position = int(wall[1])

	source_x = fixed_params['dimensions_x_m'] * fixed_params['source_x']
	source_y = fixed_params['dimensions_y_m'] * fixed_params['source_y']
	source_z = fixed_params['dimensions_z_m'] * fixed_params['source_z']

	listener_x = fixed_params['dimensions_x_m'] * fixed_params['listener_x']
	listener_y = fixed_params['dimensions_y_m'] * fixed_params['listener_y']
	listener_z = fixed_params['dimensions_z_m'] * fixed_params['listener_z']

	if refl_axis == 'x':
		wall_position = fixed_params['dimensions_x_m'] * wall_position
		source_x = (2 * wall_position) - source_x

	elif refl_axis == 'y':
		wall_position = fixed_params['dimensions_y_m'] * wall_position
		source_y = (2 * wall_position) - source_y

	elif refl_axis == 'z':
		wall_position = fixed_params['dimensions_y_m'] * wall_position
		source_z = (2 * wall_position) - source_z

	return get_azel_2points_3d(source_x, source_y, source_z, listener_x, listener_y, listener_z)


def beaforming_ambisonic(beamformer, engine, fixed_params, wall_idx_ambisonic: int = 0, wall_order=[], length: int = 144000, window: bool = True, fade_length: int = 512):
	from scripts.audio.audio_manipulation import cosine_fade

	azimuth, elevation = get_refl_angle(fixed_params, wall_idx_ambisonic, wall_order)

	# 5: azimuth
	beamformer.set_parameter(5, (azimuth/360)+0.5)
	# 6: elevation
	beamformer.set_parameter(6, (elevation/180)+0.5)

	if beamformer.get_parameter_text(5) != azimuth or beamformer.get_parameter_text(6) != elevation:
		pass

	engine.render(length)
	y = engine.get_audio()

	if window:
		peaks = find_peaks(y[0, :])

		refl_pos = peaks[0][0]

		cos_fade = np.concatenate([np.ones(refl_pos),
								   (cosine_fade(y.shape[1] - refl_pos, fade_length, fade_out=False)-1)*(-1)])

		y = y * cos_fade

	return y


def get_rir_wall_reflections_ambisonic(rir_ambisonic: np.array, fixed_params, wall_order=[], sr: int = 48000, order: int = 4):

	rir_beamform = np.zeros((n_walls+1, rir_ambisonic.shape[1]))

	# Omni channel
	rir_beamform[0, :] = rir_ambisonic[0, :]

	buffer_size = 128
	engine = daw.RenderEngine(sr, buffer_size)

	beamformer = engine.make_plugin_processor("beamformer", "C:\Program Files\SPARTA\VST\sparta_beamformer.dll")

	beamformer.set_bus(rir_ambisonic.shape[0], 1)
	# 0: order
	max_order = 7
	beamformer.set_parameter(0, (order - 1) / (max_order - 1))
	# a = beamformer.get_parameter_text(0)
	# 1: channel order -> ACN
	beamformer.set_parameter(1, 0)
	# 2: normalisation type -> SN3D
	beamformer.set_parameter(2, 0.5)
	# 3: beam type -> Hyper-Card TODO: ok?
	beamformer.set_parameter(3, 0.5)
	# 4: num beams -> 1
	beamformer.set_parameter(4, 0.01)

	playback = engine.make_playback_processor('input',  np.zeros((rir_ambisonic.shape[0], 0)))
	playback.set_data(rir_ambisonic)

	graph = [
		(playback, []),
		(beamformer, ['input'])
	]

	engine.load_graph(graph)

	for w in range(1, n_walls+1):
		rir_beamform[w, :] = beaforming_ambisonic(beamformer, engine, fixed_params=fixed_params, wall_idx_ambisonic=w,
												  wall_order=wall_order, length=rir_ambisonic.shape[1]/sr)

	return rir_beamform, beamformer, engine, playback


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
