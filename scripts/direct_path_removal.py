import os

from scripts.audio.audio_manipulation import *


def batch_trim(input_path: str, armodel_path: str, save_path=None, trim_tail: bool = True):
	cut_dict = np.load(f'{armodel_path}cut_idx_kl.npy', allow_pickle=True)[()]
	offset_dict = np.load(f'{armodel_path}rir_offset.npy', allow_pickle=True)[()]

	for audio_file in directory_filter(input_path):
		audio, sr = sf.read(f'{input_path}{audio_file}')
		audio = audio.T

		cut_idx = np.max(cut_dict[audio_file])
		offset = np.max(offset_dict[audio_file])
		trim_length = np.max(cut_idx - offset)

		no_fade = np.zeros(offset)
		fade = cosine_fade(len(audio.T) - offset, int(trim_length), trim_tail)
		trimmed_file = audio * np.concatenate([no_fade, fade], axis=0)

		current_save_path = save_path
		if not os.path.exists(current_save_path):
			os.makedirs(current_save_path)

		sf.write(f'{current_save_path}{audio_file}', trimmed_file.T, sr)


if __name__ == "__main__":
	sr = 48000
	rir = 'spergair'
	input_path = f'audio/input/chosen_rirs/stereo/{rir}/_todo/'
	armodel_path = f'audio/armodels/stereo/{rir}/'
	save_path = f'audio/trimmed_rirs/bin/{rir}/'

	batch_trim(input_path, armodel_path, save_path, trim_tail=False)

	input_path = 'audio/input/sounds/48/speech/_trimmed/'

	batch_loudnorm(input_path, -30)

	# CREATE PHASE SHIFTED RIR VERSION ///////////////////////////////////////////
	# rir = 'sdn_project'
	# vst_rir_path = f'audio/vst_rirs/stereo/{rir}/'
	# phase_shift_ir = 'audio/input/filters/phase_rot/90_deg_phaseShift.wav'
	#
	# batch_phase_shift(vst_rir_path, phase_shift_ir, f'{vst_rir_path}shifted/')

	# // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //
	# HOA ////////////////////////////////////////////////////////////////////////