import os

from scripts.audio_functions.audio_manipulation import *


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
	rir = 'sdn_project'

	input_path = f'audio/input/chosen_rirs/stereo/{rir}/_todo/'
	armodel_path = f'audio/armodels/stereo/{rir}/'
	save_path = f'audio/trimmed_rirs/bin/{rir}/'

	batch_trim(input_path, armodel_path, save_path, trim_tail=False)

	input_path = 'audio/input/sounds/48/speech/_trimmed/'

	batch_loudnorm(input_path, -30)

	input_path = 'audio/input/sounds/48/speech/_trimmed/loudnorm/'
	input_name = 'speech'

	# /////////////////////////////////////////////////////////////////////////////
	rir_path = f'audio/input/chosen_rirs/HOA/{rir}/_done/'
	result_path = f'audio/results/HOA/{rir}/{input_name}/'

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

	batch_fft_convolve(input_path, result_file_names, rir_path, result_path,
	                   return_convolved=False, scale_factor=1.0, norm=False)

	# /////////////////////////////////////////////////////////////////////////////

	rir_path = f'audio/vst_rirs/stereo/{rir}/'
	result_path = f'audio/results/stereo/{rir}/{input_name}/late_only/'

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_late_fv.wav') for x in input_file_names]

	batch_fft_convolve(input_path, result_file_names, rir_path, result_path,
	                   return_convolved=False, scale_factor=1.0, norm=False)

	# /////////////////////////////////////////////////////////////////////////////

	rir_path = f'audio/trimmed_rirs/HOA/{rir}/'
	result_path = f'audio/results/HOA/{rir}/{input_name}/early_only/'

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_early_fv.wav') for x in input_file_names]

	batch_fft_convolve(input_path, result_file_names, rir_path, result_path,
	                   return_convolved=False, scale_factor=1.0, norm=False)

	# ///////////////////////////////////////////////////////////////////

	rir_path = f'audio/trimmed_rirs/bin/{rir}/'
	result_path = f'audio/results/bin/late_only/{rir}/{input_name}/'

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_late_bin.wav') for x in input_file_names]

	batch_fft_convolve(input_path, result_file_names, rir_path, result_path,
	                   return_convolved=False, scale_factor=1.0, norm=False)
