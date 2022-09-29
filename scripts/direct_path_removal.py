from scripts.audio_functions.audio_manipulation import *


def batch_trim(input_path: str, armodel_path: str, save_path=None, trim_tail: bool = True):
	rir_names = os.listdir(input_path)
	for rir in rir_names:
		rir_path = f'{input_path}{rir}/'
		cut_dict = np.load(f'{armodel_path}{rir}/cut_idx_kl.npy', allow_pickle=True)[()]
		offset_dict = np.load(f'{armodel_path}{rir}/rir_offset.npy', allow_pickle=True)[()]

		for audio_file in os.listdir(rir_path):

			audio, sr = sf.read(f'{rir_path}{audio_file}')
			audio = audio.T

			cut_idx = np.max(cut_dict[audio_file])
			offset = np.max(offset_dict[audio_file])
			trim_length = np.max(cut_idx - offset)

			no_fade = np.zeros(offset)
			fade = cosine_fade(len(audio.T) - offset, int(trim_length), trim_tail)
			trimmed_file = audio * np.concatenate([no_fade, fade], axis=0)

			current_save_path = f'{save_path}{rir}/'
			if not os.path.exists(current_save_path):
				os.makedirs(current_save_path)

			sf.write(f'{current_save_path}{audio_file}', trimmed_file.T, sr)


if __name__ == "__main__":
	sr = 48000

	input_path = 'audio/input/chosen_rirs/stereo/'
	armodel_path = 'audio/armodels/stereo/'
	save_path = 'audio/trimmed_rirs/bin/'

	batch_trim(input_path, armodel_path, save_path, trim_tail=False)

	input_path = 'audio/input/sounds/48/speech/_trimmed/'

	batch_loudnorm(input_path, -30)

	input_path = 'audio/input/sounds/48/speech/_trimmed/loudnorm'
	input_name = 'speech'

	# /////////////////////////////////////////////////////////////////////////////
	for rir in os.listdir('audio/input/chosen_rirs/HOA/'):

		rir_path = f'audio/input/chosen_rirs/HOA/{rir}/_done/'
		result_path = f'audio/results/HOA/{rir}/bf4/{input_name}/'

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

		rir_path = f'audio/trimmed_rirs/HOA/{rir}/_done/'
		result_path = f'audio/results/HOA/{rir}/bf4/{input_name}/early_only/'

		input_file_names = os.listdir(input_path)
		result_file_names = [x.replace(".wav", '_early_fv.wav') for x in input_file_names]

		batch_fft_convolve(input_path, result_file_names, rir_path, result_path,
		                   return_convolved=False, scale_factor=1.0, norm=False)

	# ///////////////////////////////////////////////////////////////////

	rir_path = 'audio/trimmed_rirs/bin/'
	result_path = f'audio/results/bin/late_only/'

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_late_bin.wav') for x in input_file_names]

	for rir in os.listdir(rir_path):
		save_path = f'{result_path}{rir}/{input_name}/'
		batch_fft_convolve(input_path, result_file_names, f'{rir_path}{rir}/', save_path,
		                   return_convolved=False, scale_factor=1.0, norm=False)
