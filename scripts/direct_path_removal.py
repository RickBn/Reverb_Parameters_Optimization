import os

import numpy as np
from typing import List

from scripts.audio_functions.audio_manipulation import *


def batch_trim(audio_path: str, trim_length: int, trim_tail=True, save_path=None) -> List[np.ndarray]:
	audio_files = os.listdir(audio_path)
	trimmed_files = []

	for audio_file in audio_files:

		audio, sr = sf.read(audio_path + audio_file)
		audio = audio.T

		trimmed_file = audio * cosine_fade(len(audio.T), trim_length, trim_tail)
		trimmed_files.append(trimmed_file)

		if save_path is not None:
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			sf.write(save_path + 'nd_' + audio_file, trimmed_file.T, sr)

	return trimmed_files


if __name__ == "__main__":
	merged_rirs_path = 'audio/merged_rirs/'
	save_path = 'audio/nd_merged_rirs/'

	sr = 44100
	trim_length = int(5 * sr * 0.001)

	nd_merged = batch_trim(audio_path=merged_rirs_path, trim_length=trim_length, trim_tail=False, save_path=save_path)

	input_path = 'audio/input/sounds/'
	save_path = 'audio/nd_results/'

	conv = prepare_batch_convolve(merged_rirs_path, 1.0)

	convolution = prepare_batch_convolve(merged_rirs_path, mix=1.0)
	audio_file = prepare_batch_input_stereo(input_path)

	input_file_names = os.listdir(input_path)
	result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

	batch_convolve(audio_file,
					convolution,
					result_file_names,
					merged_rirs_path,
					sr,
					1.0,
					norm=False,
					save_path=save_path)
