import os
import numpy as np
import soundfile as sf

path = 'audio_functions/input/REVelation/'

for wav in os.listdir(path + '-24/'):
	f, sr = sf.read(path + '-24/' + wav)
	f = f / np.max(abs(f))
	sf.write(path + 'norm/ref_' + wav, f, sr)


path = '_done/audio_functions/results/'
test_folders = os.listdir(path)

folder_to_trim = '/drums/'
folder_to_trim = '/speech/'

for test in test_folders:
	rir_folders = os.listdir(path + test)
	for rir in rir_folders:
		stimuli = os.listdir(path + test + '/' + rir + folder_to_trim)
		for i, wav in enumerate(stimuli):
			f, sr = sf.read(path + test + '/' + rir + folder_to_trim + wav)
			f_trimmed = f[:(sr*11), :]
			sf.write(path + test + '/' + rir + folder_to_trim + wav, f_trimmed, sr)

# path = '_done/audio_functions/results/Algorithmic/REVelation/'
# test_folders = os.listdir(path)[:-1]
# test_folders.pop(0)
#
# stimulus = 'speech'
# stimuli = os.listdir(path + stimulus + '/')
# lengths = []
#
# for stim in stimuli:
# 	f, sr = sf.read(path + stimulus + '/' + stim)
# 	lengths.append(f.shape[0])
#
# max_len = np.max(lengths)
# f, sr = sf.read(path + stimulus + '/' + stimulus + '_ref.wav')
#
# f_new = np.concatenate([f, np.zeros((max_len - f.shape[0], 2))])
# sf.write(path + stimulus + '/' + stimulus + '_ref_new.wav', f_new, sr)


path = '_done/audio_functions/input/chosen_rirs/'
rir_files = os.listdir(path)

for wav in rir_files:
	f, sr = sf.read(path + wav)
	if f.shape[0] > sr * 3:
		f_trimmed = f[:(sr * 3), :]
		sf.write(path + wav, f_trimmed, sr)


path = '_done/audio_functions/results/'

for mix in os.listdir(path):
	mix_folder = path + mix + '/'
	for rir_type in os.listdir(mix_folder):
		type_folder = mix_folder + rir_type + '/'
		for rir in os.listdir(type_folder):
			rir_folder = type_folder + rir + '/'
			for sound in os.listdir(rir_folder)[:-1]:
				sound_folder = rir_folder + sound + '/'
				for audio_file in os.listdir(sound_folder):
					f, sr = sf.read(sound_folder + audio_file)
					f = f * 0.81
					sf.write(sound_folder + audio_file, f, sr)

path = '_done/final/eq_rir_ms/'

for type in os.listdir(path):
	type_dir = path + type + '/'
	for rir in os.listdir(type_dir):
		rir_folder = type_dir + rir + '/'
		for sound in os.listdir(rir_folder)[:-1]:
			sound_folder = rir_folder + sound + '/'
			for audio_file in os.listdir(sound_folder)[:-1]:
				f, sr = sf.read(sound_folder + audio_file)
				f = f.T
				f[0] = f[0] / np.max(abs(f[0]))
				f[1] = f[1] / np.max(abs(f[1]))
				f = f * 0.81
				sf.write(sound_folder + audio_file, f.T, sr)