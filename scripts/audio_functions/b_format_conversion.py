import os

from scripts.audio_functions.audio_manipulation import *

path = 'D:/Università/Magistrale/TESI/IR_Database/Chosen Ones/'

for wav in os.listdir(path + 'B-format/'):
	lr, sr = b_format_to_stereo(path + 'B-format/' + wav)
	sf.write(path + 'Stereo/lr_' + wav, lr.T, sr)