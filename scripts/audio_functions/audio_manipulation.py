from typing import Tuple, Any

import numpy as np
import os

import soundfile as sf
import librosa
import librosa.display
from numpy import ndarray
from scipy.signal import chirp
import matplotlib.pyplot as plt

from scripts.audio_functions.pedalboard_functions import *


def normalize_audio(audio: np.ndarray, scale_factor=1.0, nan_check=False) -> np.ndarray:

    norm_audio = np.divide(audio, np.max(abs(audio))) * scale_factor

    if nan_check:
        norm_audio[np.isnan(norm_audio)] = 0

    return norm_audio


def normalize_multidimensional(input: np.ndarray, byrow: bool = True) -> np.ndarray:

    if byrow:
        norm = np.divide(input.T, np.max(abs(input), axis=1)).T

    else:
        norm = np.divide(input, np.max(abs(input), axis=0))

    return norm


def ms_matrix(stereo_audio: np.ndarray) -> np.ndarray:
    inv_sq = 1 / np.sqrt(2)
    mid = (stereo_audio[0] + stereo_audio[1]) * inv_sq
    side = (stereo_audio[0] - stereo_audio[1]) * inv_sq

    return np.array([mid, side])


def b_format_to_stereo(filename: str) -> Tuple[ndarray, Any]:
    wxyz, sr = sf.read(filename)
    ms = np.array([wxyz[:, 0], wxyz[:, 2]])
    lr = ms_matrix(ms)

    return lr, sr


def prepare_batch_convolve(rir_path, mix=1.0):

    convolution_array = []

    for idx, rir_file in enumerate(os.listdir(rir_path)):
        convolution_array.append(pedalboard.Convolution(rir_path + rir_file, mix))

    return convolution_array


def prepare_batch_input_stereo(input_audio_path):

    audio_file = []
    input_audio_file = os.listdir(input_audio_path)

    for idx, wav in enumerate(input_audio_file):
        audio_file.append(sf.read(input_audio_path + input_audio_file[idx])[0])

        if audio_file[idx].ndim is 1:
            audio_file[idx] = np.stack([audio_file[idx], audio_file[idx]])

    return audio_file


def batch_convolve(input_files, convolution_array, input_files_names, rir_path, sr, scale_factor=1.0,
                   norm=True, save_path=None):

    convolved = []

    rir_files_names = os.listdir(rir_path)

    for idx, input_sound in enumerate(input_files):
        for dir, conv in enumerate(convolution_array):

            convolved_input = conv(input_sound, sr)

            convolved_input = pd_highpass_filter(convolved_input, 3, sr)

            if norm:
                convolved_input = normalize_audio(convolved_input) * scale_factor

            convolved.append(convolved_input)

            if save_path is not None:
                sp = save_path + '/' + rir_files_names[dir].replace('.wav', "") + '/'

                if not os.path.exists(sp):
                    os.makedirs(sp)

                sf.write(sp + input_files_names[idx], convolved_input.T, sr)

    return convolved


def pad_signal(input_signal: np.ndarray, n_dim: int, pad_length: int):

    padded_signal = np.concatenate([input_signal, np.zeros((n_dim, pad_length))], axis=1)

    return padded_signal


def pad_windowed_signal(input_signal: np.array, window_size: int):

    if len(input_signal) % window_size != 0:
        xpad = np.append(input_signal, np.zeros(window_size))
    else:
        xpad = input_signal

    return xpad


def cosine_fade(signal_length: int, fade_length: int, fade_out=True):
    t = np.linspace(0, np.pi, fade_length)
    no_fade = np.ones(signal_length - fade_length)

    if fade_out is True:
        return np.concatenate([no_fade, (np.cos(t) + 1) * 0.5], axis=0)
    else:
        return np.concatenate([((-np.cos(t)) + 1) * 0.5, no_fade], axis=0)
