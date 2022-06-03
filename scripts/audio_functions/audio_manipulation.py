import numpy as np
import os

import soundfile as sf
import librosa
import librosa.display
from scipy.signal import chirp

from scripts.audio_functions.pedalboard_functions import *


def pd_highpass_filter(audio: np.ndarray, order: int, sr: int, cutoff=20.0):

    filter = np.array([pedalboard.HighpassFilter(cutoff_frequency_hz=cutoff),
                       pedalboard.HighpassFilter(cutoff_frequency_hz=cutoff)])

    if audio.ndim == len(filter):
        for ch, fil in enumerate(filter):
            for i in range(0, order):
                audio[ch] = fil(audio[ch], sr)

    return audio


def normalize_audio(audio: np.ndarray, scale_factor=1.0) -> np.ndarray:

    norm_audio = np.divide(audio, np.max(abs(audio))) * scale_factor

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


def b_format_to_stereo(filename: str) -> np.ndarray:
    wxyz, sr = sf.read(filename)
    ms = np.array([wxyz[:, 0], wxyz[:, 2]])
    lr = ms_matrix(ms)

    return lr, sr


def batch_convolve(input_files, convolution_array, rir_folder, sr, scale_factor=1.0, save_file=False):

    convolved = []

    for idx, wav in enumerate(input_files):
        for dir, conv in enumerate(convolution_array):
            convoluted_ref = conv(wav, sr)

            convoluted_ref = pd_highpass_filter(convoluted_ref, 3, sr)

            convoluted_ref = convoluted_ref / np.max(abs(convoluted_ref))
            convoluted_ref = convoluted_ref * scale_factor

            convolved.append(convoluted_ref)

            if save_file:
                input_sounds = os.listdir('audio/results/' + rir_folder[dir])[:-1]

                sf.write('audio/results/' + rir_folder[dir] + '/' + input_sounds[idx] + '/' + input_sounds[
                    idx] + '_ref.wav', convoluted_ref.T, sr)

    return convolved


def cosine_fade(signal_length: int, fade_length: int, fade_out=True):
    t = np.linspace(0, np.pi, fade_length)
    no_fade = np.ones(signal_length - fade_length)

    if fade_out is True:
        return np.concatenate([no_fade, (np.cos(t) + 1) * 0.5], axis=0)
    else:
        return np.concatenate([((-np.cos(t)) + 1) * 0.5, no_fade], axis=0)
