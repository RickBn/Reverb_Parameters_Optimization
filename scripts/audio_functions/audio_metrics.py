from typing import List
import numpy as np
from scripts.utils.array_functions import *

import librosa
import librosa.display


def mel_spectrogram_l1_distance(h_1: np.ndarray,
                                h_2: np.ndarray,
                                sr: int,
                                fft_sizes: List[int] = (256, 512, 1024, 2048, 4096),
                                trim: bool = True,
                                by_row: bool = True) -> float:

    if not by_row:
        h_1 = h_1.T
        h_2 = h_2.T

    array_dimensions_match_check(h_1, h_2)

    h_1, h_2 = enlist_1D_array(h_1, h_2)

    distance = np.zeros(h_1.shape[0])

    for idx in range(0, h_1.shape[0]):
        for n_fft in fft_sizes:
            w_1 = librosa.feature.melspectrogram(y=h_1[idx], sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25),
                                                 center=False)
            w_1 = librosa.power_to_db(w_1, ref=np.max)

            w_2 = librosa.feature.melspectrogram(y=h_2[idx], sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25),
                                                 center=False)
            w_2 = librosa.power_to_db(w_2, ref=np.max)

            if trim:
                w_1[np.where(w_1 < -60.0)] = -60.0
                w_2[np.where(w_2 < -60.0)] = -60.0

            distance[idx] += np.mean(abs(w_1 - w_2))

    mean_distance = float(np.mean(distance))

    return mean_distance


def mfcc_l1_distance(h_1: np.ndarray, h_2: np.ndarray, sr: int,
                     fft_sizes: List[int] = (256, 512, 1024, 2048, 4096)) -> float:
    distance = 0.0

    func = np.sqrt(np.arange(0.0, 1.0, 0.05))
    func = (func * 10 + 1)
    func = func.reshape(20, 1)

    for n_fft in fft_sizes:
        w_1 = librosa.feature.melspectrogram(y=h_1, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25), center=False)
        w_1 = librosa.power_to_db(w_1, ref=np.max)
        mfcc_1 = librosa.feature.mfcc(S=w_1, n_mfcc=20)

        w_2 = librosa.feature.melspectrogram(y=h_2, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25), center=False)
        w_2 = librosa.power_to_db(w_2, ref=np.max)

        mfcc_2 = librosa.feature.mfcc(S=w_2, n_mfcc=20)

        mfcc_1 = mfcc_1 * func
        mfcc_2 = mfcc_2 * func

        distance += np.mean(abs(mfcc_1 - mfcc_2))

    return distance


def energy_decay_relief(h: np.ndarray, win_ms: int, sr: int, trim: bool = True):
    w_l = int(win_ms * 0.001 * sr)
    frame_len_pow = np.ceil(np.log2(abs(w_l)))
    frame_len = int(2**frame_len_pow)

    S = librosa.stft(y=h, n_fft=frame_len, hop_length=int(frame_len * 0.25), win_length=frame_len)

    n_bins, n_frames = S.shape
    energy = S * np.conjugate(S)
    edr = []

    for b in range(0, n_bins):
        edr.append(np.flip(np.cumsum(np.flip(energy[b, :]))))

    edr_db = 10 * np.log10(abs(np.array(edr)) + np.finfo(float).eps)

    if trim:
        edr_db[np.where(edr_db < -60.0)] = -60.0

    return edr_db


def edr_l1_distance(h_1: np.ndarray, h_2: np.ndarray, sr: int) -> float:

    e_1 = energy_decay_relief(h_1, 30, sr)
    e_2 = energy_decay_relief(h_2, 30, sr)

    return np.mean(abs(e_1 - e_2))


def env_l1_distance(h_1: np.ndarray, h_2: np.ndarray) -> float:

    l = np.max([len(h_1), len(h_2)])
    lam = -np.log(0.1) / l

    n = np.arange(0, l, 1)
    exp_d = np.e**(-lam * n)

    p_1 = h_1 * np.conjugate(h_1)
    p_2 = h_2 * np.conjugate(h_2)

    return np.mean(abs(p_1 - p_2)*exp_d)


def log_spectral_distance(p1: np.ndarray, p2: np.ndarray):

    return np.sqrt(np.mean((10 * np.log10(p1 / p2))**2))




# func = np.sqrt(np.arange(0.0, 1.0, 0.05))
# func = (func * 10 + 1)
# func = func.reshape(20, 1)
#
# h_1 = reference_audio
# n_fft = 2048
#
# w_1 = librosa.feature.melspectrogram(y=h_1[0], sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25))
# w_1 = librosa.power_to_db(w_1, ref=np.max)
# mfcc = librosa.feature.mfcc(S=w_1, n_mfcc=20)
# plt.bar(np.arange(0,20,1), np.mean(mfcc * func, axis=1))

# for wav in os.listdir(path)[1:]:
# 	ms = b_format_to_ms_stereo(path + '/' + wav)
# 	lr = ms_matrix(ms)
# 	lr = lr / np.max(abs(lr))
# 	sf.write('audio_functions/input/' + wav, lr.T, sr)