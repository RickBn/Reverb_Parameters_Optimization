from typing import List
import numpy as np
from scripts.utils.array_functions import *
# from scripts.audio.reverb_features import rev_tr
import scipy

import librosa
import librosa.display
from scipy.fft import rfft


def mel_spectrogram_l1_distance(h_1: np.ndarray,
                                h_2: np.ndarray,
                                sr: int,
                                # fft_sizes: List[int] = (256, 512, 1024, 2048, 4096),
                                fft_sizes: List[int] = (32, 64, 128, 256, 512),
                                trim: bool = True,
                                by_row: bool = True) -> float:

    if not by_row:
        h_1 = h_1.T
        h_2 = h_2.T

    array_dimensions_match_check(h_1, h_2)

    h_1, h_2 = enlist_1D_array(h_1, h_2)

    h_1_nonzero = np.nonzero(h_1[0,:])
    start_idx = np.min(h_1_nonzero)
    last_idx = np.max(h_1_nonzero)
    h_1 = h_1[:, start_idx:last_idx]
    h_2 = h_2[:, start_idx:last_idx]

    distance = np.zeros((h_1.shape[0], len(fft_sizes)))

    for idx in range(0, h_1.shape[0]):
        for n, n_fft in enumerate(fft_sizes):
            a = np.concatenate([h_1[idx], np.zeros(np.max((0, n_fft - h_1.shape[1])))])
            w_1 = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25),
                                                 center=False)
            w_1 = librosa.power_to_db(w_1, ref=np.max)

            a = np.concatenate([h_2[idx], np.zeros(np.max((0, n_fft - h_2.shape[1])))])
            w_2 = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25),
                                                 center=False)
            w_2 = librosa.power_to_db(w_2, ref=np.max)

            if trim:
                w_1[np.where(w_1 < -60.0)] = -60.0
                w_2[np.where(w_2 < -60.0)] = -60.0

            # distance[idx] += np.mean(abs(w_1 - w_2))
            distance[idx, n] = np.mean(abs(w_1 - w_2))

    mean_distance = float(np.mean(distance, 1))

    return mean_distance


def mfcc_l1_distance(h_1: np.ndarray, h_2: np.ndarray, sr: int, n_mfcc: int = 20,
                     fft_sizes: List[int] = (256, 512, 1024, 2048, 4096),
                     fmax: float = 24000) -> float:
    distance = 0.0

    # func = np.sqrt(np.arange(0.0, 1.0, 0.05))
    # func = (func * 10 + 1)
    # func = func.reshape(n_mfcc, 1)
    func = np.sqrt(np.linspace(0.0, 1.0, n_mfcc))
    func = (func * (n_mfcc / 2) + 1)
    func = func.reshape(n_mfcc, 1)

    for n_fft in fft_sizes:
        w_1 = librosa.feature.melspectrogram(y=h_1, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25), center=False,
                                             fmax=fmax)
        w_1 = librosa.power_to_db(w_1, ref=np.max)
        mfcc_1 = librosa.feature.mfcc(S=w_1, n_mfcc=n_mfcc)

        w_2 = librosa.feature.melspectrogram(y=h_2, sr=sr, n_fft=n_fft, hop_length=int(n_fft * 0.25), center=False,
                                             fmax=fmax)
        w_2 = librosa.power_to_db(w_2, ref=np.max)
        mfcc_2 = librosa.feature.mfcc(S=w_2, n_mfcc=n_mfcc)

        mfcc_1 = mfcc_1 * func
        mfcc_2 = mfcc_2 * func

        distance += np.mean(abs(mfcc_1 - mfcc_2))

    return distance


def energy_decay_relief(h: np.ndarray, win_ms: int, sr: int, trim: bool = True, mel: bool = False, fmax: float = 24000):
    w_l = int(win_ms * 0.001 * sr)
    frame_len_pow = np.ceil(np.log2(abs(w_l)))
    frame_len = int(2**frame_len_pow)

    if mel:
        S = librosa.feature.melspectrogram(y=h, sr=sr, n_fft=frame_len, hop_length=int(frame_len * 0.25), center=False,
                                           fmax=fmax, n_mels=6)
        # S = librosa.feature.melspectrogram(y=h, sr=sr, n_fft=frame_len, hop_length=int(frame_len * 0.25), center=False,
                                           # fmax=fmax, n_mels=6, window=scipy.signal.windows.hann, win_length=320)
    else:
        S = librosa.stft(y=h, n_fft=frame_len, hop_length=int(frame_len * 0.25), win_length=frame_len)

    n_bins, n_frames = S.shape
    # energy = S * np.conjugate(S)
    energy = abs(S)
    edr = []

    for b in range(0, n_bins):
        edr.append(np.flip(np.cumsum(np.flip(energy[b, :]))))

    edr_db = 10 * np.log10(abs(np.array(edr)) + np.finfo(float).eps)

    if trim:
        edr_db[np.where(edr_db < -60.0)] = -60.0

    return edr_db


def edr_l1_distance(h_1: np.ndarray, h_2: np.ndarray, sr: int, win_ms: float = 30, mel: bool = False, fmax: float = 24000) -> float:

    e_1 = energy_decay_relief(h_1, win_ms, sr, mel=mel, fmax=fmax)
    e_2 = energy_decay_relief(h_2, win_ms, sr, mel=mel, fmax=fmax)

    return np.mean(abs(e_1 - e_2))
    # return np.sum(abs(e_1 - e_2)) / np.sum(abs(e_1))


def env_l1_distance(h_1: np.ndarray, h_2: np.ndarray):

    l = np.max([len(h_1), len(h_2)])
    lam = -np.log(0.1) / l

    n = np.arange(0, l, 1)
    exp_d = np.e**(-lam * n)

    p_1 = h_1 * np.conjugate(h_1)
    p_2 = h_2 * np.conjugate(h_2)

    diff = abs(p_1 - p_2)*exp_d
    mean = np.mean(diff)

    return mean


def rt_loss(target_rir, matched_rir, sr):
    from scripts.audio.reverb_features import rev_tr
    if len(target_rir.shape) == 1:
        target_rir = np.expand_dims(target_rir, axis=0)
    if len(matched_rir.shape) == 1:
        matched_rir = np.expand_dims(matched_rir, axis=0)

    target_t60 = np.zeros(target_rir.shape[0])
    matched_t60 = np.zeros(matched_rir.shape[0])
    for ch in range(target_rir.shape[0]):
        target_t60[ch], _ = rev_tr(target_rir[[ch], :], sr, interval=np.array([-5, -65]))
        try:
            matched_t60[ch], _ = rev_tr(matched_rir[[ch], :], sr, interval=np.array([-5, -65]))
        except:
            matched_t60[ch] = float('nan')

    diff = abs(target_t60*1000 - matched_t60*1000)
    # return np.nanmean(diff) * np.nanstd(diff)
    return np.nanmean(diff)

def lsd_loss(target_rir, matched_rir, sr, start_fr_hz:int = 20, end_fr_hz:int = 24000):
    last_idx = np.max(np.nonzero(target_rir))
    # last_matched_rir = np.max(np.nonzero(matched_rir))
    # last_idx = np.max([last_target_rir, last_matched_rir])

    target_rir_f = rfft(target_rir, last_idx)
    matched_rir_f = rfft(matched_rir, last_idx)
    # target_rir_f = librosa.power_to_db(target_rir_f, ref=np.max)
    # matched_rir_f = librosa.power_to_db(matched_rir_f, ref=np.max)
    # target_rir_f = np.fft.rfft(target_rir, axis=0)/target_rir.shape[0]

    start_fr_bin = int(target_rir_f.shape[0] * start_fr_hz / (sr / 2))
    end_fr_bin = int(target_rir_f.shape[0] * end_fr_hz / (sr / 2))

    # loss = np.mean(np.abs(np.abs(target_rir_f[start_fr_bin:]) - np.abs(matched_rir_f[start_fr_bin:])))

    loss = log_spectral_distance(np.abs(target_rir_f[start_fr_bin:end_fr_bin]), np.abs(matched_rir_f[start_fr_bin:end_fr_bin]))

    return loss

def log_spectral_distance(p1: np.ndarray, p2: np.ndarray):

    return np.sqrt(np.mean((20 * np.log10(p1 / p2))**2))
