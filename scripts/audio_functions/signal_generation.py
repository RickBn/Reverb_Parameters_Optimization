import numpy as np

from scipy.signal import chirp


def create_impulse(length: int):
    impulse = np.concatenate([[1], np.zeros(length - 1)], axis=0)

    return impulse


def create_log_sweep(length: int, f0: float, f1: float, sr: float, silence: int):

    t = np.arange(0, int(length * sr)) / sr
    w = chirp(t, f0=f0, f1=f1, t1=length, method='logarithmic')

    w = np.concatenate([w, np.zeros(silence * sr)], axis=0)

    return w