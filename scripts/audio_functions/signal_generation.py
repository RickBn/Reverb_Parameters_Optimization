import numpy as np
from scipy.signal import chirp


def create_impulse(length: int, n_channels: int = 1):
    impulse = np.concatenate([[1], np.zeros(length - 1)], axis=0)

    if n_channels > 1:
        impulse = np.stack([impulse] * n_channels)

    return impulse


def create_log_sweep(length: int, f0: float, f1: float, sr: float, silence: int, n_channels: int = 1):

    t = np.arange(0, int(length * sr)) / sr
    sweep = chirp(t, f0=f0, f1=f1, t1=length, method='logarithmic')

    sweep = np.concatenate([sweep, np.zeros(int(silence * sr))], axis=0)

    if n_channels > 1:
        sweep = np.stack([sweep] * n_channels)

    return sweep
