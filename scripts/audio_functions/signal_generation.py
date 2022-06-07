import numpy as np
from scipy.signal import chirp


def create_impulse(length: int, stereo=False):
    impulse = np.concatenate([[1], np.zeros(length - 1)], axis=0)

    if stereo:
        impulse = np.stack([impulse, impulse])

    return impulse


def create_log_sweep(length: int, f0: float, f1: float, sr: float, silence: int, stereo=False):

    t = np.arange(0, int(length * sr)) / sr
    sweep = chirp(t, f0=f0, f1=f1, t1=length, method='logarithmic')

    sweep = np.concatenate([sweep, np.zeros(silence * sr)], axis=0)

    if stereo:
        sweep = np.stack([sweep, sweep])

    return sweep
