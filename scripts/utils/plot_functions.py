import os
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_rir_pair(path1, path2, save_path=None, ch=0):

    r1, sr = sf.read(path1)
    r1 = r1.T

    r2, sr = sf.read(path2)
    r2 = r2.T

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)

    plt.plot(r1[0])

    ax2 = fig.add_subplot(1, 2, 2)

    plt.plot(r2[0])

    if save_path is not None:
        plt.savefig(save_path, format='pdf')


def plot_melspec_pair(audio1, audio2, n_fft, hop_length, sample_rate=44100, save_path=None):
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    w1 = librosa.feature.melspectrogram(y=audio1, sr=sample_rate, n_fft=n_fft, hop_length=int(n_fft * hop_length))
    R_dB = librosa.power_to_db(w1, ref=np.max)
    img = librosa.display.specshow(R_dB, x_axis='time',
                                   y_axis='mel', sr=sample_rate, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Reference')

    ax = fig.add_subplot(1, 2, 2)
    w2 = librosa.feature.melspectrogram(y=audio2, sr=sample_rate, n_fft=n_fft, hop_length=int(n_fft * hop_length))
    G_dB = librosa.power_to_db(w2, ref=np.max)
    img = librosa.display.specshow(G_dB, x_axis='time',
                                   y_axis='mel', sr=sample_rate, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Generated')

    if save_path is not None:
        plt.savefig(save_path, format='pdf')

    return


def plot_edr_pair(edr1, edr2):

    f = librosa.fft_frequencies(sr=44100, n_fft=2048) * 0.001

    n = np.max([edr1.shape[1], edr2.shape[1]])
    t = np.arange(0, 1, 1 / n)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(10, 55)
    X, Y = np.meshgrid(t, f)
    Z = edr1
    ax.plot_surface(X, Y, Z, cmap=cm.CMRmap, linewidth=0, antialiased=False)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(10, 55)
    X, Y = np.meshgrid(t, f)
    Z = edr2
    ax.plot_surface(X, Y, Z, cmap=cm.CMRmap, linewidth=0, antialiased=False)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_zlabel('Magnitude (dB)')

    return

