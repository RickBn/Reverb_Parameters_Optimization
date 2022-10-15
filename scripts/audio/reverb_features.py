import librosa.display
from scripts.audio.signal_generation import create_impulse
from scripts.audio.audio_manipulation import *
from scripts.audio.pedalboard_functions import *


def rev_tr(ir, sr, interval: np.ndarray = np.array([-5, -25])):
	energy = 10 * np.log10(np.flip(np.cumsum(np.flip(ir**2), axis=1)) + np.finfo(float).eps)
	energy = np.mean(energy, axis=0)
	energy = energy - np.max(energy)

	t = np.arange(0, len(energy)) / sr

	edt = t[np.sum(energy > (-10))]

	a = np.sum(energy > np.max(interval))
	b = np.sum(energy > np.min(interval))

	x = t[a:b]
	y = energy[a:b]

	p = np.polyfit(x, y, 1)

	tr = -60/p[0]
	#sample_idx = round(tr * sr)

	return [tr, edt]


def rev_g(ir, sr, dt = 5):
	dt = round(np.ceil(sr * dt * 0.001))

	di = ir[:, :dt]
	lp = 10 * np.log10(np.sum(ir**2, axis=1) + np.finfo(float).eps)
	lw = 10 * np.log10(np.sum(di**2, axis=1) + np.finfo(float).eps)

	g = lp - lw
	return np.mean(g)


def rev_c(ir, sr, c_time=80):

	t = np.arange(0, len(ir.T)) / sr
	inf = np.sum(ir[:, t < c_time * 0.001]**2, axis=1) + np.finfo(float).eps
	sup = np.sum(ir[:, t > c_time * 0.001]**2, axis=1) + np.finfo(float).eps
	c = 10 * np.log10((inf / sup))

	return np.mean(c)


def rev_lf(ir, sr, lf_time=80):

	etime = round(np.ceil(sr * lf_time * 0.001))
	rir = ir[:, :etime]
	m, s = ms_matrix(rir)
	lf = 10 * np.log10((np.sum(s**2) + np.finfo(float).eps) / (np.sum(m**2) + np.finfo(float).eps))

	return lf


def rev_ts(ir, sr):

	t = np.arange(0, len(ir.T)) / sr
	ir_t = ir**2 * np.arange(0, len(ir.T))
	num = np.sum(ir_t, axis=1)
	den = np.sum(ir**2, axis=1)

	ts = np.mean(num/den)
	ts = t[round(ts)] * 1000

	return ts


def rev_sc(ir, sr):

	sc = 0.0
	for ch in ir:
		sc_i = librosa.feature.spectral_centroid(ch, sr)
		sc += sc_i
	sc = sc / ir.ndim

	return np.mean(sc)


def get_rev_features(ir, sr):

	tr, edt = rev_tr(ir, sr)
	g = rev_g(ir, sr)
	c = rev_c(ir, sr)
	lf = rev_lf(ir, sr)
	ts = rev_ts(ir, sr)
	sc = rev_sc(ir, sr)

	tr = tr * 1000
	edt = edt * 1000

	return [tr, edt, g, c, lf, ts, sc]


if __name__ == "__main__":
	sr = 44100
	ir = create_impulse(sr * 6)
	ir = np.stack([ir, ir])

	params = {'room_size': 0.2, 'damping': 0.1, 'wet_level': 0.5, 'dry_level': 0.2, 'width': 0.5}
	rev = native_reverb_set_params(params)
	ir = plugin_process(rev, ir, sr)

	tr, edt, g, c, lf, ts, sc = get_rev_features(ir, sr)
	print('tr:', tr, 'edt:', edt, 'g:', g, 'c:', c, 'lf:', lf, 'ts:', ts, 'sc:', sc)
