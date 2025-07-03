import librosa.display
from scripts.audio.signal_generation import create_impulse
from scripts.audio.audio_manipulation import *
from scripts.audio.pedalboard_functions import *
from scipy import stats


def rev_tr(ir, sr, interval: np.ndarray = np.array([-5, -25])):
	# energy = 10 * np.log10(np.flip(np.cumsum(np.flip(ir**2), axis=1)) + np.finfo(float).eps)
	energy = 20 * np.log10(np.flip(np.cumsum(np.flip(ir**2), axis=1)) + np.finfo(float).eps)
	energy = np.mean(energy, axis=0)
	energy = energy - np.max(energy)

	t = np.arange(0, len(energy)) / sr

	edt = t[np.sum(energy > (-10))]

	a = np.sum(energy > np.max(interval))
	b = np.sum(energy > np.min(interval))

	if a == b:
		tr = np.nan

	else:
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


def compute_rev_features(audio, sr):
	rev_features = {}
	rev_features['T20'] = []
	rev_features['T30'] = []
	rev_features['T60'] = []
	# rev_features['T20_pa'] = []
	# rev_features['T30_pa'] = []
	# rev_features['T60_pa'] = []
	rev_features['EDT'] = []
	# rev_features['EDT_pa'] = []
	rev_features['strenGth'] = []
	rev_features['C80'] = []
	# rev_features['LF80'] = []
	rev_features['Ts'] = []
	# rev_features['SC'] = act_sc

	for ch in range(audio.shape[0]):
		# Codice Riccardo
		act_t20, act_edt, act_g, act_c, act_lf, act_ts, act_sc = get_rev_features(audio[ch:ch + 1], sr)
		t30, _ = rev_tr(audio[ch:ch + 1], sr, interval=np.array([-5, -35]))
		t60, _ = rev_tr(audio[ch:ch + 1], sr, interval=np.array([-5, -65]))
		# t20_pa = rt_impulse_pyacoustics(audio[ch:ch + 1], sr, rt='t20')
		# t30_pa = rt_impulse_pyacoustics(audio[ch:ch + 1], sr, rt='t30')
		# t60_pa = rt_impulse_pyacoustics(audio[ch:ch + 1], sr, rt='t60')
		# edt_pa = rt_impulse_pyacoustics(audio[ch:ch + 1], sr, rt='edt')

		# # pycoustics: https://github.com/BrechtDeMan/pycoustics/blob/master/pycoustics/measures.py
		# edc = EDC(audio[ch,:], sr)
		# act_t20 = RT(edc, sr, -25)
		# act_t20 = act_t20[1] * 1000
		# t30 = RT(edc, sr, -35)
		# t30 = t30[1] * 1000
		# try:
		#     t60 = RT(edc, sr, -65)
		#     t60 = t60[1] * 1000
		# except:
		#     t60 = np.nan
		# act_edt = EDT(edc, sr)
		# act_edt = act_edt[1] * 1000
		# act_c = C80(audio[ch,:], sr, delay=0)
		# act_ts = TS(audio[ch,:], sr, delay=0)

		rev_features['T20'].append(act_t20)
		rev_features['EDT'].append(act_edt)
		rev_features['T30'].append(t30 * 1000)
		rev_features['T60'].append(t60 * 1000)
		# rev_features['T20_pa'].append(t20_pa)
		# rev_features['EDT_pa'].append(edt_pa)
		# rev_features['T30_pa'].append(t30_pa)
		# rev_features['T60_pa'].append(t60_pa)
		rev_features['strenGth'].append(act_g)
		rev_features['C80'].append(act_c)
		# rev_features['LF80'] = act_lf)
		rev_features['Ts'].append(act_ts)
		# rev_features['SC'] = act_sc)

	# measure_rt60(audio, fs=sr, decay_db=20, plot=True, rt60_tgt=rev_features['T20'])
	# rev_features['T20'] = t60_impulse(path, bands_rt, rt='t20')
	# rev_features['EDT'] = t60_impulse(path, bands_rt, rt='edt')
	# rev_features['C80'] = clarity(80, audio, bands_rt)
	return rev_features


def rt_impulse_pyacoustics(rir, fs, rt='t30'):
	rt = rt.lower()
	if rt == 't60':
		init = -5.0
		end = -65.0
		factor = 1.0
	if rt == 't30':
		init = -5.0
		end = -35.0
		factor = 2.0
	elif rt == 't20':
		init = -5.0
		end = -25.0
		factor = 3.0
	elif rt == 't10':
		init = -5.0
		end = -15.0
		factor = 6.0
	elif rt == 'edt':
		init = 0.0
		end = -10.0
		factor = 6.0

	# Filtering signal
	abs_signal = np.abs(rir) / np.max(np.abs(rir))

	# Schroeder integration
	sch = np.cumsum(abs_signal[::-1]**2)[::-1]
	sch_db = 10.0 * np.log10(sch / np.max(sch))

	# Linear regression
	sch_init = sch_db[np.abs(sch_db - init).argmin()]
	sch_end = sch_db[np.abs(sch_db - end).argmin()]
	init_sample = np.where(sch_db == sch_init)[0][0]
	end_sample = np.where(sch_db == sch_end)[0][0]
	x = np.arange(init_sample, end_sample + 1) / fs
	y = sch_db[init_sample:end_sample + 1]
	slope, intercept = stats.linregress(x, y)[0:2]

	# Reverberation time (T30, T20, T10 or EDT)
	db_regress_init = (init - intercept) / slope
	db_regress_end = (end - intercept) / slope
	t60 = factor * (db_regress_end - db_regress_init)

	return t60


def get_rev_features(ir, sr, interval: np.ndarray = np.array([-5, -25])):

	tr, edt = rev_tr(ir, sr, interval)
	g = rev_g(ir, sr)
	c = rev_c(ir, sr)
	lf = np.nan#rev_lf(ir, sr)
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
