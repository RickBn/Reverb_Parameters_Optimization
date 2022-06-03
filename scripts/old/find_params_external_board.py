from pedalboard import Pedalboard, pedalboard
import numpy as np
import matplotlib.pyplot as plt
import librosa

from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.audio_functions.audio_manipulation import create_impulse, energy_decay_relief, edr_l1_distance
from scripts.audio_functions.pedalboard_functions import external_vst3_set_params, native_reverb_set_params,\
    plugin_process, retrieve_external_vst3_params, board_process
from scripts.utils.plot_functions import plot_melspec_pair, plot_edr_pair

sr = 44100
impulse = create_impulse(44100)
impulse = np.array([impulse, impulse])

reference_params = {'room_size': 0.8, 'damping': 0.2, 'wet_level': 0.7, 'dry_level': 0.4, 'width': 1.0}
reference_reverb = native_reverb_set_params(reference_params)
reference_audio = plugin_process(reference_reverb, impulse, sr)


# vst = pedalboard.load_plugin("vst3/FdnReverb.vst3")
# param_names, param_ranges = retrieve_external_vst3_params(vst)

rev = pedalboard.load_plugin("vst3/FdnReverb.vst3")
rev_param_names, rev_param_ranges = retrieve_external_vst3_params(rev)

#rev_param_names.pop('fdn_size_internal')
#rev_param_ranges = rev_param_ranges[:-1]

eq = pedalboard.load_plugin("vst3/MS_BandEQ.vst3")
eq_param_names, eq_param_ranges = retrieve_external_vst3_params(eq)

param_names = [eq_param_names, rev_param_names]
param_ranges = eq_param_ranges + rev_param_ranges

board = Pedalboard([eq, rev], sample_rate=sr)
#board[1].fdn_size_internal = 64

def reverb_distance(params, pedalboard=board,
                    params_dict=param_names, audio=impulse, ref_audio=reference_audio, sample_rate=sr):

    par = 0
    for vst3 in range(0, len(pedalboard)):
        for i, p in enumerate(params_dict[vst3].keys()):
            params_dict[vst3][p] = params[par]
            par += 1
        pedalboard[vst3] = external_vst3_set_params(params_dict[vst3], board[vst3])

    audio_to_match = board_process(pedalboard, audio)

    return np.mean([edr_l1_distance(audio_to_match[0], ref_audio[0], sample_rate), edr_l1_distance(audio_to_match[1], ref_audio[1], sample_rate)])
    #return np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate), mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])


res = gp_minimize(reverb_distance, param_ranges, acq_func="EI", n_calls=150, n_random_starts=5, random_state=1234)

print(res.x)
plot_convergence(res)

offset = 0
for v in range(0, len(board)):
    optimal_params = param_names[v]
    for i, p in enumerate(optimal_params):
        optimal_params[p] = res.x[offset:offset + len(optimal_params)][i]
    external_vst3_set_params(optimal_params, board[v])
    offset += len(optimal_params)

generated_audio = board_process(board, impulse)

#model_store('models/ParamsMelSpec2StereoRaw.json', res.x)
#model_store('models/EDR2StereoRaw.json', res.x)


fig1, ax1 = plt.subplots()
ax1.plot(generated_audio[0][:44100])
fig2, ax2 = plt.subplots()
ax2.plot(reference_audio[0][:44100])

plot_melspec_pair(reference_audio[0], generated_audio[0], 2048, 0.25, sr)

#sf.write("audio_functions/Reference.wav", reference_audio, sr)
#sf.write("audio_functions/ReferenceStereo.wav", reference_audio.T, sr)

#sf.write("audio_functions/GeneratedMelSpec1.wav", generated_audio, sr)
#sf.write("audio_functions/GeneratedMelSpec1Stereo.wav", generated_audio.T, sr)

#sf.write("audio_functions/GeneratedEDR1.wav", generated_audio, sr)
#sf.write("audio_functions/GeneratedEDR2Stereo.wav", generated_audio.T, sr)

e_ref = energy_decay_relief(reference_audio[0], 30, sr)
e_gen = energy_decay_relief(generated_audio[0], 30, sr)

plot_edr_pair(e_ref, e_gen)


def match_ms(gain, ms_ref: np.ndarray, ms_to_match: np.ndarray):

    loss = np.mean([abs(librosa.feature.rms(ms_ref) - librosa.feature.rms(ms_to_match * gain[0]))])

    return loss


# match_mid = functools.partial(match_ms, ms_ref=ref_ms[0], ms_to_match=rev_ms[0])
# match_side = functools.partial(match_ms, ms_ref=ref_ms[1], ms_to_match=rev_ms[1])
#
# res_mid = gp_minimize(match_mid, [(0.0, 16.0)], acq_func="gp_hedge",
#                       n_calls=50, n_random_starts=5, random_state=1234)
# res_side = gp_minimize(match_side, [(0.0, 16.0)], acq_func="gp_hedge",
#                        n_calls=50, n_random_starts=5, random_state=1234)
#
# generated_audio = ms_matrix(np.array([rev_ms[0] * res_mid.x[0], rev_ms[1] * res_side.x[0]]))