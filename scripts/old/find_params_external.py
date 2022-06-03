import functools
import librosa.display

from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.pedalboard_functions import *
from scripts.utils.plot_functions import *

# impulse, sr = sf.read('audio_functions/input/voice.wav')
# impulse, sr = sf.read('audio_functions/input/impulse_response.wav')
# impulse = np.append(impulse, np.zeros(176400))

# l = model_load('audio_functions/input/l.json')
# r = model_load('audio_functions/input/r.json')
# reference_audio = np.stack([l, r])
# impulse = create_impulse(len(reference_audio[0]))
# impulse = np.stack([impulse, impulse])

rir, sr = sf.read('audio_functions/input/air_lecture_X_0_6.wav')

impulse = create_log_sweep(3, 20, 20000, sr, 5)
impulse = np.stack([impulse, impulse])

convolution = pedalboard.Convolution('audio_functions/input/air_lecture_X_0_6.wav', 1.0)
reference_audio = convolution(impulse, sr)

# reference_params = {'room_size': 0.8, 'damping': 0.2, 'wet_level': 1.0, 'dry_level': 0.0, 'width': 1.0}
# reference_reverb = native_reverb_set_params(reference_params)
#reference_audio = plugin_process(reference_reverb, impulse, sr)

reference_norm = reference_audio / np.max(abs(reference_audio))

# sf.write('audio_functions/stereo/ir/Reference.wav', reference_audio.T, sr)
# sf.write('audio_functions/stereo/ir/Norm_Reference.wav', reference_norm.T, sr)

rev = pedalboard.load_plugin("vst3/FdnReverb.vst3")
rev_param_names, rev_param_ranges = retrieve_external_vst3_params(rev)

rev_param_names.pop('fdn_size_internal')
rev_param_ranges = rev_param_ranges[:-1]

# rev.lows_cutoff_frequency_hz._AudioProcessorParameter__get_cpp_parameter().args[0].\
#     __init__(rev, 'Lows Cutoff Frequency', search_steps=20000)

hp = pedalboard.HighpassFilter(cutoff_frequency_hz=50)


def reverb_distance(params, vst3, params_dict, audio, ref_audio, sample_rate, loss_func):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    external_vst3_set_params(params_dict, vst3)
    audio_to_match = plugin_process(vst3, audio, sample_rate)

    audio_to_match = hp(audio_to_match, sr)
    audio_to_match = hp(audio_to_match, sr)
    audio_to_match = hp(audio_to_match, sr)

    loss = 0.0

    if loss_func == 'edr':
        loss = np.mean([edr_l1_distance(audio_to_match[0], ref_audio[0], sample_rate),
                        edr_l1_distance(audio_to_match[1], ref_audio[1], sample_rate)])

    elif loss_func == 'mel_spec':
        loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
                        mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])

    elif loss_func == 'mfcc':
        loss = np.mean([mfcc_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
                        mfcc_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])

    return loss


# l_f = 'edr'
l_f = 'mel_spec'
# l_f = 'mfcc'

distance_func = functools.partial(reverb_distance, vst3=rev, params_dict=rev_param_names,
                                  audio=impulse, ref_audio=reference_audio,
                                  sample_rate=sr, loss_func=l_f)

res_rev = gp_minimize(distance_func, rev_param_ranges, acq_func="gp_hedge",
                      n_calls=200, n_random_starts=10, random_state=1234)

print(res_rev.x)
plot_convergence(res_rev)
optimal_params = rev_param_names

for i, p in enumerate(optimal_params):
    optimal_params[p] = res_rev.x[i]

external_vst3_set_params(optimal_params, rev)
reverb_audio = plugin_process(rev, impulse, sr)

reverb_audio = hp(reverb_audio, sr)
reverb_audio = hp(reverb_audio, sr)
reverb_audio = hp(reverb_audio, sr)

reverb_norm = reverb_audio / np.max(abs(reverb_audio))

fig1, ax1 = plt.subplots()
librosa.display.waveshow(reverb_norm[0], sr, ax=ax1)
fig2, ax2 = plt.subplots()
librosa.display.waveshow(reference_norm[0], sr, ax=ax2)

plot_melspec_pair(reference_norm[0], reverb_norm[0], 2048, 0.25, sr)

e_ref = energy_decay_relief(reference_norm[0], 30, sr)
e_gen = energy_decay_relief(reverb_norm[0], 30, sr)

plot_edr_pair(e_ref, e_gen)

# ref_ms = ms_matrix(reference_audio)
# rev_ms = ms_matrix(reverb_audio)
#
# rev_ms[0] /= np.max(ref_ms[0])
# rev_ms[1] /= np.max(ref_ms[1])
#
# generated_audio = ms_matrix(np.array([rev_ms[0], rev_ms[1]]))
# generated_norm = generated_audio / np.max(abs(generated_audio))

# fig1, ax1 = plt.subplots()
# librosa.display.waveshow(generated_norm[0], sr, ax=ax1)
# fig2, ax2 = plt.subplots()
# librosa.display.waveshow(reference_norm[0], sr, ax=ax2)
#
# plot_melspec_pair(reference_norm[0], generated_norm[0], 2048, 0.25, sr)