import functools
import librosa.display

from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.pedalboard_functions import *
from scripts.utils.plot_functions import *
#plt.switch_backend('agg')

rir_path = 'audio_functions/input/chosen_rirs/Auditorium - Atherton Hall (M).wav'
rir, sr = sf.read(rir_path)
rir = rir.T

impulse = create_log_sweep(3, 20, 20000, sr, 5)
impulse = np.stack([impulse, impulse])

test_sound = impulse

#audio_file = sf.read('audio_functions/input/sounds/snare_hit.wav')[0].T
audio_file = sf.read('audio_functions/input/sounds/gtr_loop_b1.wav')[0]
audio_file = np.concatenate([audio_file, np.zeros(2 * sr)], axis=0)

if audio_file.ndim is 1:
    audio_file = np.stack([audio_file, audio_file])

convolution = pedalboard.Convolution(rir_path, 1.0)
reference_audio = convolution(test_sound, sr)

reference_norm = reference_audio / np.max(abs(reference_audio))

#sf.write('audio_functions/results/Auditorium - Atherton Hall (M)/guitar/gtr_ref.wav', reference_norm.T, sr)

rev_param_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
rev_param_names = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

hp = pedalboard.HighpassFilter(cutoff_frequency_hz=50)


def reverb_distance(params, params_dict, audio, ref_audio, sample_rate, loss_func):

    for i, p in enumerate(params_dict):
        params_dict[p] = params[i]

    reverb_to_match = native_reverb_set_params(params_dict)
    audio_to_match = plugin_process(reverb_to_match, audio, sample_rate)

    audio_to_match = hp(audio_to_match, sr)
    audio_to_match = hp(audio_to_match, sr)
    audio_to_match = hp(audio_to_match, sr)

    # # #####
    # ref_audio = ref_audio / np.max(abs(ref_audio))
    # audio_to_match = audio_to_match / np.max(abs(audio_to_match))
    # ref_audio[np.isnan(ref_audio)] = 0
    # audio_to_match[np.isnan(audio_to_match)] = 0
    # # #####

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
    #return distance.pdist([ref_audio, audio_to_match], 'euclidean')[0]


# l_f = 'edr'
l_f = 'mel_spec'
# l_f = 'mfcc'

distance_func = functools.partial(reverb_distance, params_dict=rev_param_names,
                                  audio=test_sound, ref_audio=reference_audio,
                                  sample_rate=sr, loss_func=l_f)

res_rev = gp_minimize(distance_func, rev_param_ranges, acq_func="gp_hedge",
                      n_calls=200, n_random_starts=10, random_state=1234)

print(res_rev.x)
plot_convergence(res_rev)
optimal_params = rev_param_names

for i, p in enumerate(optimal_params):
    optimal_params[p] = res_rev.x[i]

#model_store('audio_functions/results/Auditorium - Atherton Hall (M)/_models/sweep_ms_fv.json', optimal_params)
#model_store('audio_functions/results/Auditorium - Atherton Hall (M)/_models/snare_ms_fv.json', optimal_params)
#model_store('audio_functions/results/Auditorium - Atherton Hall (M)/_models/gtr_ms_fv.json', optimal_params)

#optimal_params = model_load('audio_functions/results/Auditorium - Atherton Hall (M)/_models/sweep_ms_fv.json')

opt_rev = native_reverb_set_params(optimal_params)
reverb_audio = plugin_process(opt_rev, audio_file, sr)

reverb_audio = hp(reverb_audio, sr)
reverb_audio = hp(reverb_audio, sr)
reverb_audio = hp(reverb_audio, sr)

reverb_norm = reverb_audio / np.max(abs(reverb_audio))

#model_store('audio_functions/sweep_snare_ms_fv2.json', optimal_params)
#sf.write('audio_functions/gtr_ms_fvn2.wav', reverb_norm.T, sr)

fig1, ax1 = plt.subplots()
librosa.display.waveshow(reverb_norm[0], sr, ax=ax1)
fig2, ax2 = plt.subplots()
librosa.display.waveshow(reference_norm[0], sr, ax=ax2)

plot_melspec_pair(reference_norm[0], reverb_norm[0], 2048, 0.25, sr)

e_ref = energy_decay_relief(reference_norm[0], 30, sr)
e_gen = energy_decay_relief(reverb_norm[0], 30, sr)

plot_edr_pair(e_ref, e_gen)

#sf.write('audio_functions/results/Auditorium - Atherton Hall (M)/snare/sweep_snare_ms_fv.wav', reverb_norm.T, sr)
#sf.write('audio_functions/results/Auditorium - Atherton Hall (M)/snare/snare_ms_fv.wav', reverb_norm.T, sr)

#sf.write('audio_functions/results/Auditorium - Atherton Hall (M)/guitar/sweep_gtr_ms_fv.wav', reverb_norm.T, sr)
#sf.write('audio_functions/results/Auditorium - Atherton Hall (M)/guitar/gtr_ms_fv.wav', reverb_norm.T, sr)