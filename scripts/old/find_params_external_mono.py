from pedalboard import pedalboard
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.audio_functions.audio_manipulation import create_impulse, mel_spectrogram_l1_distance, \
    energy_decay_relief
from scripts.audio_functions.pedalboard_functions import external_vst3_set_params, native_reverb_set_params,\
    plugin_process, retrieve_external_vst3_params
from scripts.utils.plot_functions import plot_melspec_pair, plot_edr_pair

sr = 44100
impulse = create_impulse(44100)

reference_params = {'room_size': 0.8, 'damping': 0.2, 'wet_level': 0.7, 'dry_level': 0.4, 'width': 1.0}
reference_reverb = native_reverb_set_params(reference_params)
reference_audio = plugin_process(reference_reverb, impulse, sr)


vst = pedalboard.load_plugin("vst3/FdnReverb.vst3")
param_names, param_ranges = retrieve_external_vst3_params(vst)

def reverb_distance(params, reverb=vst, params_dict=param_names, audio=impulse, ref_audio=reference_audio, sample_rate=sr):

    for i, p in enumerate(params_dict.keys()):
        params_dict[p] = params[i]

    reverb_to_match = external_vst3_set_params(params_dict, reverb)
    audio_to_match = plugin_process(reverb_to_match, audio, sample_rate)

    #return edr_l1_distance(audio_to_match, ref_audio, sample_rate)
    return mel_spectrogram_l1_distance(ref_audio, audio_to_match, sample_rate)


res = gp_minimize(reverb_distance, param_ranges, acq_func="EI", n_calls=150, n_random_starts=5, random_state=1234)

print(res.x)
plot_convergence(res)
optimal_params = param_names

for i, p in enumerate(optimal_params):
    optimal_params[p] = res.x[i]

#model_store('models/ParamsEDR1.json', optimal_params)
#model_store('models/ParamsEDR3040.json', optimal_params)
#optimal_params = model_load('models/ParamsMelSpec1.json')
#optimal_params = model_load('models/EDR2.json')

generated_reverb = external_vst3_set_params(optimal_params, vst)
generated_audio = plugin_process(generated_reverb, impulse, sr)

fig1, ax1 = plt.subplots()
ax1.plot(generated_audio[:44100])
fig2, ax2 = plt.subplots()
ax2.plot(reference_audio[:44100])

plot_melspec_pair(reference_audio, generated_audio, 2048, 0.25, sr)

#sf.write("audio_functions/Reference2.wav", reference_audio, sr)
#sf.write("audio_functions/GeneratedMelSpec2.wav", generated_audio, sr)
#sf.write("audio_functions/GeneratedEDR1.wav", generated_audio, sr)

e_ref = energy_decay_relief(reference_audio, 30, sr)
e_gen = energy_decay_relief(generated_audio, 30, sr)

plot_edr_pair(e_ref, e_gen)