import functools

from skopt import gp_minimize

from scripts.parameters_learning import *
from scripts.utils.json_functions import *
from scripts.audio_functions.DSPfunc import *
plt.switch_backend('agg')

rir_path = 'audio_functions/input/chosen_rirs/'
rir_file = os.listdir(rir_path)
rir_folder = ['REVelation']

rir, sr = sf.read(rir_path + rir_file[0])
rir = rir.T

#mix = 'full_wet/'
#mix = '0_5/'
mix = 'new_0_5/'

revelation_path = 'audio_functions/input/REVelation/'
revelation, sr = sf.read(revelation_path + mix + 'revelation_sine_sweep.wav')
revelation = revelation.T
revelation = filter_order(revelation, 3, sr)
revelation = revelation / np.max(abs(revelation))

impulse = create_log_sweep(3, 20, 20000, sr, 3)
impulse = np.stack([impulse, impulse])

test_sound = impulse
rir_eq_coeffs = np.load('audio_functions/armodels/revelation_coeffs_ms.npy', allow_pickle=True)[()]

input_audio_path = 'audio_functions/input/sounds/'
input_audio_file = os.listdir(input_audio_path)

rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
rev_param_names_ex.pop('fdn_size_internal')
rev_param_ranges = rev_param_ranges_ex[:-1]

reference_audio = []

reference_audio.append(revelation)

audio_file = []
for idx, wav in enumerate(input_audio_file):
    audio_file.append(sf.read(input_audio_path + input_audio_file[idx])[0])
    #audio_file[idx] = np.concatenate([audio_file[idx], np.zeros(2 * sr)], axis=0)

    if audio_file[idx].ndim is 1:
        audio_file[idx] = np.stack([audio_file[idx], audio_file[idx]])


# for wav in os.listdir(revelation_path + mix + 'no_norm/'):
#     f, sr = sf.read(revelation_path + mix + 'no_norm/' + wav)
#     f = filter_order(f, 3, sr)
#     f = f / np.max(abs(f))
#     f = f * 0.81
#     sf.write(revelation_path + mix + 'norm/' + wav, f, sr)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pre_norm = False
pre_norm = True

for ref_idx, ref in enumerate(reference_audio):

    rir_eq = list(rir_eq_coeffs.values())[ref_idx]

    test_mid = filters([1], rir_eq[0], test_sound[0])
    # test_side = filters([1], rir_eq[1], test_sound[1])

    # test_eq = ms_matrix(np.array([test_mid, test_side]))
    test_eq = np.stack([test_mid, test_mid])

    distance_func_native = functools.partial(reverb_distance_native, params_dict=rev_param_names_nat,
                                      input_audio=test_eq, ref_audio=ref,
                                      sample_rate=sr, pre_norm=pre_norm)

    distance_func_external = functools.partial(reverb_distance_external, vst3=rev_external, params_dict=rev_param_names_ex,
                                      input_audio=test_eq, ref_audio=ref,
                                      sample_rate=sr, pre_norm=pre_norm)

    # Freeverb

    current_effect = 'fv' if not pre_norm else 'fv_norm'
    effect_folder = 'fv'

    res_rev_native = gp_minimize(distance_func_native, rev_param_ranges_nat, acq_func="gp_hedge",
                                 n_calls=180, n_random_starts=10, random_state=1234)

    print(res_rev_native.x)
    # plot_convergence(res_rev_native)
    optimal_params_nat = rev_param_names_nat

    for i, p in enumerate(optimal_params_nat):
        optimal_params_nat[p] = res_rev_native.x[i]

    current_rir_path = 'audio_functions/results/' + rir_folder[ref_idx] + '/'
    model_store(current_rir_path + '_models/' + effect_folder + '/ms_' + current_effect + '.json', optimal_params_nat)

    opt_rev_native = native_reverb_set_params(optimal_params_nat)

    for audio_idx, input_audio in enumerate(audio_file):
        input_mid = filters([1], rir_eq[0], input_audio[0])
        # input_side = filters([1], rir_eq[1], input_audio[1])
        #
        # input_eq = ms_matrix(np.array([input_mid, input_side]))
        input_eq = np.stack([input_mid, input_mid])

        reverb_audio_native = plugin_process(opt_rev_native, input_eq, sr)
        reverb_audio_native = filter_order(reverb_audio_native, 3, sr)

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        rev_ms = ms_matrix(reverb_audio_native)
        rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
        reverb_audio_native = ms_matrix(rev_ms)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        reverb_norm_native = reverb_audio_native / np.max(abs(reverb_audio_native))
        reverb_norm_native = reverb_norm_native * 0.81

        current_sound = os.listdir('audio_functions/results/' + rir_folder[ref_idx])[:-1][audio_idx]

        sf.write(current_rir_path + current_sound + '/' + current_sound + '_ms_' + current_effect + '.wav',
                 reverb_norm_native.T, sr)

    # FdnReverb

    current_effect = 'fdn' if not pre_norm else 'fdn_norm'
    effect_folder = 'fdn'

    res_rev_external = gp_minimize(distance_func_external, rev_param_ranges_ex, acq_func="gp_hedge",
                                 n_calls=180, n_random_starts=10, random_state=1234)

    print(res_rev_external.x)
    # plot_convergence(res_rev_external)
    optimal_params_ex = rev_param_names_ex

    for i, p in enumerate(optimal_params_ex):
        optimal_params_ex[p] = res_rev_external.x[i]

    current_rir_path = 'audio_functions/results/' + rir_folder[ref_idx] + '/'
    model_store(current_rir_path + '_models/' + effect_folder + '/ms_' + current_effect + '.json', optimal_params_ex)

    external_vst3_set_params(optimal_params_ex, rev_external)

    for audio_idx, input_audio in enumerate(audio_file):
        input_mid = filters([1], rir_eq[0], input_audio[0])
        # input_side = filters([1], rir_eq[1], input_audio[1])
        #
        # input_eq = ms_matrix(np.array([input_mid, input_side]))
        input_eq = np.stack([input_mid, input_mid])

        reverb_audio_external = plugin_process(rev_external, input_eq, sr)
        reverb_audio_external = filter_order(reverb_audio_external, 3, sr)

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        rev_ms = ms_matrix(reverb_audio_external)
        rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
        reverb_audio_external = ms_matrix(rev_ms)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        reverb_norm_external = reverb_audio_external / np.max(abs(reverb_audio_external))
        reverb_norm_external = reverb_norm_external * 0.81

        current_sound = os.listdir('audio_functions/results/' + rir_folder[ref_idx])[:-1][audio_idx]

        sf.write(current_rir_path + current_sound + '/' + current_sound + '_ms_' + current_effect + '.wav',
                 reverb_norm_external.T, sr)

