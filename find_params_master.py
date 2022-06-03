import functools

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.parameters_learning import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.audio_metrics import *
from scripts.audio_functions.signal_generation import *
from scripts.utils.plot_functions import *
from scripts.direct_sound_eq import *

#plt.switch_backend('agg')

def find_params():
    rir_path = 'audio/input/chosen_rirs/'
    rir_file = os.listdir(rir_path)
    rir_folder = os.listdir('audio/results')

    rir, sr = sf.read(rir_path + rir_file[0])
    rir = rir.T

    impulse = create_log_sweep(3, 20, 20000, sr, 3)
    impulse = np.stack([impulse, impulse])

    test_sound = impulse
    rir_eq_coeffs = np.load('audio/armodels/rir_eq_coeffs_ms.npy', allow_pickle=True)[()]

    input_audio_path = 'audio/input/sounds/'
    input_audio_file = os.listdir(input_audio_path)

    rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

    rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
    rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
    rev_param_names_ex.pop('fdn_size_internal')
    rev_param_ranges_ex = rev_param_ranges_ex[:-1]

    convolution = []
    reference_audio = []
    reference_norm = []

    mix = 1.0

    for idx, rir_file in enumerate(os.listdir(rir_path)):
        convolution.append(pedalboard.Convolution(rir_path + rir_file, mix))
        reference_audio.append(convolution[idx](test_sound, sr))
        reference_audio[idx] = pd_highpass_filter(reference_audio[idx], 3, sr)
        reference_norm.append(normalize_audio(reference_audio[idx]))

    audio_file = []
    for idx, wav in enumerate(input_audio_file):
        audio_file.append(sf.read(input_audio_path + input_audio_file[idx])[0])

        if audio_file[idx].ndim is 1:
            audio_file[idx] = np.stack([audio_file[idx], audio_file[idx]])

    convolved = batch_convolve(audio_file, convolution, rir_folder, sr, 0.70)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # pre_norm = False
    pre_norm = True

    for ref_idx, ref in enumerate(reference_audio):

        rir_eq = list(rir_eq_coeffs.values())[ref_idx]

        test_mid = filters([1], rir_eq[0], test_sound[0])
        test_eq = np.stack([test_mid, test_mid])

        distance_func_native = functools.partial(reverb_distance_native, params_dict=rev_param_names_nat,
                                                 input_audio=test_eq, ref_audio=ref,
                                                 sample_rate=sr, pre_norm=pre_norm)

        distance_func_external = functools.partial(reverb_distance_external, vst3=rev_external,
                                                   params_dict=rev_param_names_ex,
                                                   input_audio=test_eq, ref_audio=ref,
                                                   sample_rate=sr, pre_norm=pre_norm)

        current_rir_path = 'audio/results/' + rir_folder[ref_idx] + '/'

        # Freeverb

        current_effect = 'fv' if not pre_norm else 'fv_norm'
        effect_folder = 'fv'

        res_rev_native = gp_minimize(distance_func_native, rev_param_ranges_nat, acq_func="gp_hedge",
                                     n_calls=180, n_random_starts=10, random_state=1234)

        optimal_params_nat = rev_param_names_nat

        for i, p in enumerate(optimal_params_nat):
            optimal_params_nat[p] = res_rev_native.x[i]

        current_params_path = 'audio/params/' + rir_folder[ref_idx] + '/' + effect_folder + '/'
        model_store(current_params_path + 'ms_' + effect_folder + '_norm.json',
                    optimal_params_nat)

        opt_rev_native = native_reverb_set_params(optimal_params_nat)

        # print(res_rev_native.x)
        # func_val = [round(v) for v in sorted(res_rev_native.func_vals, reverse=True)]
        # min_f = round(res_rev_native.fun)
        # np.savetxt('images/convergence/' + rir_folder[ref_idx] + '_fv.txt',
        #            [func_val.index(min_f), res_rev_native.fun, min_f], fmt='%1.2f')

        # fig = plt.figure()
        # plot_convergence(res_rev_native)
        # plt.savefig('images/convergence/' + rir_folder[ref_idx] + '_fv.png')

        # plot_melspec_pair(ref[0], plugin_process(opt_rev_native, test_eq, sr)[0], 2048, 0.25, 44100,
        #                   'images/melspec/' + rir_folder[ref_idx] + '_fv.png')

        for audio_idx, input_audio in enumerate(audio_file):
            input_mid = filters([1], rir_eq[0], input_audio[0])
            input_eq = np.stack([input_mid, input_mid])

            reverb_audio_native = plugin_process(opt_rev_native, input_eq, sr)
            reverb_audio_native = pd_highpass_filter(reverb_audio_native, 3, sr)

            rev_ms = ms_matrix(reverb_audio_native)
            rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
            reverb_audio_native = ms_matrix(rev_ms)

            reverb_norm_native = reverb_audio_native / np.max(abs(reverb_audio_native))
            reverb_norm_native = reverb_norm_native * 0.70

            current_sound = os.listdir('audio/results/' + rir_folder[ref_idx])[:-1][audio_idx]

            sf.write(current_rir_path + current_sound + '/' + current_sound + '_ms_' + current_effect + '.wav',
                 reverb_norm_native.T, sr)

        # FdnReverb

        current_effect = 'fdn' if not pre_norm else 'fdn_norm'
        effect_folder = 'fdn'

        res_rev_external = gp_minimize(distance_func_external, rev_param_ranges_ex, acq_func="gp_hedge",
                                       n_calls=180, n_random_starts=10, random_state=1234)

        optimal_params_ex = rev_param_names_ex

        for i, p in enumerate(optimal_params_ex):
            optimal_params_ex[p] = res_rev_external.x[i]

        current_params_path = 'audio/params/' + rir_folder[ref_idx] + '/' + effect_folder + '/'
        model_store(current_params_path + 'ms_' + effect_folder + '_norm.json',
                    optimal_params_ex)

        external_vst3_set_params(optimal_params_ex, rev_external)

        # print(res_rev_external.x)
        # func_val = [round(v) for v in sorted(res_rev_external.func_vals, reverse=True)]
        # min_f = round(res_rev_external.fun)
        # np.savetxt('images/convergence/' + rir_folder[ref_idx] + '_fdn.txt',
        #            [func_val.index(min_f), res_rev_external.fun, min_f], fmt='%1.2f')

        # fig = plt.figure()
        # plot_convergence(res_rev_external)
        # plt.savefig('images/convergence/' + rir_folder[ref_idx] + '_fdn.png')

        # plot_melspec_pair(ref[0], plugin_process(rev_external, test_eq, sr)[0], 2048, 0.25, 44100,
        #                   'images/melspec/' + rir_folder[ref_idx] + '_fdn.png')

        for audio_idx, input_audio in enumerate(audio_file):
            input_mid = filters([1], rir_eq[0], input_audio[0])
            input_eq = np.stack([input_mid, input_mid])

            reverb_audio_external = plugin_process(rev_external, input_eq, sr)
            reverb_audio_external = pd_highpass_filter(reverb_audio_external, 3, sr)

            rev_ms = ms_matrix(reverb_audio_external)
            rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
            reverb_audio_external = ms_matrix(rev_ms)

            reverb_norm_external = reverb_audio_external / np.max(abs(reverb_audio_external))
            reverb_norm_external = reverb_norm_external * 0.70

            current_sound = os.listdir('audio/results/' + rir_folder[ref_idx])[:-1][audio_idx]

            sf.write(current_rir_path + current_sound + '/' + current_sound + '_ms_' + current_effect + '.wav',
                 reverb_norm_external.T, sr)


if __name__ == "__main__":
    find_params()



