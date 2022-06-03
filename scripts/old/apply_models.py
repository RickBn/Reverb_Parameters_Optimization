import functools

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence

from scripts.parameters_learning import *
from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
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
    rir_eq_coeffs = np.load('audio/armodels/rir_eq_coeffs_kl_s.npy', allow_pickle=True)[()]

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
        reference_audio[idx] = filter_order(reference_audio[idx], 3, sr)
        reference_norm.append(reference_audio[idx] / np.max(abs(reference_audio[idx])))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for ref_idx, ref in enumerate(reference_audio):

        rir_eq = list(rir_eq_coeffs.values())[ref_idx]

        test_mid = filters([1], rir_eq[0], test_sound[0])
        test_eq = np.stack([test_mid, test_mid])

        sf.write('audio/final_rirs/' + rir_folder[ref_idx] + '/reference.wav', ref.T, sr)

        # Freeverb

        effect_folder = 'fv'

        current_params_path = 'audio/params/' + rir_folder[ref_idx] + '/' + effect_folder + '/'
        optimal_params_nat = model_load(current_params_path + 'ms_' + effect_folder + '_norm.json')

        opt_rev_native = native_reverb_set_params(optimal_params_nat)

        reverb_audio_native = plugin_process(opt_rev_native, test_eq, sr)
        reverb_audio_native = filter_order(reverb_audio_native, 3, sr)

        rev_ms = ms_matrix(reverb_audio_native)
        rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
        reverb_audio_native = ms_matrix(rev_ms)

        reverb_norm_native = reverb_audio_native / np.max(abs(reverb_audio_native))

        sf.write('audio/final_rirs/' + rir_folder[ref_idx] + '/' + effect_folder + '.wav', reverb_norm_native.T, sr)

        plot_melspec_pair(ref[0], reverb_norm_native[0], 2048, 0.25, 44100,
                          'audio/final_rirs/' + rir_folder[ref_idx] + '/melspec_' + effect_folder + '.pdf')

        # FdnReverb

        effect_folder = 'fdn'

        current_params_path = 'audio/params/' + rir_folder[ref_idx] + '/' + effect_folder + '/'
        optimal_params_ex = model_load(current_params_path + 'ms_' + effect_folder + '_norm.json')

        external_vst3_set_params(optimal_params_ex, rev_external)

        reverb_audio_external = plugin_process(rev_external, test_eq, sr)
        reverb_audio_external = filter_order(reverb_audio_external, 3, sr)

        rev_ms = ms_matrix(reverb_audio_external)
        rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
        reverb_audio_external = ms_matrix(rev_ms)

        reverb_norm_external = reverb_audio_external / np.max(abs(reverb_audio_external))

        sf.write('audio/final_rirs/' + rir_folder[ref_idx] + '/' + effect_folder + '.wav', reverb_norm_external.T, sr)

        plot_melspec_pair(ref[0], reverb_norm_external[0], 2048, 0.25, 44100,
                          'audio/final_rirs/' + rir_folder[ref_idx] + '/melspec_' + effect_folder + '.pdf')

find_params()