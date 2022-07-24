import functools
from skopt import gp_minimize
from skopt.plots import plot_convergence

import timeit

from scripts.parameters_learning import *
from scripts.audio_functions.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.plot_functions import plot_melspec_pair
from scripts.old.direct_sound_eq import *
import scipy.signal

#plt.switch_backend('agg')


def find_params_merged(rir_path: str,
                       er_path: str,
                       result_path: str,
                       input_path: str,
                       generate_references: bool = True,
                       pre_norm: bool = True):

    rir_file = os.listdir(rir_path)
    rir_folder = os.listdir(rir_path)

    rir, sr = sf.read(rir_path + rir_file[0])

    impulse = create_impulse(sr * 3, n_channels=25)
    sweep = create_log_sweep(3, 20, 20000, sr, 0, n_channels=25)

    test_sound = sweep

    # rir, sr = sf.read('audio/input/chosen_rirs/HOA/MARCo/0deg_066_Eigen_4th_Bformat_ACN_SN3D.wav')
    # rir = rir.T
    # speech, sr_2 = sf.read('audio/input/sounds/speech.wav')
    # speech = speech.T
    # speech = np.stack([speech] * rir.shape[0])
    # a = scipy.signal.fftconvolve(impulse, rir, mode='full', axes=1)
    # sf.write('audio/a.wav', a.T, sr)


    #rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

    scale_parameter = skopt.space.space.Real(0.0, 1.0, transform='identity')
    rev_param_ranges_nat = [scale_parameter, scale_parameter, scale_parameter, scale_parameter]

    rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'width': 0.0, 'scale': 0.5}

    rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
    rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)

    rev_param_names_ex.pop('fdn_size_internal')
    rev_param_ranges_ex = rev_param_ranges_ex[:-1]

    #//////////////////////////////////
    rev_param_names_ex.pop('dry_wet')
    rev_param_ranges_ex.pop(-2)

    rev_param_names_ex['scale'] = 0.5
    rev_param_ranges_ex.append(scale_parameter)

    rev_external.__setattr__('dry_wet', 1.0)

    rev_plugins = {'Freeverb': [None, rev_param_names_nat, rev_param_ranges_nat],
                   'FdnReverb': [rev_external, rev_param_names_ex, rev_param_ranges_ex]}

    convolution = prepare_batch_convolve(rir_path, mix=1.0)
    audio_file = prepare_batch_input_stereo(input_path)

    input_file_names = os.listdir(input_path)
    result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

    if generate_references:
        batch_convolve(audio_file, convolution, result_file_names, rir_path, sr, 0.70,
                       norm=True, save_path=result_path)

    reference_audio = batch_convolve([test_sound], convolution, result_file_names, rir_path, sr, 1.0, norm=False)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    overall_time = 0
    times = []

    for ref_idx, ref in enumerate(reference_audio):

        rir_er_path = er_path + os.listdir(er_path)[ref_idx]

        rir_er, sr_er = sf.read(rir_er_path)
        rir_er = rir_er.T

        fade_in = int(5 * sr * 0.001)

        current_rir_path = result_path + rir_folder[ref_idx].replace('.wav', '') + '/'
        if not os.path.exists(current_rir_path):
            os.makedirs(current_rir_path)

        merged_rir_path = 'audio/merged_rirs/'
        if not os.path.exists(merged_rir_path):
            os.makedirs(merged_rir_path)

        for rev in rev_plugins:

            current_effect = rev
            effect_folder = rev
            rev_plugin = rev_plugins[rev][0]
            rev_param_names = rev_plugins[rev][1]
            rev_param_ranges = rev_plugins[rev][2]

            distance_func = functools.partial(merged_rir_distance,
                                              params_dict=rev_param_names,
                                              input_audio=test_sound,
                                              ref_audio=ref,
                                              er_path=rir_er_path,
                                              sample_rate=sr,
                                              vst3=rev_plugin,
                                              pre_norm=pre_norm)

            start = timeit.default_timer()

            res_rev = gp_minimize(distance_func, rev_param_ranges, acq_func="gp_hedge",
                                  n_calls=180, n_random_starts=10, random_state=1234)

            stop = timeit.default_timer()

            time_s = stop - start
            overall_time += time_s
            times.append(time_s)

            optimal_params = rev_param_names

            for i, p in enumerate(optimal_params):
                optimal_params[p] = res_rev.x[i]

            # Save params
            current_params_path = 'audio/params/' + rir_folder[ref_idx] + '/' + effect_folder + '/'

            if not os.path.exists(current_params_path):
                os.makedirs(current_params_path)

            model_store(current_params_path + current_effect + '.json', optimal_params)

            scale = optimal_params['scale']
            opt_params = exclude_keys(optimal_params, 'scale')

            # Process tail with optimized params
            rir_tail = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

            # Merge tail with ER
            merged_rir = merge_er_tail_rir(rir_er, rir_tail, sr, fade_length=fade_in, trim=3)

            current_merged_rir = merged_rir_path + rir_folder[ref_idx] + '_' + current_effect + '.wav'
            sf.write(current_merged_rir, merged_rir.T, sr)

            # Prepare convolution and generate/save reverberated sweep
            conv = pedalboard.Convolution(current_merged_rir, mix=1.0)
            reverberated_sweep = conv(test_sound, sr)

            fig = plt.figure()
            plot_convergence(res_rev)
            plt.savefig('images/convergence/' + rir_folder[ref_idx] + '_' + current_effect + '.pdf')

            plot_melspec_pair(ref[0], reverberated_sweep[0], 2048, 0.25, sr,
                              'images/melspec/' + rir_folder[ref_idx] + '_' + current_effect + '.pdf')

            # Convolve with all input files
            for audio_idx, input_audio in enumerate(audio_file):

                reverb_native = conv(input_audio, sr)
                reverb_norm = normalize_audio(reverb_native, 0.70)

                current_sound = os.listdir('audio/input/sounds/')[audio_idx]

                sf.write(current_rir_path + current_sound.replace('.wav', '') + '_' + current_effect + '.wav',
                         reverb_norm.T, sr)

            with open('audio/' + current_effect + '_times.txt', 'w') as f:
                for item in times:
                    f.write("%s\n" % item)
                f.write('Average time elapsed [s]:' + "%s\n" % np.mean(times))
