import functools
from skopt import gp_minimize
from skopt.plots import plot_convergence

import timeit

from scripts.parameters_learning import *
from scripts.audio_functions.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.plot_functions import plot_melspec_pair
from scripts.utils.json_functions import *
import scipy.signal

#plt.switch_backend('agg')
#plt.switch_backend('TkAgg')


def find_params_merged(rir_path: str,
                       er_path: str,
                       armodel_path: str,
                       merged_rir_path: str,
                       vst_rir_path: str,
                       result_path: str,
                       input_path: str,
                       generate_references: bool = True,
                       pre_norm: bool = True):

    rir_file = os.listdir(rir_path)
    rir_folder = os.listdir(rir_path)
    rir_offset = np.load(armodel_path + 'rir_offset.npy', allow_pickle=True)[()]

    rir, sr = sf.read(rir_path + rir_file[0])
    print(sr)

    impulse = create_impulse(sr * 3, n_channels=1)
    sweep = create_log_sweep(1, 20, 20000, sr, 2, n_channels=1)

    test_sound = sweep

    scale_parameter = skopt.space.space.Real(0.0, 1.0, transform='identity')

    # rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

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

    # rev_plugins = {'Freeverb': [None, rev_param_names_nat, rev_param_ranges_nat],
    #                'FdnReverb': [rev_external, rev_param_names_ex, rev_param_ranges_ex]}

    rev_plugins = {'Freeverb': [None, rev_param_names_nat, rev_param_ranges_nat]}
    #rev_plugins = {'FdnReverb': [rev_external, rev_param_names_ex, rev_param_ranges_ex]}

    # convolution = prepare_batch_pb_convolve(rir_path, mix=1.0)
    audio_file = prepare_batch_input_multichannel(input_path, num_channels=1)

    input_file_names = os.listdir(input_path)
    result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

    if generate_references:
        batch_fft_convolve(input_path, result_file_names, rir_path, result_path, scale_factor=1.0, norm=False)

    reference_audio = batch_fft_convolve([test_sound], result_file_names,
                                         rir_path, save_path=None, scale_factor=1.00, norm=False)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if not os.path.exists(merged_rir_path):
        os.makedirs(merged_rir_path)

    if not os.path.exists(vst_rir_path):
        os.makedirs(vst_rir_path)

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

        offset_list = rir_offset[rir_folder[ref_idx]]

        for rev in rev_plugins:

            current_effect = rev
            effect_folder = rev
            rev_plugin = rev_plugins[rev][0]
            rev_param_names = rev_plugins[rev][1]
            rev_param_ranges = rev_plugins[rev][2]

            rir_tail = np.array([])

            for ch in range(0, ref.shape[0]):

                print(len(rir_er[ch]))
                print(abs(len(rir_er[ch]) - offset_list[ch]))

                # distance_func = functools.partial(merged_rir_distance,
                #                 #                                   params_dict=rev_param_names,
                #                 #                                   input_audio=test_sound,
                #                 #                                   ref_audio=ref,
                #                 #                                   er_path=rir_er_path,
                #                 #                                   sample_rate=sr,
                #                 #                                   vst3=rev_plugin,
                #                 #                                   pre_norm=pre_norm)

                distance_func = functools.partial(merged_rir_distance_1D,
                                                  params_dict=rev_param_names,
                                                  input_audio=test_sound,
                                                  ref_audio=ref[ch],
                                                  rir_er=rir_er[ch],
                                                  offset=offset_list[ch],
                                                  sample_rate=sr,
                                                  vst3=rev_plugin,
                                                  pre_norm=pre_norm)

                #start = timeit.default_timer()

                res_rev = gp_minimize(distance_func, rev_param_ranges, acq_func="gp_hedge",
                                      n_calls=100, n_random_starts=10, random_state=1234)

                fig = plt.figure()
                plot_convergence(res_rev)
                plt.savefig('images/convergence/' + rir_folder[ref_idx] + '_' + current_effect + '_' + str(ch) + '.png')

                #stop = timeit.default_timer()

                # time_s = stop - start
                # overall_time += time_s
                # times.append(time_s)

                optimal_params = rev_param_names

                for i, p in enumerate(optimal_params):
                    optimal_params[p] = res_rev.x[i]

                # Save params
                current_params_path = f'audio/params/' + rir_folder[ref_idx].replace('.wav', '') + \
                                      '/' + effect_folder + '/'

                if not os.path.exists(current_params_path):
                    os.makedirs(current_params_path)

                model_store(current_params_path + current_effect + '_' + str(ch) + '.json', optimal_params)

                scale = optimal_params['scale']
                opt_params = exclude_keys(optimal_params, 'scale')

                # Process tail with optimized params

                #rir_tail = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

                rt = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)
                rt = rt * cosine_fade(len(rt), fade_length=abs(len(rir_er[ch]) - offset_list[ch]), fade_out=False)
                rt = pad_signal([rt], 1, offset_list[ch], pad_end=False)[:, :(sr*3)]

                if ch == 0:
                    rir_tail = rt
                else:
                    rir_tail = np.concatenate((rir_tail, rt))

            # Save VST generated RIR tail

            sf.write(vst_rir_path + rir_folder[ref_idx].replace('.wav', '') + '_' + current_effect + '.wav',
                     rir_tail.T, sr)

            # Merge tail with ER

            # merged_rir = merge_er_tail_rir(rir_er, np.stack([rir_tail] * rir_er.shape[0]),
            #                                sr, fade_length=fade_in, trim=3)

            merged_rir = merge_er_tail_rir(rir_er, rir_tail,
                                           sr, fade_length=fade_in, trim=3, fade=False)

            sf.write(merged_rir_path + rir_folder[ref_idx].replace('.wav', '') + '_' + current_effect + '.wav',
                     merged_rir.T, sr)

            # Prepare convolution and generate/save reverberated sweep

            # reverberated_sweep = scipy.signal.fftconvolve(np.stack([test_sound] * merged_rir.shape[0]), merged_rir,
            #                                              mode='full', axes=1)
            # fig = plt.figure()
            # plot_convergence(res_rev)
            # plt.savefig('images/convergence/' + rir_folder[ref_idx] + '_' + current_effect + '.pdf')
            #
            # plot_melspec_pair(ref[0], reverberated_sweep[0], 2048, 0.25, sr,
            #                   'images/melspec/' + rir_folder[ref_idx] + '_' + current_effect + '.pdf')

            # Convolve with all input files
            for audio_idx, input_audio in enumerate(audio_file):

                reverb_audio = scipy.signal.fftconvolve(np.stack([input_audio] * merged_rir.shape[0]), merged_rir,
                                                        mode='full', axes=1)

                #reverb_audio = normalize_audio(reverb_audio, 0.70)

                current_sound = os.listdir(input_path)[audio_idx]

                sf.write(current_rir_path + current_sound.replace('.wav', '') + '_' + current_effect + '.wav',
                         reverb_audio.T, sr)

            # with open('audio/' + current_effect + '_times.txt', 'w') as f:
            #     for item in times:
            #         f.write("%s\n" % item)
            #     f.write('Average time elapsed [s]:' + "%s\n" % np.mean(times))
