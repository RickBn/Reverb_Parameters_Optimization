import functools
from skopt import gp_minimize
from skopt.plots import plot_convergence
import yaml
import pandas as pd

import datetime

from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.plot_functions import plot_melspec_pair
from scripts.utils.json_functions import *
from scripts.params_dim_reduction import get_dim_red_model, reconstruct_original_params
from er_detection import detect_er
import scipy.signal

#plt.switch_backend('agg')
#plt.switch_backend('TkAgg')

n_walls = 6
n_wall_bands = 8


def find_params(rir_path: str,
                er_path: str,
                armodel_path: str,
                merged_rir_path: str,
                vst_rir_path: str,
                params_path: str,
                result_path: str,
                input_path: str,
                fixed_params_path: str = None,
                generate_references: bool = True,
                original_er: bool = False,
                pre_norm: bool = False,
                vst_path: str = "vst3/Real time SDN.vst3",
                n_iterations: int = 200,
                match_only_late: bool = True,
                apply_dim_red: bool = True,
                same_coef_walls: bool = False,
                force_last2_bands_equal: bool = False,
                n_initial_points: int = 10):

    fade_length = 128

    if fixed_params_path is not None:
        with open(fixed_params_path, "r") as stream:
            try:
                fixed_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        fixed_params = dict()

    rir_folder = os.listdir(rir_path)

    venv_name = os.path.basename(os.path.normpath(params_path))

    if match_only_late:
        # armodel_filename = armodel_path + 'rir_offset.npy'
        armodel_filename = armodel_path + 'cut_idx_kl.npy'
        if not os.path.exists(armodel_filename):
            detect_er(venv_name)

        rir_offset = np.load(armodel_filename, allow_pickle=True)[()]

    rir, sr = sf.read(rir_path + rir_folder[0])
    print(sr)

    impulse = create_impulse(sr * 3, n_channels=2)
    sweep = create_log_sweep(1, 20, 20000, sr, 2, n_channels=1)

    test_sound = sweep

    # if match_only_late:
    #     scale_parameter = skopt.space.space.Real(0.0, 1.0, transform='identity')
    #
    #     rev_param_ranges_nat = [scale_parameter, scale_parameter, scale_parameter, scale_parameter]
    #     rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'width': 0.0, 'scale': 0.5}

    rev_external = pedalboard.load_plugin(vst_path)
    rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)

    for i, r in enumerate(rev_param_ranges_ex):
        try:
            if r.bounds == (0, 1):
                rev_param_ranges_ex[i] = skopt.space.space.Real(0.001,
                                                                0.999,
                                                                transform='identity')
        except:
            continue

    # if match_only_late:
    #     rev_param_names_ex['scale'] = 0.5
    #     rev_param_ranges_ex.append(scale_parameter)

    rev_plugins = {'SDN': [rev_external, rev_param_names_ex, rev_param_ranges_ex]}

    input_file_names = os.listdir(input_path)
    result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

    if generate_references:
        batch_fft_convolve(input_path, result_file_names, rir_path, rir_names=None, save_path=result_path,
                           return_convolved=False, scale_factor=1.0, norm=False)

    # Convolve the sweep with RIRs
    reference_audio = batch_fft_convolve([test_sound], result_file_names,
                                         rir_path, rir_names=None, save_path=None, scale_factor=1.0, norm=False)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if not os.path.exists(merged_rir_path) and match_only_late:
        os.makedirs(merged_rir_path)

    if not os.path.exists(vst_rir_path):
        os.makedirs(vst_rir_path)

    overall_time = 0
    times = []

    loss_end = []

    # Iterate over the RIRs of the room
    for ref_idx, ref in enumerate(reference_audio):

        start = datetime.datetime.now()

        rir_file = rir_folder[ref_idx]
        rir_name = rir_file.replace('.wav', '')

        print(f'POSITION: {rir_name}')

        n_channels = ref.shape[0]

        input_audio = np.stack([test_sound] * n_channels)

        if match_only_late:
            rir_er_path = er_path + rir_file

            rir_er, sr_er = sf.read(rir_er_path)
            rir_er = rir_er.T

            # fade_in = int(5 * sr * 0.001)
            fade_len = 128

            offset_list = rir_offset[rir_file]

            ##V2: fade-in prima della loss di late matchata e della RIR originale (per togliere er), poi calcolare la loss
            # rir_late, sr = sf.read(rir_path + rir_file)
            # rir_late = rir_late.T
            # for ch in range(n_channels):
            #     rir_late[ch,:] = rir_late[ch,:] * np.concatenate([np.zeros(int(offset_list[ch])),
            #                                                 cosine_fade(int(len(rir_late[ch, :].T) - offset_list[ch]),
            #                                                             fade_length, False)])
            #
            # rir_late = scipy.signal.fftconvolve(input_audio, rir_late, mode='full', axes=1)
            # ref = rir_late
            ##V2

        else:
            rir_er = None
            offset_list = None

        # current_rir_path = f'{result_path}{rir_name}/'
        # if not os.path.exists(current_rir_path):
        #     os.makedirs(current_rir_path)

        for rev in rev_plugins:

            current_effect = rev
            effect_folder = rev
            rev_plugin = rev_plugins[rev][0]
            rev_param_names = rev_plugins[rev][1]
            rev_param_ranges = rev_plugins[rev][2]

            rir_tail = np.array([])

            # Fixed parameters for the current position
            fixed_params_pos = fixed_params[rir_folder[ref_idx].split('.')[0]]

            # Remove the fixed parameters from the list of parameters tu tune
            fixed_params_bool = [p in fixed_params_pos.keys() for p in rev_param_names.keys()]
            rev_param_ranges_to_tune = [rev_param_ranges[idx] for idx, p in enumerate(fixed_params_bool) if p == False]
            rev_param_names_to_tune = {p: rev_param_names[p] for idx, p in enumerate(rev_param_names)
                                       if fixed_params_bool[idx] == False}

            if same_coef_walls:
                rev_param_ranges_to_tune = rev_param_ranges_to_tune[:int(len(rev_param_ranges_to_tune)/n_walls)]

                # rev_param_names_to_tune_keys = list(rev_param_names_to_tune.keys())
                # temp_dict = dict()
                # for k in rev_param_names_to_tune_keys[:int(len(rev_param_names_to_tune_keys)/n_walls)]:
                #     temp_dict[k] = rev_param_names_to_tune[k]
                # rev_param_names_to_tune = temp_dict

            if force_last2_bands_equal:
                rev_param_ranges_to_tune = [rev_param_ranges_to_tune[i]
                                            for i in range(len(rev_param_ranges_to_tune)) if i%n_wall_bands < 6]

            if apply_dim_red:
                dim_red_mdl = get_dim_red_model()
                dim_red_mdl.x_min = np.floor(dim_red_mdl.x_min * 10) / 10
                dim_red_mdl.x_max = np.ceil(dim_red_mdl.x_max * 10) / 10
                rev_param_ranges_to_tune = []

                # If the walls have the same coeff then consider as if there is only 1 wall
                n_walls_dimred = 1 if same_coef_walls else n_walls

                for n in range(n_walls_dimred):
                    for c in range(dim_red_mdl.n_components):
                        rev_param_ranges_to_tune.append(skopt.space.space.Real(dim_red_mdl.x_min[c],
                                                                               dim_red_mdl.x_max[c],
                                                                               transform='identity'))
            else:
                dim_red_mdl = None

            distance_func = functools.partial(rir_distance,
                                              params_dict=rev_param_names,
                                              input_audio=input_audio,
                                              ref_audio=ref,
                                              rir_er=rir_er,
                                              offset=offset_list,
                                              sample_rate=sr,
                                              vst3=rev_plugin,
                                              merged=original_er,
                                              pre_norm=pre_norm,
                                              fixed_params=fixed_params_pos,
                                              fixed_params_bool=fixed_params_bool,
                                              match_only_late=match_only_late,
                                              dim_red_mdl=dim_red_mdl,
                                              same_coef_walls=same_coef_walls,
                                              force_last2_bands_equal=force_last2_bands_equal,
                                              fade_length=fade_length)

            # start = timeit.default_timer()

            res_rev = gp_minimize(distance_func, rev_param_ranges_to_tune, acq_func="gp_hedge",
                                  n_calls=n_iterations, n_initial_points=n_initial_points, random_state=1234, n_jobs=1)

            fig = plt.figure()
            plot_convergence(res_rev)
            converge_img_path = f'images/convergence/{venv_name}/{rir_name}_{current_effect}.png'
            os.makedirs(os.path.dirname(converge_img_path), exist_ok=True)
            plt.savefig(converge_img_path)

            # stop = timeit.default_timer()

            # time_s = stop - start
            # overall_time += time_s
            # times.append(time_s)

            if dim_red_mdl is not None:
                res_rev.x = reconstruct_original_params(dim_red_mdl, res_rev.x)

            optimal_params = rev_param_names_to_tune

            if force_last2_bands_equal:
                new_params = []
                for i, p in enumerate(res_rev.x):
                    new_params.append(p)
                    if i % (n_wall_bands - 2) == 5:
                        new_params.append(p)
                        new_params.append(p)

                res_rev.x = new_params
                del new_params

            for i, p in enumerate(optimal_params):
                if same_coef_walls:
                    optimal_params[p] = np.float(res_rev.x[i % n_wall_bands])
                else:
                    optimal_params[p] = np.float(res_rev.x[i])

            optimal_params = {**fixed_params_pos, **optimal_params}

            loss_end.append(res_rev.fun)
            dict_to_save = {'loss_end_value': np.float(loss_end[ref_idx]), 'parameters': optimal_params}

            # Save params
            current_params_path = f'{params_path}{rir_name}/{effect_folder}/'
            if not os.path.exists(current_params_path):
                os.makedirs(current_params_path)

            json_store(f'{current_params_path}{current_effect}.json', dict_to_save)
            with open(f'{current_params_path}{current_effect}.yml', 'w') as outfile:
                yaml.dump(dict_to_save, outfile, default_flow_style=False, sort_keys=False)

            # if match_only_late:
            #     scale = optimal_params['scale']
            #     opt_params = exclude_keys(optimal_params, 'scale')
            # else:
            scale = 1
            opt_params = optimal_params

            # Process tail with optimized params

            # rir_tail = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

            rt = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

            # rt = rt * cosine_fade(len(rt), fade_length=abs(len(rir_er) - offset_list), fade_out=False)
            # print(f'Fade Length {abs(len(rir_er) - offset_list)}')
            # print(f'Len ER {len(rir_er)}')

            # if match_only_late:
            #     rir_tail = pad_signal(rt, n_channels, np.max(offset_list), pad_end=False)[:, :(sr * 3)]
            # else:
            rir_tail = rt

            # rir_tail = np.concatenate((rir_tail, rt))

            # Save VST generated RIR tail
            matched_rir_folder = vst_rir_path + current_effect + '/'
            matched_rir_filename = matched_rir_folder + rir_name + '_' + current_effect + '.wav'
            os.makedirs(os.path.dirname(matched_rir_folder), exist_ok=True)
            sf.write(matched_rir_filename, rir_tail.T, sr)

            # Merge tail with ER

            # merged_rir = merge_er_tail_rir(rir_er, np.stack([rir_tail] * rir_er.shape[0]),
            #                                sr, fade_length=fade_in, trim=3)

            if match_only_late:
                # rir_er = pad_signal(rir_er, n_channels, (sr * 3) - rir_er.shape[1], pad_end=True)
                merged_rir = merge_er_tail_rir(rir_er, rir_tail,
                                               sr, fade_length=fade_len, trim=3, offset=offset_list, fade=True)

                merged_rir_folder = merged_rir_path + current_effect + '/'
                merged_rir_filename = merged_rir_folder + rir_name + '_' + current_effect + '.wav'
                os.makedirs(os.path.dirname(merged_rir_folder), exist_ok=True)
                sf.write(merged_rir_filename, merged_rir.T, sr)

        stop = datetime.datetime.now()

        elapsed = stop-start
        times.append(elapsed)

        print(f'{venv_name}-{rir_name} elapsed time: {elapsed}')

    print('FINAL LOSS FUNCTION VALUES')
    for idx, l in enumerate(loss_end):
        print(f'    {rir_folder[idx]}: {l}')


def find_params_merged(rir_path: str,
                       er_path: str,
                       armodel_path: str,
                       merged_rir_path: str,
                       vst_rir_path: str,
                       params_path: str,
                       result_path: str,
                       input_path: str,
                       generate_references: bool = True,
                       pre_norm: bool = False,
                       vst_path: str = "vst3/Real time SDN.vst3"):

    rir_folder = os.listdir(rir_path)
    rir_offset = np.load(armodel_path + 'rir_offset.npy', allow_pickle=True)[()]

    rir, sr = sf.read(rir_path + rir_folder[0])
    print(sr)

    impulse = create_impulse(sr * 3, n_channels=1)
    sweep = create_log_sweep(1, 20, 20000, sr, 2, n_channels=1)

    test_sound = sweep

    scale_parameter = skopt.space.space.Real(0.0, 1.0, transform='identity')

    # rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

    rev_param_ranges_nat = [scale_parameter, scale_parameter, scale_parameter, scale_parameter]
    rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'width': 0.0, 'scale': 0.5}

    rev_external = pedalboard.load_plugin(vst_path)
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
        batch_fft_convolve(input_path, result_file_names, rir_path, rir_names=None, save_path=result_path,
                           return_convolved=False, scale_factor=1.0, norm=False)

    reference_audio = batch_fft_convolve([test_sound], result_file_names,
                                         rir_path, rir_names=None, save_path=None, scale_factor=1.0, norm=False)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if not os.path.exists(merged_rir_path):
        os.makedirs(merged_rir_path)

    if not os.path.exists(vst_rir_path):
        os.makedirs(vst_rir_path)

    overall_time = 0
    times = []

    for ref_idx, ref in enumerate(reference_audio):

        rir_file = rir_folder[ref_idx]
        rir_name = rir_file.replace('.wav', '')
        print(rir_name)

        rir_er_path = er_path + rir_file

        rir_er, sr_er = sf.read(rir_er_path)
        rir_er = rir_er.T

        fade_in = int(5 * sr * 0.001)

        current_rir_path = f'{result_path}{rir_name}/'
        if not os.path.exists(current_rir_path):
            os.makedirs(current_rir_path)

        offset_list = rir_offset[rir_file]

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
                plt.savefig(f'images/convergence/{rir_name}_{current_effect}_{str(ch)}.png')

                #stop = timeit.default_timer()

                # time_s = stop - start
                # overall_time += time_s
                # times.append(time_s)

                optimal_params = rev_param_names

                for i, p in enumerate(optimal_params):
                    optimal_params[p] = res_rev.x[i]

                # Save params
                current_params_path = f'{params_path}{rir_name}/{effect_folder}/'
                if not os.path.exists(current_params_path):
                    os.makedirs(current_params_path)

                json_store(f'{current_params_path}{current_effect}_{ch}.json', optimal_params)

                scale = optimal_params['scale']
                opt_params = exclude_keys(optimal_params, 'scale')

                # Process tail with optimized params

                #rir_tail = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

                rt = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

                rt = rt * cosine_fade(len(rt), fade_length=abs(len(rir_er[ch]) - offset_list[ch]), fade_out=False)
                print(f'Fade Length {abs(len(rir_er[ch]) - offset_list[ch])}')
                print(f'Len ER {len(rir_er[ch])}')

                rt = pad_signal([rt], 1, offset_list[ch], pad_end=False)[:, :(sr*3)]

                if ch == 0:
                    rir_tail = rt
                else:
                    rir_tail = np.concatenate((rir_tail, rt))

            # Save VST generated RIR tail

            sf.write(vst_rir_path + rir_name + '_' + current_effect + '.wav',
                     rir_tail.T, sr)

            # Merge tail with ER

            # merged_rir = merge_er_tail_rir(rir_er, np.stack([rir_tail] * rir_er.shape[0]),
            #                                sr, fade_length=fade_in, trim=3)

            merged_rir = merge_er_tail_rir(rir_er, rir_tail,
                                           sr, fade_length=fade_in, trim=3, fade=False)

            sf.write(merged_rir_path + rir_name + '_' + current_effect + '.wav',
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
            # for audio_idx, input_audio in enumerate(audio_file):
            #
            #     reverb_audio = scipy.signal.fftconvolve(np.stack([input_audio] * merged_rir.shape[0]), merged_rir,
            #                                             mode='full', axes=1)
            #
            #     #reverb_audio = normalize_audio(reverb_audio, 0.70)
            #
            #     current_sound = os.listdir(input_path)[audio_idx]
            #
            #     sf.write(current_rir_path + current_sound.replace('.wav', '') + '_' + current_effect + '.wav',
            #              reverb_audio.T, sr)

            # with open('audio/' + current_effect + '_times.txt', 'w') as f:
            #     for item in times:
            #         f.write("%s\n" % item)
            #     f.write('Average time elapsed [s]:' + "%s\n" % np.mean(times))
