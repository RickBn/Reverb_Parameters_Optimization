import functools
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.plots import plot_convergence
import yaml
import pandas as pd

import plotly.express as px

import datetime

from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.audio.rir_functions import get_rir_wall_reflections_ambisonic
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

coef_bands = ['125hz_wall', '250hz_wall', '500hz_wall', '1000hz_wall', '2000hz_wall', '4000hz_wall', '8000hz_wall', '16000hz_wall']
wall_order = ['omni', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']


def find_params(rir_path: str,
                er_path: str,
                armodel_path: str,
                merged_rir_path: str,
                vst_rir_path: str,
                params_path: str,
                original_params_path: str,
                result_path: str,
                input_path: str,
                optimizer: str = 'gp_minimize',
                optimizer_kappa: float = 1.96,
                optimizer_xi: float = 0.01,
                fixed_params_path: str = None,
                generate_references: bool = True,
                original_er: bool = False,
                pre_norm: bool = False,
                vst_path: str = "vst3/Real time SDN.vst3",
                n_iterations: int = 200,
                match_only_late: bool = True,
                apply_dim_red = True,
                same_coef_walls: bool = False,
                force_last2_bands_equal: bool = False,
                n_initial_points: int = 10,
                inv_interp: bool = False,
                unit_circle: bool = False,
                polar_coords: bool = False,
                fade_length: int = 256,
                remove_direct: bool = False,
                n_jobs: int = 1):

    if fixed_params_path is not None:
        with open(fixed_params_path, "r") as stream:
            try:
                fixed_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        fixed_params = dict()

    format_mode = fixed_params[list(fixed_params.keys())[0]]['output_mode']
    ambisonic = 'ambisonic' in format_mode.lower()

    rir_folder = os.listdir(rir_path)

    venv_name = os.path.basename(os.path.normpath(params_path))

    # Retrieve original params
    with open(original_params_path, "r") as stream:
        try:
            original_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if match_only_late:
        if ambisonic:
            pass
        # armodel_filename = armodel_path + 'rir_offset.npy'
        armodel_filename = armodel_path + 'cut_idx_kl.npy'
        if not os.path.exists(armodel_filename):
            detect_er(venv_name)

        rir_offset = np.load(armodel_filename, allow_pickle=True)[()]

    rir_tmp, sr = sf.read(rir_path + rir_folder[0])
    print(f'Sample rate: {sr}')

    sweep = create_log_sweep(1, 20, 20000, sr, 2, n_channels=1)

    # if match_only_late:
    #     scale_parameter = skopt.space.space.Real(0.0, 1.0, transform='identity')
    #
    #     rev_param_ranges_nat = [scale_parameter, scale_parameter, scale_parameter, scale_parameter]
    #     rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'width': 0.0, 'scale': 0.5}

    if ambisonic:
        vst_path_split = vst_path.split('.')
        vst_path = f'{vst_path_split[0]}_25ch.{vst_path_split[1]}'

    rev_vst = pedalboard.load_plugin(vst_path)
    rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_vst)

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

    rev_plugins = {'SDN': [rev_vst, rev_param_names_ex, rev_param_ranges_ex]}

    input_file_names = os.listdir(input_path)
    result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

    if generate_references:
        batch_fft_convolve(input_path, result_file_names, rir_path, rir_names=None, save_path=result_path,
                           return_convolved=False, scale_factor=1.0, norm=False)

    # Convolve the sweep with RIRs
    target_rirs_sweep, target_rirs = batch_fft_convolve([sweep], result_file_names, rir_path,
                                                        rir_names=None, save_path=None, scale_factor=1.0, norm=False, remove_direct=remove_direct)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if not os.path.exists(merged_rir_path) and match_only_late:
        os.makedirs(merged_rir_path)

    if not os.path.exists(vst_rir_path):
        os.makedirs(vst_rir_path)

    overall_time = 0
    times = []

    loss_end = {}

    optimized_params_dict_rir = {}

    # Iterate over the RIRs of the room
    for ref_idx, target_rir_sweep in enumerate(target_rirs_sweep):

        start = datetime.datetime.now()

        target_rir = target_rirs[ref_idx]

        rir_filename = rir_folder[ref_idx]
        rir_name = rir_filename.replace('.wav', '')

        print(f'POSITION: {rir_name}')

        n_channels_rir = target_rir_sweep.shape[0]
        impulse = create_impulse(rir_tmp.shape[0], n_channels=n_channels_rir)

        if ambisonic:
            n_channels_to_consider = 1
        else:
            n_channels_to_consider = n_channels_rir

        input_sweep = np.stack([sweep] * n_channels_to_consider)

        if match_only_late:
            rir_er_path = er_path + rir_filename

            rir_er, sr_er = sf.read(rir_er_path)
            rir_er = rir_er.T

            # fade_in = int(5 * sr * 0.001)

            offset_list = rir_offset[rir_filename]

            ##V2: fade-in prima della loss di late matchata e della RIR originale (per togliere er), poi calcolare la loss
            # rir_late, sr = sf.read(rir_path + rir_filename)
            # rir_late = rir_late.T
            # for ch in range(n_channels):
            #     rir_late[ch,:] = rir_late[ch,:] * np.concatenate([np.zeros(int(offset_list[ch])),
            #                                                 cosine_fade(int(len(rir_late[ch, :].T) - offset_list[ch]),
            #                                                             fade_length, False)])
            #
            # rir_late = scipy.signal.fftconvolve(input_sweep, rir_late, mode='full', axes=1)
            # target_rir_sweep = rir_late
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

            # rir_tail = np.array([])

            # Fixed parameters for the current position
            fixed_params_pos = fixed_params[rir_folder[ref_idx].split('.')[0]]

            # Remove the fixed parameters from the list of parameters tu tune
            fixed_params_bool = [p in fixed_params_pos.keys() for p in rev_param_names.keys()]
            rev_param_ranges_to_tune = [rev_param_ranges[idx] for idx, p in enumerate(fixed_params_bool) if p == False]
            rev_param_names_to_tune = {p: rev_param_names[p] for idx, p in enumerate(rev_param_names)
                                       if fixed_params_bool[idx] == False}

            if same_coef_walls or ambisonic:
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
                dim_red_mdl = get_dim_red_model(dim_red_alg='pca',
                                                voronoi=False,
                                                inv_interp=inv_interp,
                                                unit_circle=unit_circle,
                                                path=apply_dim_red
                                                # materials_to_exclude=original_params['materials']
                                                )
                dim_red_mdl.x_min = np.floor(dim_red_mdl.x_min * 10) / 10
                dim_red_mdl.x_max = np.ceil(dim_red_mdl.x_max * 10) / 10
                rev_param_ranges_to_tune = []

                # If the walls have the same coeff then consider as if there is only 1 wall
                n_walls_dimred = 1 if same_coef_walls else n_walls

                for n in range(n_walls_dimred):
                    if polar_coords:
                        if dim_red_mdl.n_components == 2:
                            rev_param_ranges_to_tune.append(skopt.space.space.Real(0,
                                                                                   1,
                                                                                   transform='identity'))
                            rev_param_ranges_to_tune.append(skopt.space.space.Real(0,
                                                                                   2 * np.pi,
                                                                                   transform='identity'))
                        else:
                            raise Exception("You cannot use polar coordinate with more than 2 components")
                    else:
                        for c in range(dim_red_mdl.n_components):
                            rev_param_ranges_to_tune.append(skopt.space.space.Real(dim_red_mdl.x_min[c],
                                                                                   dim_red_mdl.x_max[c],
                                                                                   transform='identity'))
            else:
                dim_red_mdl = None

            if ambisonic:
                # Use beamforming to isolate the RIR reflections on the shoebox walls
                target_rir_beamforming, beamformer, engine, playback = get_rir_wall_reflections_ambisonic(target_rir, fixed_params=fixed_params_pos, wall_order=wall_order, sr=sr, order=int(format_mode[0]))
                n_fittings = target_rir_beamforming.shape[0]
            else:
                n_fittings = n_walls
                beamformer = None
                engine = None
                playback = None

            params_omni2d = None

            # Set fixed params
            optimized_params_dict = rev_param_names
            for idx, par in enumerate(optimized_params_dict):
                if fixed_params_bool[idx]:
                    optimized_params_dict[par] = fixed_params_pos[par]

            loss_end[rir_name] = {}
            optimized_params2d_dict = {}

            for n_fit in range(n_fittings):
                print(f'Optimizing {wall_order[n_fit]}...')

                distance_func = functools.partial(rir_distance,
                                                  params_dict=optimized_params_dict,
                                                  input_sweep=input_sweep,
                                                  target_rir=target_rir_beamforming[n_fit:n_fit+1,:] if ambisonic else target_rir_sweep,
                                                  rir_er=rir_er,
                                                  offset=offset_list,
                                                  sample_rate=sr,
                                                  vst3=rev_plugin,
                                                  merged=original_er,
                                                  pre_norm=pre_norm,
                                                  fixed_params_bool=fixed_params_bool,
                                                  match_only_late=match_only_late,
                                                  dim_red_mdl=dim_red_mdl,
                                                  same_coef_walls=same_coef_walls,
                                                  force_last2_bands_equal=force_last2_bands_equal,
                                                  fade_length=fade_length,
                                                  polar_coords=polar_coords,
                                                  impulse=impulse,
                                                  remove_direct=remove_direct,
                                                  wall_idx_ambisonic=n_fit if ambisonic else None,
                                                  beamformer=beamformer,
                                                  engine=engine,
                                                  playback=playback,
                                                  wall_order=wall_order
                                                  )

                # start = timeit.default_timer()

                if optimizer == 'gp_minimize':
                    res_rev = gp_minimize(distance_func, rev_param_ranges_to_tune, acq_func="gp_hedge",
                                          n_calls=n_iterations, n_initial_points=n_initial_points, random_state=1234,
                                          kappa=optimizer_kappa, n_jobs=n_jobs,
                                          x0=params_omni2d,
                                          xi=optimizer_xi)
                elif optimizer == 'forest_minimize':
                    res_rev = forest_minimize(distance_func, rev_param_ranges_to_tune, base_estimator='RF', acq_func="EI",
                                              n_calls=n_iterations, n_initial_points=n_initial_points, random_state=1234,
                                              kappa=optimizer_kappa, n_jobs=n_jobs,
                                              x0=params_omni2d,
                                              xi=optimizer_xi)
                elif optimizer == 'gbrt_minimize':
                    res_rev = gbrt_minimize(distance_func, rev_param_ranges_to_tune, acq_func="LCB",
                                            n_calls=n_iterations, n_initial_points=n_initial_points, random_state=1234,
                                            kappa=optimizer_kappa, n_jobs=n_jobs,
                                            x0=params_omni2d,
                                            xi=optimizer_xi)


                if polar_coords:
                    res_rev.x = pol2cart(res_rev.x)

                params_2d = res_rev.x
                params_optimized = reconstruct_original_params(dim_red_mdl, res_rev.x)

                if force_last2_bands_equal:
                    params_optimized.append(params_optimized[-1])
                    params_optimized.append(params_optimized[-1])

                if ambisonic:
                    loss_end[rir_name][wall_order[n_fit]] = np.float(res_rev.fun)
                    optimized_params2d_dict[wall_order[n_fit]] = [np.float(p) for p in params_2d]

                    if n_fit == 0:
                        params_omni2d = res_rev.x
                        params_omni = params_optimized

                        params_count = 0
                        for idx, par in enumerate(optimized_params_dict):
                            if not fixed_params_bool[idx]:
                                optimized_params_dict[par] = np.float(params_omni[params_count])
                                params_count = (params_count + 1) % n_wall_bands

                    else:
                        for i, b in enumerate(coef_bands):
                            optimized_params_dict[f'{b}_{wall_order[n_fit]}'] = np.float(params_optimized[i])

                else:
                    loss_end[rir_name] = np.float(res_rev.fun)
                    optimized_params2d_dict = [np.float(p) for p in params_2d]
                    params_count = 0
                    for idx, par in enumerate(optimized_params_dict):
                        if not fixed_params_bool[idx]:
                            optimized_params_dict[par] = np.float(params_optimized[params_count])
                            if same_coef_walls:
                                params_count = (params_count + 1) % n_wall_bands
                            else:
                                params_count = params_count + 1

                fig = plt.figure()
                plot_convergence(res_rev)
                converge_img_path = f'images/convergence/{venv_name}/{rir_name}_{current_effect}_{n_fit}.png'
                os.makedirs(os.path.dirname(converge_img_path), exist_ok=True)
                plt.savefig(converge_img_path)

            # stop = timeit.default_timer()

            # time_s = stop - start
            # overall_time += time_s
            # times.append(time_s)

            # if dim_red_mdl is not None:
            #     if polar_coords:
            #         res_rev.x = pol2cart(res_rev.x)
            #     params_2d = res_rev.x
            #     res_rev.x = reconstruct_original_params(dim_red_mdl, res_rev.x)
            #
            # optimal_params = rev_param_names_to_tune
            #
            # if force_last2_bands_equal:
            #     new_params = []
            #     for i, p in enumerate(res_rev.x):
            #         new_params.append(p)
            #         if i % (n_wall_bands - 2) == 5:
            #             new_params.append(p)
            #             new_params.append(p)
            #
            #     res_rev.x = new_params
            #     del new_params
            #
            # for i, p in enumerate(optimal_params):
            #     if same_coef_walls:
            #         optimal_params[p] = np.float(res_rev.x[i % n_wall_bands])
            #     else:
            #         optimal_params[p] = np.float(res_rev.x[i])
            #
            # optimal_params = {**fixed_params_pos, **optimal_params}

            # loss_end[rir_name] = res_rev.fun

            dict_to_save = {'loss_end_value': loss_end[rir_name], 'parameters': optimized_params_dict}
            if dim_red_mdl is not None:
                dict_to_save['parameters_2D'] = optimized_params2d_dict

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

            # pred_params[rir_name] = np.array([optimized_params_dict['125hz_wall_x_0'],
            #                                   optimized_params_dict['250hz_wall_x_0'],
            #                                   optimized_params_dict['500hz_wall_x_0'],
            #                                   optimized_params_dict['1000hz_wall_x_0'],
            #                                   optimized_params_dict['2000hz_wall_x_0'],
            #                                   optimized_params_dict['4000hz_wall_x_0']
            #                                   ])

            # Process tail with optimized params

            # rir_tail = vst_reverb_process(opt_params, impulse, sr, scale_factor=scale, rev_vst=rev_plugin)

            rir_tail = vst_reverb_process(optimized_params_dict, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

            # rt = rt * cosine_fade(len(rt), fade_length=abs(len(rir_er) - offset_list), fade_out=False)
            # print(f'Fade Length {abs(len(rir_er) - offset_list)}')
            # print(f'Len ER {len(rir_er)}')

            # if match_only_late:
            #     rir_tail = pad_signal(rt, n_channels, np.max(offset_list), pad_end=False)[:, :(sr * 3)]
            # else:
            # rir_tail = rt

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
                                               sr, fade_length=fade_length, trim=3, offset=offset_list, fade=True)

                merged_rir_folder = merged_rir_path + current_effect + '/'
                merged_rir_filename = merged_rir_folder + rir_name + '_' + current_effect + '.wav'
                os.makedirs(os.path.dirname(merged_rir_folder), exist_ok=True)
                sf.write(merged_rir_filename, merged_rir.T, sr)

            optimized_params_dict_rir[rir_name] = optimized_params_dict

        stop = datetime.datetime.now()

        elapsed = stop-start
        times.append(elapsed)

        print(f'{venv_name}-{rir_name} elapsed time: {elapsed}')

    print('MATCH RESULTS')
    for rir_name in rir_folder:
        abs_err = pd.DataFrame(None, columns=coef_bands + ['Mean per wall'], index=wall_order[1:]  + ['Mean per band'])
        r_n = rir_name.rstrip('.wav')

        for wall in wall_order[1:]:
            orig = np.array([original_params[r_n][f'{b}_{wall}'] for b in coef_bands])
            pred = np.array([optimized_params_dict_rir[r_n][f'{b}_{wall}'] for b in coef_bands])

        # pred = pred_params[r_n]

            abs_err.loc[wall,coef_bands] = abs(orig - pred)

        mae_wall = abs_err.mean(axis=1)
        mae_band = abs_err.mean(axis=0)
        mae_overall = abs_err.loc[wall_order[1:],coef_bands].values.mean()

        abs_err.loc['Mean per band', coef_bands] = mae_band[coef_bands]
        abs_err.loc[wall_order[1:], 'Mean per wall'] = mae_wall[wall_order[1:]]
        abs_err.loc['Mean per band', 'Mean per wall'] = mae_overall

        print(f' -> {r_n}:')
        # print(f'     - Loss: {round(loss_end[r_n], 2)} dB')
        # print(f'     - Parameters abs error: {abs_err}')
        print()
        print('     - MAE per wall')
        print(mae_wall[wall_order[1:]])
        print()
        print('     - MAE per band')
        print(mae_band[coef_bands])
        print(f'     - Parameters MAE overall: {round(mae_overall, 3)}')

        fig = px.imshow(abs_err, title="Absolute error")
        fig.update_layout(xaxis_title="Frequency band", yaxis_title="Wall", xaxis={'side': 'top'})
        fig.show()

        current_params_path = f'{params_path}{r_n}/{effect_folder}/abs_err'

        abs_err.to_csv(f'{current_params_path}.csv')
        fig.write_html(f'{current_params_path}.html')




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
