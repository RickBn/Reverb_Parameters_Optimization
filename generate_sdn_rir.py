from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
# import pedalboard
from pedalboard_change_channel_limit import pedalboard
import soundfile as sf
import os
import yaml
from random import random, uniform, randrange, gauss
import pandas as pd
import warnings

sr = 48000
rir_len_sec = 3
scale = 1
n_coef_bands = 8
n_walls = 6
vst_path = "vst3/Real time SDN_25ch.vst3"
base_save_path = "audio/input/chosen_rirs/stereo/"
fixed_params_path = 'fixed_parameters/SDN'
dict_walls = {'x_0': 0,
              'x_1': 1,
              'y_0': 2,
              'y_1': 3,
              'z_0': 4,
              'z_1': 5
              }
bands_idx = [0, 1, 2, 3, 4, 5, 5, 5]

dimensions_x = 3
dimensions_y = 3
dimensions_z = 4

output_mode = '4th order Ambisonic'

params = {
    'pos01': {
        'output_mode': output_mode,
        'source_gain_db': -12,
        'render_line_of_sight': False,
        'air_absorption': False,
        'source_x': 0.25,
        'source_y': 0.5,
        'source_z': 0.5,
        'listener_x': 0.75,
        'listener_y': 0.5,
        'listener_z': 0.5,
        'listener_pitch': 0.0,# Pitch, yaw, roll not used for Ambisonics output
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x_m': dimensions_x,
        'dimensions_y_m': dimensions_y,
        'dimensions_z_m': dimensions_z,
        '125hz_wall_x_0': 0.5,
        '250hz_wall_x_0': 0.4,
        '500hz_wall_x_0': 0.45,
        '1000hz_wall_x_0': 0.45,
        '2000hz_wall_x_0': 0.6,
        '4000hz_wall_x_0': 0.7,
        '8000hz_wall_x_0': 0.7,
        '16000hz_wall_x_0': 0.7,
        '125hz_wall_x_1': 0.49,
        '250hz_wall_x_1': 0.66,
        '500hz_wall_x_1': 0.8,
        '1000hz_wall_x_1': 0.88,
        '2000hz_wall_x_1': 0.82,
        '4000hz_wall_x_1': 0.7,
        '8000hz_wall_x_1': 0.7,
        '16000hz_wall_x_1': 0.7,
        '125hz_wall_y_0': 0.2,
        '250hz_wall_y_0': 0.35,
        '500hz_wall_y_0': 0.55,
        '1000hz_wall_y_0': 0.3,
        '2000hz_wall_y_0': 0.25,
        '4000hz_wall_y_0': 0.3,
        '8000hz_wall_y_0': 0.3,
        '16000hz_wall_y_0': 0.3,
        '125hz_wall_y_1': 0.51,
        '250hz_wall_y_1': 0.64,
        '500hz_wall_y_1': 0.75,
        '1000hz_wall_y_1': 0.8,
        '2000hz_wall_y_1': 0.82,
        '4000hz_wall_y_1': 0.83,
        '8000hz_wall_y_1': 0.83,
        '16000hz_wall_y_1': 0.83,
        '125hz_wall_z_0': 0.65,
        '250hz_wall_z_0': 0.71,
        '500hz_wall_z_0': 0.82,
        '1000hz_wall_z_0': 0.86,
        '2000hz_wall_z_0': 0.76,
        '4000hz_wall_z_0': 0.62,
        '8000hz_wall_z_0': 0.62,
        '16000hz_wall_z_0': 0.62,
        '125hz_wall_z_1': 0.11,
        '250hz_wall_z_1': 0.14,
        '500hz_wall_z_1': 0.37,
        '1000hz_wall_z_1': 0.43,
        '2000hz_wall_z_1': 0.27,
        '4000hz_wall_z_1': 0.25,
        '8000hz_wall_z_1': 0.25,
        '16000hz_wall_z_1': 0.25,
    },
    # 'pos02': {
    #     'output_mode': output_mode,
    #     'source_gain_db': 0,
    #     'render_line_of_sight': False,
    #     'source_x': 0.75,
    #     'source_y': 0.75,
    #     'source_z': 0.5,
    #     'listener_x': 0.25,
    #     'listener_y': 0.25,
    #     'listener_z': 0.5,
    #     'listener_pitch': 0.0,
    #     'listener_yaw': 0.0,
    #     'listener_roll': 0.0,
    #     'dimensions_x_m': dimensions_x,
    #     'dimensions_y_m': dimensions_y,
    #     'dimensions_z_m': dimensions_z,
    #     '125hz_wall_x_0': 0.31,
    #     '250hz_wall_x_0': 0.33,
    #     '500hz_wall_x_0': 0.14,
    #     '1000hz_wall_x_0': 0.1,
    #     '2000hz_wall_x_0': 0.1,
    #     '4000hz_wall_x_0': 0.12,
    #     '8000hz_wall_x_0': 0.12,
    #     '16000hz_wall_x_0': 0.12,
    #     '125hz_wall_x_1': 0.31,
    #     '250hz_wall_x_1': 0.33,
    #     '500hz_wall_x_1': 0.14,
    #     '1000hz_wall_x_1': 0.1,
    #     '2000hz_wall_x_1': 0.1,
    #     '4000hz_wall_x_1': 0.12,
    #     '8000hz_wall_x_1': 0.12,
    #     '16000hz_wall_x_1': 0.12,
    #     '125hz_wall_y_0': 0.31,
    #     '250hz_wall_y_0': 0.33,
    #     '500hz_wall_y_0': 0.14,
    #     '1000hz_wall_y_0': 0.1,
    #     '2000hz_wall_y_0': 0.1,
    #     '4000hz_wall_y_0': 0.12,
    #     '8000hz_wall_y_0': 0.12,
    #     '16000hz_wall_y_0': 0.12,
    #     '125hz_wall_y_1': 0.31,
    #     '250hz_wall_y_1': 0.33,
    #     '500hz_wall_y_1': 0.14,
    #     '1000hz_wall_y_1': 0.1,
    #     '2000hz_wall_y_1': 0.1,
    #     '4000hz_wall_y_1': 0.12,
    #     '8000hz_wall_y_1': 0.12,
    #     '16000hz_wall_y_1': 0.12,
    #     '125hz_wall_z_0': 0.31,
    #     '250hz_wall_z_0': 0.33,
    #     '500hz_wall_z_0': 0.14,
    #     '1000hz_wall_z_0': 0.1,
    #     '2000hz_wall_z_0': 0.1,
    #     '4000hz_wall_z_0': 0.12,
    #     '8000hz_wall_z_0': 0.12,
    #     '16000hz_wall_z_0': 0.12,
    #     '125hz_wall_z_1': 0.31,
    #     '250hz_wall_z_1': 0.33,
    #     '500hz_wall_z_1': 0.14,
    #     '1000hz_wall_z_1': 0.1,
    #     '2000hz_wall_z_1': 0.1,
    #     '4000hz_wall_z_1': 0.12,
    #     '8000hz_wall_z_1': 0.12,
    #     '16000hz_wall_z_1': 0.12,
    # }
}

def generate_rirs(param_type='dict', same_coef_per_wall=False):

    if param_type == 'random':
        if same_coef_per_wall:
            coef = np.random.rand(n_coef_bands)

            counter = 0
            for pos in params.keys():
                for par in params[pos].keys():
                    if '_wall_' in par:
                        params[pos][par] = float(coef[counter])
                        counter = (counter + 1) % n_coef_bands

        else:
            for p in params['pos01'].keys():
                if '_wall_' in p:
                    params['pos01'][p] = random()
                    params['pos02'][p] = params['pos01'][p]

    elif param_type == 'sample':
        coef_path = r'.\wall_coeff_dim_reduction\PCA_data\324000_iterations\filters_data.csv'
        coef = pd.read_csv(coef_path, header=None).values

        rnd_coef_idx = []
        walls_lbl = list(dict_walls.keys())
        for w in range(n_walls):
            rnd_coef_idx.append(randrange(coef.shape[0]))
            params['pos01'][f'coef_idx_{walls_lbl[w]}'] = rnd_coef_idx[-1]

        band_count = 0
        for p in params['pos01'].keys():
            if '_wall_' in p:
                params['pos01'][p] = float(coef[rnd_coef_idx[dict_walls[p[-3:]]], bands_idx[(band_count % n_coef_bands)]])

                band_count += 1

            if p == 'source_x' or p == 'source_y' or p == 'source_z' or p == 'listener_x' or p == 'listener_y' or p == 'listener_z':
                params['pos01'][p] = float(uniform(0.01, 0.99))

            if p.startswith('dimensions_'):
                params['pos01'][p] = float(np.max((1, gauss(7, 5))))

    # Create impulse
    if output_mode == 'Mono':
        n_ch = 1
    elif output_mode.endswith('order Ambisonic'):
        n_ch = pow(int(output_mode[0]) + 1, 2)
    else:
        n_ch = 2

    impulse = create_impulse(sr * rir_len_sec, n_channels=n_ch, amp=1)

    # Load SDN VST plugin
    rev_plugin = pedalboard.load_plugin(vst_path)
    # rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_plugin)

    # Find the last SDN folder
    folder_counter = 0
    for folder in os.listdir(base_save_path):
        if folder.startswith("SDN"):
            folder_n = int(folder.split('SDN')[1])
            if folder_n >= folder_counter:
                folder_counter = int(folder.split('SDN')[1]) + 1

    save_folder = f'{base_save_path}SDN{folder_counter:03d}/_todo'
    os.makedirs(save_folder)

    fixed_params = dict()

    # Iterate over the positions
    for pos_key, pos_param in params.items():
        # print(rev_plugin.render_line_of_sight)
        # Generate RIR
        sdn_ir = vst_reverb_process(pos_param, impulse, sr, scale_factor=scale, rev_external=rev_plugin,
                                    norm=False)

        # if np.any(sdn_ir >= 1):
        #     warnings.warn('Amplitude >= 1 found!')
        # max_amp = np.max(sdn_ir)
        # sdn_ir = sdn_ir / max_amp

        # Save RIR
        sf.write(f'{save_folder}/{pos_key}.wav', sdn_ir.T, sr, subtype='FLOAT')

        # Retrieve fixed parameters
        fixed_params[pos_key] = {k: pos_param[k] for k in pos_param if 'hz_wall' not in k}

        # Add scale
        params[pos_key]['scale'] = scale

    # Save parameters
    with open(f'{save_folder}/../parameters.yml', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False, sort_keys=False)

    # Save fixed parameters
    with open(f'{fixed_params_path}/SDN{folder_counter:03d}.yml', 'w') as outfile:
        yaml.dump(fixed_params, outfile, default_flow_style=False, sort_keys=False)

    return f'SDN{folder_counter:03d}'


if __name__ == "__main__":
    # rand_coeff = False
    # dict, sample, random
    param_type = 'sample'
    same_coef_per_wall = True

    rir_name = generate_rirs(param_type, same_coef_per_wall)
