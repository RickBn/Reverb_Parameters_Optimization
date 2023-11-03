from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
# import pedalboard
from pedalboard_change_channel_limit import pedalboard
import soundfile as sf
import os
import yaml
from random import random
import warnings

sr = 48000
rir_len_sec = 3
scale = 1
n_coef_bands = 8
vst_path = "vst3/Real time SDN_25ch.vst3"
base_save_path = "audio/input/chosen_rirs/stereo/"
fixed_params_path = 'fixed_parameters/SDN'

dimensions_x = 7
dimensions_y = 12
dimensions_z = 4

output_mode = '4th order Ambisonic'

params = {
    'pos01': {
        'output_mode': output_mode,
        'source_gain_db': 0,
        'render_line_of_sight': False,
        'source_x': 0.5,
        'source_y': 0.5,
        'source_z': 0.5,
        'listener_x': 0.4,
        'listener_y': 0.6,
        'listener_z': 0.6,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x_m': dimensions_x,
        'dimensions_y_m': dimensions_y,
        'dimensions_z_m': dimensions_z,
        '125hz_wall_x_0': 0.04,
        '250hz_wall_x_0': 0.14,
        '500hz_wall_x_0': 0.49,
        '1000hz_wall_x_0': 0.35,
        '2000hz_wall_x_0': 0.31,
        '4000hz_wall_x_0': 0.36,
        '8000hz_wall_x_0': 0.36,
        '16000hz_wall_x_0': 0.36,
        '125hz_wall_x_1': 0.04,
        '250hz_wall_x_1': 0.14,
        '500hz_wall_x_1': 0.49,
        '1000hz_wall_x_1': 0.35,
        '2000hz_wall_x_1': 0.31,
        '4000hz_wall_x_1': 0.36,
        '8000hz_wall_x_1': 0.36,
        '16000hz_wall_x_1': 0.36,
        '125hz_wall_y_0': 0.04,
        '250hz_wall_y_0': 0.14,
        '500hz_wall_y_0': 0.49,
        '1000hz_wall_y_0': 0.35,
        '2000hz_wall_y_0': 0.31,
        '4000hz_wall_y_0': 0.36,
        '8000hz_wall_y_0': 0.36,
        '16000hz_wall_y_0': 0.36,
        '125hz_wall_y_1': 0.04,
        '250hz_wall_y_1': 0.14,
        '500hz_wall_y_1': 0.49,
        '1000hz_wall_y_1': 0.35,
        '2000hz_wall_y_1': 0.31,
        '4000hz_wall_y_1': 0.36,
        '8000hz_wall_y_1': 0.36,
        '16000hz_wall_y_1': 0.36,
        '125hz_wall_z_0': 0.04,
        '250hz_wall_z_0': 0.14,
        '500hz_wall_z_0': 0.49,
        '1000hz_wall_z_0': 0.35,
        '2000hz_wall_z_0': 0.31,
        '4000hz_wall_z_0': 0.36,
        '8000hz_wall_z_0': 0.36,
        '16000hz_wall_z_0': 0.36,
        '125hz_wall_z_1': 0.04,
        '250hz_wall_z_1': 0.14,
        '500hz_wall_z_1': 0.49,
        '1000hz_wall_z_1': 0.35,
        '2000hz_wall_z_1': 0.31,
        '4000hz_wall_z_1': 0.36,
        '8000hz_wall_z_1': 0.36,
        '16000hz_wall_z_1': 0.36,
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

if __name__ == "__main__":
    # Whether to generate the coefficients randomly or use the ones hard coded above
    rand_coeff = False
    same_coef_per_wall = True

    if rand_coeff:
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
        sdn_ir = vst_reverb_process(pos_param, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

        # if np.any(sdn_ir >= 1):
        #     warnings.warn('Amplitude >= 1 found!')
        # max_amp = np.max(sdn_ir)
        # sdn_ir = sdn_ir / max_amp

        # Save RIR
        sf.write(f'{save_folder}/{pos_key}.wav', sdn_ir.T, sr)

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
