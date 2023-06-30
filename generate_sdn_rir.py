from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
import pedalboard
import soundfile as sf
import os
import yaml

sr = 48000
rir_len_sec = 3
scale = 1
vst_path = "vst3/Real time SDN.vst3"
base_save_path = "audio/input/chosen_rirs/stereo/"
fixed_params_path = 'fixed_parameters/SDN'

dimensions_x = 5
dimensions_y = 7
dimensions_z = 3

params = {
    'pos01': {
        'output_mode': "Stereo",
        'source_gain_db': 0,
        'render_line_of_sight': False,
        'source_x': 0.1,
        'source_y': 0.1,
        'source_z': 0.5,
        'listener_x': 0.6,
        'listener_y': 0.8,
        'listener_z': 0.4,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x_m': dimensions_x,
        'dimensions_y_m': dimensions_y,
        'dimensions_z_m': dimensions_z,
        '125hz_wall_x_0': 0.01,
        '250hz_wall_x_0': 0.01,
        '500hz_wall_x_0': 0.01,
        '1000hz_wall_x_0': 0.02,
        '2000hz_wall_x_0': 0.02,
        '4000hz_wall_x_0': 0.02,
        '8000hz_wall_x_0': 0.02,
        '16000hz_wall_x_0': 0.02,
        '125hz_wall_x_1': 0.015,
        '250hz_wall_x_1': 0.015,
        '500hz_wall_x_1': 0.015,
        '1000hz_wall_x_1': 0.025,
        '2000hz_wall_x_1': 0.025,
        '4000hz_wall_x_1': 0.025,
        '8000hz_wall_x_1': 0.025,
        '16000hz_wall_x_1': 0.025,
        '125hz_wall_y_0': 0.03,
        '250hz_wall_y_0': 0.03,
        '500hz_wall_y_0': 0.02,
        '1000hz_wall_y_0': 0.03,
        '2000hz_wall_y_0': 0.04,
        '4000hz_wall_y_0': 0.05,
        '8000hz_wall_y_0': 0.05,
        '16000hz_wall_y_0': 0.05,
        '125hz_wall_y_1': 0.31,
        '250hz_wall_y_1': 0.33,
        '500hz_wall_y_1': 0.14,
        '1000hz_wall_y_1': 0.10,
        '2000hz_wall_y_1': 0.10,
        '4000hz_wall_y_1': 0.12,
        '8000hz_wall_y_1': 0.12,
        '16000hz_wall_y_1': 0.12,
        '125hz_wall_z_0': 0.50,
        '250hz_wall_z_0': 0.10,
        '500hz_wall_z_0': 0.30,
        '1000hz_wall_z_0': 0.50,
        '2000hz_wall_z_0': 0.65,
        '4000hz_wall_z_0': 0.70,
        '8000hz_wall_z_0': 0.80,
        '16000hz_wall_z_0': 0.95,
        '125hz_wall_z_1': 0.01,
        '250hz_wall_z_1': 0.01,
        '500hz_wall_z_1': 0.01,
        '1000hz_wall_z_1': 0.01,
        '2000hz_wall_z_1': 0.02,
        '4000hz_wall_z_1': 0.02,
        '8000hz_wall_z_1': 0.02,
        '16000hz_wall_z_1': 0.02
    },
    'pos02': {
        'output_mode': "Stereo",
        'source_gain_db': 0,
        'render_line_of_sight': False,
        'source_x': 0.5,
        'source_y': 0.8,
        'source_z': 0.8,
        'listener_x': 0.2,
        'listener_y': 0.5,
        'listener_z': 0.5,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x_m': dimensions_x,
        'dimensions_y_m': dimensions_y,
        'dimensions_z_m': dimensions_z,
        '125hz_wall_x_0': 0.01,
        '250hz_wall_x_0': 0.01,
        '500hz_wall_x_0': 0.01,
        '1000hz_wall_x_0': 0.02,
        '2000hz_wall_x_0': 0.02,
        '4000hz_wall_x_0': 0.02,
        '8000hz_wall_x_0': 0.02,
        '16000hz_wall_x_0': 0.02,
        '125hz_wall_x_1': 0.015,
        '250hz_wall_x_1': 0.015,
        '500hz_wall_x_1': 0.015,
        '1000hz_wall_x_1': 0.025,
        '2000hz_wall_x_1': 0.025,
        '4000hz_wall_x_1': 0.025,
        '8000hz_wall_x_1': 0.025,
        '16000hz_wall_x_1': 0.025,
        '125hz_wall_y_0': 0.03,
        '250hz_wall_y_0': 0.03,
        '500hz_wall_y_0': 0.02,
        '1000hz_wall_y_0': 0.03,
        '2000hz_wall_y_0': 0.04,
        '4000hz_wall_y_0': 0.05,
        '8000hz_wall_y_0': 0.05,
        '16000hz_wall_y_0': 0.05,
        '125hz_wall_y_1': 0.31,
        '250hz_wall_y_1': 0.33,
        '500hz_wall_y_1': 0.14,
        '1000hz_wall_y_1': 0.10,
        '2000hz_wall_y_1': 0.10,
        '4000hz_wall_y_1': 0.12,
        '8000hz_wall_y_1': 0.12,
        '16000hz_wall_y_1': 0.12,
        '125hz_wall_z_0': 0.50,
        '250hz_wall_z_0': 0.10,
        '500hz_wall_z_0': 0.30,
        '1000hz_wall_z_0': 0.50,
        '2000hz_wall_z_0': 0.65,
        '4000hz_wall_z_0': 0.70,
        '8000hz_wall_z_0': 0.80,
        '16000hz_wall_z_0': 0.95,
        '125hz_wall_z_1': 0.01,
        '250hz_wall_z_1': 0.01,
        '500hz_wall_z_1': 0.01,
        '1000hz_wall_z_1': 0.01,
        '2000hz_wall_z_1': 0.02,
        '4000hz_wall_z_1': 0.02,
        '8000hz_wall_z_1': 0.02,
        '16000hz_wall_z_1': 0.02
    }
}

if __name__ == "__main__":
    # Create impulse
    impulse = create_impulse(sr * rir_len_sec, n_channels=2)

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

        # Generate RIR
        sdn_ir = vst_reverb_process(pos_param, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

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
