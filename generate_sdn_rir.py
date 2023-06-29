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

params = {
    'pos01': {
        # 'output_mode': "Stereo",
        # 'source_gain_db': 0,
        # 'render_line_of_sight': False,
        'source_x': 0.4,
        'source_y': 0.75,
        'source_z': 0.5,
        'listener_x': 0.6,
        'listener_y': 0.25,
        'listener_z': 0.5,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x': 10.0,
        'dimensions_y': 15.0,
        'dimensions_z': 10.0,
        '125hz_wall_x_0': 0.5,
        '250hz_wall_x_0': 0.5,
        '500hz_wall_x_0': 0.5,
        '1000hz_wall_x_0': 0.5,
        '2000hz_wall_x_0': 0.6,
        '4000hz_wall_x_0': 0.6,
        '8000hz_wall_x_0': 0.6,
        '16000hz_wall_x_0': 0.7,
        '125hz_wall_x_1': 0.5,
        '250hz_wall_x_1': 0.5,
        '500hz_wall_x_1': 0.5,
        '1000hz_wall_x_1': 0.5,
        '2000hz_wall_x_1': 0.7,
        '4000hz_wall_x_1': 0.8,
        '8000hz_wall_x_1': 0.8,
        '16000hz_wall_x_1': 0.9,
        '125hz_wall_y_0': 0.5,
        '250hz_wall_y_0': 0.5,
        '500hz_wall_y_0': 0.5,
        '1000hz_wall_y_0': 0.5,
        '2000hz_wall_y_0': 0.5,
        '4000hz_wall_y_0': 0.5,
        '8000hz_wall_y_0': 0.5,
        '16000hz_wall_y_0': 0.5,
        '125hz_wall_y_1': 0.5,
        '250hz_wall_y_1': 0.5,
        '500hz_wall_y_1': 0.5,
        '1000hz_wall_y_1': 0.5,
        '2000hz_wall_y_1': 0.5,
        '4000hz_wall_y_1': 0.5,
        '8000hz_wall_y_1': 0.5,
        '16000hz_wall_y_1': 0.5,
        '125hz_wall_z_0': 0.5,
        '250hz_wall_z_0': 0.5,
        '500hz_wall_z_0': 0.5,
        '1000hz_wall_z_0': 0.5,
        '2000hz_wall_z_0': 0.5,
        '4000hz_wall_z_0': 0.5,
        '8000hz_wall_z_0': 0.5,
        '16000hz_wall_z_0': 0.5,
        '125hz_wall_z_1': 0.5,
        '250hz_wall_z_1': 0.5,
        '500hz_wall_z_1': 0.5,
        '1000hz_wall_z_1': 0.5,
        '2000hz_wall_z_1': 0.5,
        '4000hz_wall_z_1': 0.5,
        '8000hz_wall_z_1': 0.5,
        '16000hz_wall_z_1': 0.5
    },
    'pos02': {
        # 'output_mode': "Stereo",
        # 'source_gain_db': 0,
        # 'render_line_of_sight': False,
        'source_x': 0.4,
        'source_y': 0.75,
        'source_z': 0.5,
        'listener_x': 0.1,
        'listener_y': 0.8,
        'listener_z': 0.6,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x': 10.0,
        'dimensions_y': 15.0,
        'dimensions_z': 10.0,
        '125hz_wall_x_0': 0.6,
        '250hz_wall_x_0': 0.6,
        '500hz_wall_x_0': 0.6,
        '1000hz_wall_x_0': 0.5,
        '2000hz_wall_x_0': 0.6,
        '4000hz_wall_x_0': 0.6,
        '8000hz_wall_x_0': 0.6,
        '16000hz_wall_x_0': 0.7,
        '125hz_wall_x_1': 0.5,
        '250hz_wall_x_1': 0.5,
        '500hz_wall_x_1': 0.6,
        '1000hz_wall_x_1': 0.5,
        '2000hz_wall_x_1': 0.7,
        '4000hz_wall_x_1': 0.8,
        '8000hz_wall_x_1': 0.6,
        '16000hz_wall_x_1': 0.6,
        '125hz_wall_y_0': 0.5,
        '250hz_wall_y_0': 0.6,
        '500hz_wall_y_0': 0.5,
        '1000hz_wall_y_0': 0.5,
        '2000hz_wall_y_0': 0.5,
        '4000hz_wall_y_0': 0.5,
        '8000hz_wall_y_0': 0.8,
        '16000hz_wall_y_0': 0.7,
        '125hz_wall_y_1': 0.5,
        '250hz_wall_y_1': 0.5,
        '500hz_wall_y_1': 0.5,
        '1000hz_wall_y_1': 0.5,
        '2000hz_wall_y_1': 0.5,
        '4000hz_wall_y_1': 0.5,
        '8000hz_wall_y_1': 0.5,
        '16000hz_wall_y_1': 0.5,
        '125hz_wall_z_0': 0.5,
        '250hz_wall_z_0': 0.5,
        '500hz_wall_z_0': 0.5,
        '1000hz_wall_z_0': 0.5,
        '2000hz_wall_z_0': 0.5,
        '4000hz_wall_z_0': 0.3,
        '8000hz_wall_z_0': 0.6,
        '16000hz_wall_z_0': 0.6,
        '125hz_wall_z_1': 0.5,
        '250hz_wall_z_1': 0.3,
        '500hz_wall_z_1': 0.2,
        '1000hz_wall_z_1': 0.5,
        '2000hz_wall_z_1': 0.5,
        '4000hz_wall_z_1': 0.6,
        '8000hz_wall_z_1': 0.7,
        '16000hz_wall_z_1': 0.7
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
