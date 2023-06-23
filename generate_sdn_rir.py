from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
import pedalboard
import soundfile as sf
import os

sr = 48000
rir_len_sec = 3
scale = 1
vst_path = "vst3/Real time SDN.vst3"
base_save_path = "audio/input/chosen_rirs/stereo/"

params = {
    'pos01': {
        'source_x': 0.5,
        'source_y': 0.2,
        'source_z': 0.5,
        'listener_x': 0.5,
        'listener_y': 0.2,
        'listener_z': 0.2,
        'listener_pitch': 0.0,
        'listener_yaw': 0.0,
        'listener_roll': 0.0,
        'dimensions_x': 10.0,
        'dimensions_y': 10.0,
        'dimensions_z': 10.0,
        '125hz_wall_x_0': 0.5,
        '250hz_wall_x_0': 0.5,
        '500hz_wall_x_0': 0.5,
        '1000hz_wall_x_0': 0.5,
        '2000hz_wall_x_0': 0.5,
        '4000hz_wall_x_0': 0.5,
        '8000hz_wall_x_0': 0.5,
        '16000hz_wall_x_0': 0.5,
        '125hz_wall_x_1': 0.5,
        '250hz_wall_x_1': 0.5,
        '500hz_wall_x_1': 0.5,
        '1000hz_wall_x_1': 0.5,
        '2000hz_wall_x_1': 0.5,
        '4000hz_wall_x_1': 0.5,
        '8000hz_wall_x_1': 0.5,
        '16000hz_wall_x_1': 0.5,
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
    }
}

if __name__ == "__main__":
    # Create impulse
    impulse = create_impulse(sr * rir_len_sec, n_channels=2)

    # Load SDN VST plugin
    rev_plugin = pedalboard.load_plugin(vst_path)

    # Find the last SDN folder
    folder_counter = 0
    for folder in os.listdir(base_save_path):
        if folder.startswith("SDN"):
            folder_n = int(folder.split('SDN')[1])
            if folder_n >= folder_counter:
                folder_counter = int(folder.split('SDN')[1]) + 1

    save_folder = f'{base_save_path}/SDN{folder_counter:03d}'
    os.makedirs(save_folder)

    # Iterate over the positions
    for pos_key, pos_param in params.items():

        # Generate RIR
        sdn_ir = vst_reverb_process(params, impulse, sr, scale_factor=scale, rev_external=rev_plugin)

        # Save RIR
        sf.write(f'{save_folder}/{pos_key}.wav', sdn_ir.T, sr)
