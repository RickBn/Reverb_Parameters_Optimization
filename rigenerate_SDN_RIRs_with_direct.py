from scripts.parameters_learning import *
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
from pedalboard_change_channel_limit import pedalboard
from scripts.audio.rir_functions import get_ir_deconv_sweep
import soundfile as sf
import os
import yaml

sr = 48000
rir_len_sec = 3
scale = 1

rooms_to_compare = ['SDN050', 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058',
                    'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
# rooms_to_compare = ['SDN062']

# config_label = '150_hypercardioid_20log'
# config_label = '150_hypercardioid_20log_3stepMatcht60+-02'
config_label = '150_hypercardioid_20log_3stepMatcht60+-02_newSDNsweep'

vst_path = "vst3/Real time SDN_25ch.vst3"
folders = [
    rf'.\audio\params\stereo\dim_red_{config_label}\omni_starting _point',
           rf'.\audio\params\stereo\dim_red_pca_classic_{config_label}\omni_starting _point',
           rf'.\audio\params\stereo\no_dim_red_{config_label}\omni_starting _point',
           r'.\audio\input\chosen_rirs\stereo'
]
param_path = [
    '\pos01\SDN\SDN.yml', '\pos01\SDN\SDN.yml',
    '\pos01\SDN\SDN.yml',
    '/parameters.yml'
]
param_key = [
    'parameters', 'parameters',
    'parameters',
    'pos01'
]

save_folders = [
    f'results_SDN_matching/matched_rirs_with_direct/dim_red_{config_label}',
                f'results_SDN_matching/matched_rirs_with_direct/dim_red_pca_classic_{config_label}',
                f'results_SDN_matching/matched_rirs_with_direct/no_dim_red_{config_label}',
                'results_SDN_matching/matched_rirs_with_direct/target'
]
# save_folders = ['results_SDN_matching/matched_rirs_with_direct/target']

save_folder_baseline = 'results_SDN_matching/matched_rirs_with_direct/baseline'

ir_method = 'sweep'#'impulse'
pre_silence_impulse_sec = 0.03

if __name__ == "__main__":
    # Load SDN VST plugin
    rev_plugin = pedalboard.load_plugin(vst_path)

    for n, folder in enumerate(folders):
        for folder_rir in os.listdir(folder):
            if folder_rir not in rooms_to_compare:
                continue
            print(f'{folder_rir}')
            params_path = folder + r'/' + folder_rir + param_path[n]

            with open(params_path, "r") as stream:
                try:
                    parameters = yaml.safe_load(stream)
                    # tuned_params[pos] = dict_raw['parameters']

                    # losses.append(dict_raw['loss_end_value'])
                    # print(f'      - Loss: {losses[-1]:.2f} dB')
                except yaml.YAMLError as exc:
                    print(exc)

            if parameters[param_key[n]]['output_mode'] == 'Mono':
                n_ch = 1
            elif parameters[param_key[n]]['output_mode'].endswith('order Ambisonic'):
                n_ch = pow(int(parameters[param_key[n]]['output_mode'][0]) + 1, 2)
            else:
                n_ch = 2

            parameters[param_key[n]]['render_line_of_sight'] = True

            if ir_method == 'impulse':
                impulse = create_impulse(sr * rir_len_sec, n_channels=n_ch, amp=1,
                                         add_pre_silence_samples=int(sr * pre_silence_impulse_sec))

                sdn_ir = vst_reverb_process(parameters[param_key[n]], impulse, sr, scale_factor=scale, rev_external=rev_plugin,
                                            norm=False)
                sdn_ir = sdn_ir[:, int(sr * pre_silence_impulse_sec):]

            elif ir_method == 'sweep':
                sdn_ir = get_ir_deconv_sweep(rev_plugin, parameters[param_key[n]], sr, n_ch, max_len_sec=3)

            # if np.any(sdn_ir >= 1):
            #     warnings.warn('Amplitude >= 1 found!')
            # max_amp = np.max(sdn_ir)
            # sdn_ir = sdn_ir / max_amp

            # Save RIR
            os.makedirs(save_folders[n], exist_ok=True)
            sf.write(f'{save_folders[n]}/{folder_rir}.wav', sdn_ir.T, sr, subtype='FLOAT')

            if n == 0 and folder == rf'.\audio\params\stereo\dim_red_{config_label}\omni_starting _point':
                parameters['baseline_parameters']['render_line_of_sight'] = True

                if ir_method == 'impulse':
                    sdn_ir = vst_reverb_process(parameters['baseline_parameters'], impulse, sr, scale_factor=scale,
                                                rev_external=rev_plugin,
                                                norm=False)

                elif ir_method == 'sweep':
                    sdn_ir = get_ir_deconv_sweep(rev_plugin, parameters['baseline_parameters'], sr, n_ch, max_len_sec=3)

                os.makedirs(save_folder_baseline, exist_ok=True)
                # Save RIR
                sf.write(f'{save_folder_baseline}/{folder_rir}.wav', sdn_ir.T, sr, subtype='FLOAT')