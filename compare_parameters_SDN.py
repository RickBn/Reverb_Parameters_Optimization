import os
import yaml
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
from scripts.audio.reverb_features import get_rev_features

actual_path = 'audio/input/chosen_rirs/stereo/'
tuned_params_path = 'audio/params/stereo/'
tuned_audio_path = 'audio/vst_rirs/stereo/'

freqs = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz', '8000 Hz', '16000 Hz']
walls_lbl = ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']

plot_each_param_err = True

def abs_err(x1, x2):
    return abs(x1 - x2)


if __name__ == "__main__":

    rooms_to_compare = ['SDN028']

    print('LOSS EVALUATION')

    for folder in os.listdir(actual_path):
        if folder.startswith("SDN") and os.path.isdir(os.path.join(actual_path, folder)):

            if rooms_to_compare is [] or (rooms_to_compare is not [] and folder in rooms_to_compare):

                with open(os.path.join(actual_path, folder, 'parameters.yml'), "r") as stream:
                    try:
                        actual_params = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)

                positions = list(actual_params.keys())

                tuned_params_path_room = os.path.join(tuned_params_path, folder)
                tuned_audio_path_room = os.path.join(tuned_audio_path, folder, 'SDN')

                if not os.path.exists(tuned_params_path_room):
                    continue

                print(f'RIRs {folder}')

                params_to_compare = [str(k) for k in actual_params[positions[0]] if 'hz_wal' in str(k)]

                if 'mae_df' not in locals():
                    index = pd.MultiIndex.from_tuples([(folder, p) for p in positions], names=["env", "pos"])
                    mae_df = pd.DataFrame([], columns=params_to_compare, index=index)

                tuned_params = dict()
                losses = []

                for pos in positions:
                    print(f'   -> {pos}')

                    # PARAMETERS
                    with open(os.path.join(tuned_params_path_room, pos, 'SDN', 'SDN.yml'), "r") as stream:
                        try:
                             dict_raw = yaml.safe_load(stream)
                             tuned_params[pos] = dict_raw['parameters']

                             losses.append(dict_raw['loss_end_value'])
                             print(f'      - Loss: {losses[-1]:.2f} dB')
                        except yaml.YAMLError as exc:
                            print(exc)

                    for par in params_to_compare:
                        mae_df.loc[(folder, pos), par] = abs_err(actual_params[pos][par], tuned_params[pos][par])

                    if plot_each_param_err:
                        val_wall = dict()
                        for v in walls_lbl:
                            val_wall[v] = [actual_params[pos][f"{f.replace(' ', '').lower()}_wall_{v}"] -
                                           tuned_params[pos][f"{f.replace(' ', '').lower()}_wall_{v}"]
                                           for f in freqs]

                        fig = go.Figure(data=[go.Bar(name=w, x=freqs, y=val_wall[w]) for w in walls_lbl])
                        # Change the bar mode
                        fig.update_layout(title=f'Error for each frequency and wall for position {pos} of {folder}', barmode='group')
                        fig.show()

                    # AUDIO
                    actual_audio, sr = sf.read(os.path.join(actual_path, folder, '_todo', f'{pos}.wav'))
                    actual_audio = actual_audio.T
                    act_tr, act_edt, act_g, act_c, act_lf, act_ts, act_sc = get_rev_features(actual_audio, sr)

                    tuned_audio, sr = sf.read(os.path.join(tuned_audio_path_room, f'{pos}_SDN.wav'))
                    tuned_audio = tuned_audio.T
                    tun_tr, tun_edt, tun_g, tun_c, tun_lf, tun_ts, tun_sc = get_rev_features(tuned_audio, sr)

                    print(f'      - RT: {act_tr - tun_tr:.2f} ms ({act_tr:.2f} - {tun_tr:.2f})')
                    print(f'      - EDT: {act_edt - tun_edt:.2f} ms ({act_edt:.2f} - {tun_edt:.2f})')
                    print(f'      - Ts: {act_ts - tun_ts:.2f} ms ({act_ts:.2f} - {tun_ts:.2f})')
                    print(f'      - C80: {act_c - tun_c:.2f} dB ({act_c:.2f} - {tun_c:.2f})')
                    print(f'      - LF80: {act_lf - tun_lf:.2f} dB ({act_lf:.2f} - {tun_lf:.2f})')
                    print(f'      - G: {act_g - tun_g:.2f} dB ({act_g:.2f} - {tun_g:.2f})')
                    print(f'      - SC: {act_sc - tun_sc:.2f} Hz ({act_sc:.2f} - {tun_sc:.2f})')

    print('PARAMETERS EVALUATION')

    print('Mean MAE parameters per room/position:')
    print(mae_df.mean(axis=1))

    print('Mean MAE per coefficient:')
    mae_params_mean = mae_df.mean(axis=0)
    print(mae_params_mean)

    print('Mean MAE overall:')
    print(f'{mae_df.values.mean()} ± {mae_df.values.std()}')

    # Plots
    val_wall = dict()
    for v in walls_lbl:
        val_wall[v] = [mae_params_mean[f"{f.replace(' ', '').lower()}_wall_{v}"] for f in freqs]

    fig = go.Figure(data=[go.Bar(name=w, x=freqs, y=val_wall[w]) for w in walls_lbl])
    # Change the bar mode
    fig.update_layout(title='Mean absolute error (MAE) for each frequency and wall', barmode='group')
    fig.show()

