import os
import yaml
import pandas as pd
import plotly.graph_objects as go

actual_params_path = 'audio/input/chosen_rirs/stereo/'
tuned_params_path = 'audio/params/stereo/'


def abs_err(x1, x2):
    return abs(x1 - x2)


if __name__ == "__main__":

    for folder in os.listdir(actual_params_path):
        if folder.startswith("SDN") and os.path.isdir(os.path.join(actual_params_path, folder)):

            with open(os.path.join(actual_params_path, folder, 'parameters.yml'), "r") as stream:
                try:
                    actual_params = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            positions = list(actual_params.keys())

            tuned_params_path_room = os.path.join(tuned_params_path, folder, positions[0], 'SDN', 'SDN.yml')

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
                with open(tuned_params_path_room, "r") as stream:
                    try:
                         dict_raw = yaml.safe_load(stream)
                         tuned_params[pos] = dict_raw['parameters']

                         losses.append(dict_raw['loss_end_value'])
                         print(f'   - Loss {pos}: {losses[-1]}')
                    except yaml.YAMLError as exc:
                        print(exc)

                for par in params_to_compare:
                    mae_df.loc[(folder, pos), par] = abs_err(actual_params[pos][par], tuned_params[pos][par])

    print('Mean per room/position:')
    print(mae_df.mean(axis=1))

    print('Mean per coefficient:')
    mae_params_mean = mae_df.mean(axis=0)
    print(mae_params_mean)

    print('Mean overall:')
    print(f'{mae_df.values.mean()} ± {mae_df.values.std()}')

    # Plots
    freqs = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz', '8000 Hz', '16000 Hz']

    val_wall = dict()
    walls_lbl = ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']
    for v in walls_lbl:
        val_wall[v] = [mae_params_mean[f"{f.replace(' ', '').lower()}_wall_{v}"] for f in freqs]

    fig = go.Figure(data=[go.Bar(name=w, x=freqs, y=val_wall[w]) for w in walls_lbl])
    # Change the bar mode
    fig.update_layout(title='Mean absolute error (MAE) for each frequency and wall', barmode='group')
    fig.show()

