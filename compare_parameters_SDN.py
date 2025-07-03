import os
import yaml
import math
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import numpy as np
from scripts.audio.reverb_features import compute_rev_features
from autorank import autorank, plot_stats, create_report, latex_table
import copy
from scipy.signal import butter, filtfilt
# from acoustics.room import clarity, t60_impulse
# from acoustics.bands import third
# from pyroomacoustics.experimental.rt60 import measure_rt60

# TO SET -----------------------------------------------------------------------------------------------------
rooms_to_compare = ['SDN050', 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058',
                    'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
# rooms_to_compare = ['SDN050', 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055']
# rooms_to_compare = ['SDN062']

add_nodimred_comparison = True
add_pca_classic_comparison = True
add_baseline_comparison = True

# base_folder = 'results_SDN_matching/matched_rirs_with_direct_newSDN_sweep'
base_folder = 'results_SDN_matching/matched_rirs_with_direct_newSDN_sweep_rerunmatching'
# base_folder = 'results_SDN_matching/before_review/matched_rirs_with_direct'

iter = '150'
omni_folder = 'omni_starting _point'
log20 = '_20log'

# '': Max-rE; '_hypercardioid': hyper-cardioid; '_cardioid': cardioid
polar_pattern = '_hypercardioid'

# step3 = '_3stepMatcht60+-02'
step3 = '_3stepMatcht60+-02_newSDNsweep'
# step3 = ''

# Whether to load the acoustic features of AURORA or to compute them
acoustic_features_source = 'code'#'aurora'#

filter_rirs = False
# ------------------------------------------------------------------------------------------------------------

n_rirs = len(rooms_to_compare)

actual_path = 'audio/input/chosen_rirs/stereo/'
tuned_params_path = 'audio/params/stereo/dim_red_' + iter + polar_pattern + log20 + step3 + '/' + omni_folder
tuned_audio_path = base_folder + '/dim_red_' + iter + polar_pattern + log20 + step3 + '/'
tuned_aurora_path = base_folder + '/dim_red_' + iter + polar_pattern + log20 + step3 + '/'
tuned_params_path_nodimred = 'audio/params/stereo/no_dim_red_' + iter + polar_pattern + log20 + step3 + '/' + omni_folder
tuned_audio_path_nodimred = base_folder + '/no_dim_red_' + iter + polar_pattern + log20 + step3 + '/'
tuned_aurora_path_nodimred = base_folder + '/no_dim_red_' + iter + polar_pattern + log20 + step3 + '/'
tuned_params_path_pcaclassic = 'audio/params/stereo/dim_red_pca_classic_' + iter + polar_pattern + log20 + step3 + '/' + omni_folder
tuned_audio_path_pcaclassic = base_folder + '/dim_red_pca_classic_' + iter + polar_pattern + log20 + step3 + '/'
tuned_aurora_path_pcaclassic = base_folder + '/dim_red_pca_classic_' + iter + polar_pattern + log20 + step3 + '/'

baseline_audio_path = base_folder + '/baseline'
baseline_aurora_path = base_folder + '/baseline'
target_aurora_path = base_folder + r'\target'

aurora_filename = 'AP_pos01 1.txt'

coef_bands = ['125hz_wall', '250hz_wall', '500hz_wall', '1000hz_wall', '2000hz_wall', '4000hz_wall', '8000hz_wall', '16000hz_wall']
coef_bands_eval = ['125hz_wall', '250hz_wall', '500hz_wall', '1000hz_wall', '2000hz_wall', '4000hz_wall']
coef_bands_eval_nowall = ['125hz', '250hz', '500hz', '1000hz', '2000hz', '4000hz']
wall_order = ['omni', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']
freqs = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz', '8000 Hz', '16000 Hz']
freqs_6bands = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz']
walls_lbl = ['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']

plot_each_param_err = True

n_walls = 6
n_bands = len(coef_bands)
n_bands_eval = len(coef_bands_eval)

dict_walls = {'x_0': 0,
              'x_1': 1,
              'y_0': 2,
              'y_1': 3,
              'z_0': 4,
              'z_1': 5
              }
dict_bands = {'125hz': 0,
              '250hz': 1,
              '500hz': 2,
              '1000hz': 3,
              '2000hz': 4,
              '4000hz': 5
              }

baseline_name = 'Baseline'
nodimred_name = 'OctBand'#'NoDimRed'
pcaclassic_name = 'PCA'
ourmethod_name = '2PS'

cond_names = []
if add_baseline_comparison:
    cond_names.append(baseline_name)
if add_nodimred_comparison:
    cond_names.append(nodimred_name)
if add_pca_classic_comparison:
    cond_names.append(pcaclassic_name)
cond_names.append(ourmethod_name)

colors = {baseline_name: '#fb8072',#'#f2e177',
          nodimred_name: '#fdb462',#'#bebada',#'#8dd3c7', #'#B72142',
          pcaclassic_name: '#8dd3c7',#'#fb8072',#'#F9A26F',
          ourmethod_name: '#80b1d3',#'#6076A4',
          'Target': '#7DCAB0'}
# colors = {baseline_name: '#CC79A7',#'#f2e177',
#           nodimred_name: '#E69F00',#'#bebada',#'#8dd3c7', #'#B72142',
#           pcaclassic_name: '#009E73',#'#fb8072',#'#F9A26F',
#           ourmethod_name: '#0072B2',#'#6076A4',
#           'Target': '#7DCAB0'}

font = 'Consolas'

features_aurora = ['Tuser', 'T30', 'T20', 'EDT', 'C80', 'Ts', 'strenGth']
t30_nan = 0
t20_nan = 0

unit = {}
unit['RT'] = 'ms'
unit['T20'] = 'ms'
unit['T30'] = 'ms'
unit['T60'] = 'ms'
unit['Tuser'] = 'ms'
unit['EDT'] = 'ms'
unit['Ts'] = 'ms'
unit['C80'] = 'dB'
unit['LF80'] = 'dB'
unit['G'] = 'dB'
unit['strenGth'] = 'dB'
unit['SC'] = 'Hz'

features_stat_analysis = ['T60', 'EDT', 'C80', 'Ts']

# bands_rt = third(25, 20000)

save_path = base_folder#'results_SDN_matching'

def high_pass_filter(signal, sample_rate, cutoff=20, order=4):
    """
    Applies a high-pass Butterworth filter to the input signal.

    Parameters:
    - signal: The input signal (1D NumPy array).
    - sample_rate: Sampling rate of the signal in Hz.
    - cutoff: Cutoff frequency of the high-pass filter in Hz (default 20 Hz).
    - order: Order of the Butterworth filter (default 4).

    Returns:
    - filtered_signal: The high-pass filtered signal.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def EDC(impulse_response, fs):
    fs = float(fs)  # make sure this is treated as a floating point number
    impulse_response = np.array(impulse_response)  # convert to np array if it isn't already

    # Schroeder's energy decay curve
    cumul = 10.0 * np.log10(np.sum(np.square(impulse_response)))
    decay_curve = 10.0 * np.log10(np.flipud(np.cumsum(np.flipud(np.square(impulse_response))))) - cumul
    return decay_curve


def RT(decay_curve, fs, end_level, start_level=-5):
    t_start = np.argmax(decay_curve < start_level) / fs  # time at which EDC drops below -5dB or start_level
    t_end = np.argmax(decay_curve < end_level) / fs  # time at which EDC drops below end_level
    # decay method
    RT_d = (t_end - t_start) * 60 / (end_level - start_level)
    # regression method
    s_start = np.ceil(t_start * fs).astype(int)  # convert start time to integer sample number
    s_end = np.ceil(t_end * fs).astype(int)  # convert end time to integer sample number
    p = np.polyfit(np.arange(s_start, s_end), decay_curve[np.arange(s_start, s_end)], 1)
    RT_r = -60 / p[0] / fs
    return (RT_d, RT_r, p)


def RT20(decay_curve, fs):
    return RT(decay_curve, fs, -25)


def RT30(decay_curve, fs):
    return RT(decay_curve, fs, -35)


def EDT(decay_curve, fs):
    return RT(decay_curve, fs, -10, 0)


def C50(impulse_response, fs, delay=0):
    return clarity(.050, impulse_response, fs, delay)


def C80(impulse_response, fs, delay=0):
    return clarity(.080, impulse_response, fs, delay)


def clarity(early_time, impulse_response, fs, delay=0):
    D = definition(early_time, impulse_response, fs, delay)
    return 10.0 * np.log10(D / (1 - D))


def D50(impulse_response, fs, delay=0):
    return definition(.050, impulse_response, fs, delay)

def definition(early_time, impulse_response, fs, delay=0):
    # disregard everything before delay
    impulse_response = impulse_response[np.round(delay * fs).astype(int):]
    early_energy = np.sum(np.square(impulse_response[:np.round(early_time * fs).astype(int)]))
    total_energy = np.sum(np.square(impulse_response))
    return early_energy / total_energy


def TS(impulse_response, fs, delay=0):
    # disregard everything before delay
    impulse_response = impulse_response[np.round(delay * fs).astype(int):]
    numerator = np.sum(np.arange(len(impulse_response)) / fs * np.square(impulse_response))  # ∑ t * p^2(t)
    total_energy = np.sum(np.square(impulse_response))  # ∑ p^2(t)
    return 1000.0 * numerator / total_energy  # expressed in ms


def compute_error(x1, x2, metric='abs'):
    if metric == 'abs':
        err = abs(x1 - x2)
    # elif metric == 'rmse':
    #     np.sqrt(np.mean((x1 - x2) ** 2))
    return err


def load_parameters(path):
    with open(path, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
            # tuned_params[pos] = dict_raw['parameters']

            # losses.append(dict_raw['loss_end_value'])
            # print(f'      - Loss: {losses[-1]:.2f} dB')
        except yaml.YAMLError as exc:
            print(exc)

    return parameters


def extract_rir_params_matrix(params_dict: dict):
    params_list = [str(k) for k in params_dict
                   if (str(k).split('_wall')[0] in coef_bands_eval_nowall) and (str(k).split('_wall_')[1] in walls_lbl)]

    params_matrix = np.zeros((n_walls, n_bands_eval))
    for p in params_list:
        p_split = p.split('_wall_')
        band_p = p_split[0]
        wall_p = p_split[1]

        params_matrix[dict_walls[wall_p], dict_bands[band_p]] = params_dict[p]

    return params_matrix


def plot_err_bands_conditions(cond_names, errors, show=True, title='', y_name='', filename='', width=460, height=260):

    coef_x = [b.split('hz')[0] for b in coef_bands_eval]

    fig = go.Figure()

    for c in cond_names:
        if len(errors[c].shape) == 3:

            y = np.mean(errors[c], axis=(0, 2))

            # error_mean_rir_band = np.mean(errors[c], axis=0)
            # y_25_perc = np.percentile(error_mean_rir_band, 25, axis=1)
            # y_75_perc = np.percentile(error_mean_rir_band, 75, axis=1)

            y_std = np.std(errors[c], axis=(0, 2))

            fig.add_trace(
                go.Scatter(x=coef_x, y=y, name=f'<i>{c}</i>', marker_color=colors[c], marker_size=7,
                           line=dict(width=2.5)
                           # error_y=dict(
                           #     type='data',
                           #     symmetric=False,
                           #     array=y_75_perc,
                           #     arrayminus=y_25_perc)
                           #  error_y = dict(
                           #  type='data',  # value of error bar given in data coordinates
                           #  array=y_std,
                           #  visible=True)
                           )
            )

        else:
            y = errors[c]

            fig.add_trace(
                go.Scatter(x=coef_x, y=y, name=c, marker_color=colors[c])
            )

    fig.update_layout(title=title,
                      xaxis_title="Octave band [Hz]",
                      yaxis_title=y_name,
                      width=width,
                      height=height,
                      font_family=font,
                      plot_bgcolor='white',
                      legend=dict(
                          yanchor="top",
                          y=1,
                          xanchor="right",
                          x=1,
                          borderwidth=0.5
                      ),
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0 if title == '' else 25, 'b': 0},
                      )
    fig.update_xaxes(
        #     mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey'
        #         ticktext=bands
    )
    fig.update_yaxes(
        #     mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zerolinecolor='lightgrey'
    )

    # fig.update_yaxes(range=[0.038, 0.345])#, gridcolor='#000000')
    fig.update_yaxes(range=[0.05, 0.345])#, gridcolor='#000000')
    # fig.update_xaxes(gridcolor='#000000')
    #
    # fig.update_layout(plot_bgcolor="#FFFFFF")

    save_filename = f'{save_path}/{filename}'

    if show:
        fig.show()

        fig.write_html(f'{save_filename}.html')

    fig.write_image(f'{save_filename}.pdf', format='pdf', engine='orca')


def plot_matrix_err(cond_names, errors, width=400, height=400, show=True):
    err_matrix = {}
    max_err = 0
    for cond in cond_names:

        err_matrix[cond] = np.mean(errors[cond], axis=2)

        err_cond_max = np.max(err_matrix[cond])

        if err_cond_max > max_err:
            max_err = err_cond_max

    for cond in cond_names:
        # fig = px.imshow(err_matrix, title=f"Absolute error - {type} - [{venv_name} - {r_n}]", zmin=0, zmax=1)
        fig = go.Figure(data=go.Heatmap(z=err_matrix[cond], zmin=0, zmax=max_err,
                                        x=[b.split(' ')[0] for b in freqs_6bands],
                                        y=[f'${w}$' for w in walls_lbl],
                                        # xgap=1,
                                        # ygap=1,
                                        colorscale='Viridis',
                                        # reversescale=True,
                                        # colorscale=['#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
                                        ))
        fig.update_layout(title=cond,
                          xaxis_title="Frequency band [Hz]", yaxis_title="Wall", xaxis={'side': 'top'},
                          width=width,
                          height=height)

        save_filename = f'{save_path}/err_matrix_{cond}'

        if show:
            fig.show()

            # fig.write_html(f'{save_filename}.html')

        fig.write_image(f'{save_filename}.pdf', format='pdf', engine='orca')

def statistical_analysis(cond_names, errors):
    data = pd.DataFrame()
    for cond in cond_names:
        data[cond] = errors[cond].flatten()

    result = autorank(data, alpha=0.05, verbose=False)

    pd.set_option('display.max_columns', 7)
    # print(result)

    create_report(result)

    # latex_table(result)


def statistical_analysis_acoustic_param(data):
    df = pd.DataFrame(data)

    result = autorank(df, alpha=0.05, verbose=False)
    pd.set_option('display.max_columns', 7)
    create_report(result)


def read_aurora_res(path):
    # bands = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    #          315, 400, 500, 630, 800, 1000, 1300, 1600, 2000, 2500,
    #          3200, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]

    results_path = os.path.join(path)
    res = pd.read_csv(results_path, sep='\t', skiprows=5)#, nrows=14)
    res.rename(columns={'Frq.band [Hz]': 'Feature', 'Filename': 'Channel'}, inplace=True)
    # t30_nan += (res[res['Feature'] == 'T30']['Lin'] == '--').sum()
    # print(f"T30 NaN: {(res[res['Feature'] == 'T30']['Lin'] == '--').sum()}")
    res = res[res['Feature'].isin(features_aurora)]
    res.set_index('Feature', inplace=True)
    # res.drop(labels=['Filename'], axis=1, inplace=True)

    rev_features = {}

    # t20_nan += (res.loc['T20']['Lin'] == '--').sum()
    # print(f"T20 NaN: {(res.loc['T20']['Lin'] == '--').sum()}")

    # for index, row in res.iterrows():
    for f in features_aurora:
        # rev_features[f] = res.loc[f]['Lin'].replace('--', np.nan).astype(float).tolist()
        rev_features[f] = res.loc[f]['A'].replace('--', np.nan).astype(float).tolist()

        # # Keep only omni channel
        # rev_features[f] = [rev_features[f][0]]

        if f in ['Tuser', 'T30', 'T20', 'EDT']:
            rev_features[f] = [r * 1000 for r in rev_features[f]]
    # for index, row in res.iterrows():
    #     if index in features_aurora:
    #         rev_features[index] = float(row['Lin'])
    #         if index == 'T20' or index == 'EDT':
    #             rev_features[index] = rev_features[index] * 1000

    return rev_features


def extract_rev_features(aurora_path, path, room, source='aurora'):
    if source == 'aurora':
        rev_features = read_aurora_res(os.path.join(aurora_path, f'AP_{room} 1.txt'))

    elif 'code':
        audio, sr = sf.read(path)
        audio = audio.T

        # audio = audio[:, :sr*3]

        if filter_rirs:
            audio = high_pass_filter(audio, sample_rate=sr, cutoff=20)

        rev_features = compute_rev_features(audio, sr)

    return rev_features


def compare_reverb_features(target, conditions):
    print('=====================> REVERBERATION FEATURES')

    rooms = list(conditions.keys())
    conditions_names = list(conditions[rooms[0]].keys())
    features_names = list(conditions[rooms[0]][conditions_names[0]].keys())

    diff_rev_feat = dict.fromkeys(features_names)
    for feature in features_names:
        diff_rev_feat[feature] = dict.fromkeys(conditions_names)
        for cond in conditions_names:
            diff_rev_feat[feature][cond] = []
    diff_rev_feat_perc = copy.deepcopy(diff_rev_feat)
    diff_rev_feat_perchannel = copy.deepcopy(diff_rev_feat)

    n_nan = dict.fromkeys(features_names, 0)

    for n, room in enumerate(conditions):
        print(f'\nRIR ({n + 1}/{n_rirs}): {room}')

        for cond in conditions[room]:

            print(f'   - {cond}:')

            for feature in conditions[room][cond]:
                n_nan[feature] += sum([math.isnan(x) for x in target[room][feature]])
                n_nan[feature] += sum([math.isnan(x) for x in conditions[room][cond][feature]])

                abs_diff = abs(np.array(target[room][feature]) - np.array(conditions[room][cond][feature]))
                diff = np.nanmean(abs_diff)
                diff_std = np.nanstd(abs_diff)

                abs_diff_perc = abs_diff / np.array(target[room][feature]) * 100
                abs_diff_perc[abs_diff_perc == float('inf')] = np.nan
                diff_perc = np.nanmean(abs_diff_perc)
                diff_std_perc = np.nanstd(abs_diff_perc)

                # diff = target[room][feature] - conditions[room][cond][feature]
                diff_rev_feat[feature][cond].append(abs(diff))
                diff_rev_feat_perchannel[feature][cond].extend(list(abs_diff))
                diff_rev_feat_perc[feature][cond].append(abs(diff_perc))

                print(f'       - {feature}: {diff:.1f} ± {diff_std:.1f} {unit[feature]} ({diff_perc:.2f} ± {diff_std_perc:.2f}%)')
                # print(f'       - {feature}: {diff:.3f} {unit[feature]} ({target[room][feature]:.3f} - {conditions[room][cond][feature]:.3f})')

    print(f'Number of NaN: {n_nan}')

    print(f'\nSTATISITICAL ANALYSIS FOR ACOUSTICS PARAMETERS')
    for feature in features_stat_analysis:
        print(f'{feature}')
        statistical_analysis_acoustic_param(diff_rev_feat_perchannel[feature])
        print()

    print(f'\nMEAN')

    for feature in features_names:
        print(f'   - {feature}:')

        for cond in conditions_names:
            print(f'       - {cond}: {np.nanmean(diff_rev_feat[feature][cond]):.1f} ± {np.nanstd(diff_rev_feat[feature][cond]):.1f} {unit[feature]} ({np.nanmean(diff_rev_feat_perc[feature][cond]):.1f} ± {np.nanstd(diff_rev_feat_perc[feature][cond]):.1f} %)')


if __name__ == "__main__":

    actual_params = np.zeros((n_walls, n_bands_eval, n_rirs))
    actual_rev = {}
    tuned_params = np.zeros((n_walls, n_bands_eval, n_rirs))
    conditions_rev = {}
    if add_baseline_comparison:
        baseline_params = np.zeros((n_walls, n_bands_eval, n_rirs))

    if add_nodimred_comparison:
        tuned_params_nodimred = np.zeros((n_walls, n_bands_eval, n_rirs))

    if add_pca_classic_comparison:
        tuned_params_pcaclassic = np.zeros((n_walls, n_bands_eval, n_rirs))

    print('=====================> ABSORPTION COEFFICIENTS')
    for n, room in enumerate(rooms_to_compare):
        print(f'RIR ({n+1}/{n_rirs}): {room}')

        conditions_rev[room] = {}

        actual_params_r = load_parameters(os.path.join(actual_path, room, 'parameters.yml'))

        positions = list(actual_params_r.keys())
        pos = positions[0]
        actual_params[:, :, n] = extract_rir_params_matrix(actual_params_r[pos])

        actual_rev[room] = extract_rev_features(target_aurora_path,
                                                # os.path.join(actual_path, room, '_todo', f'{pos}.wav'),
                                                os.path.join(target_aurora_path, f'{room}.wav'),
                                                room, source=acoustic_features_source)

        tuned_params_path_room = os.path.join(tuned_params_path, room)

        if not os.path.exists(tuned_params_path_room):
            raise Exception(f"Path {tuned_params_path_room} not found")

        tuned_params_r_raw = load_parameters(os.path.join(tuned_params_path_room, pos, 'SDN', 'SDN.yml'))
        tuned_params_r = tuned_params_r_raw['parameters']
        tuned_params[:, :, n] = extract_rir_params_matrix(tuned_params_r)

        if add_baseline_comparison:
            baseline_params_r = tuned_params_r_raw['baseline_parameters']
            baseline_params[:, :, n] = extract_rir_params_matrix(baseline_params_r)

            # baseline_audio_path_room = os.path.join(baseline_audio_path, room, f'{pos}_SDN.wav')
            baseline_audio_path_room = os.path.join(baseline_audio_path, f'{room}.wav')
            conditions_rev[room][baseline_name] = extract_rev_features(baseline_aurora_path, baseline_audio_path_room, room,
                                                source=acoustic_features_source)

            baseline_err_r = compute_error(actual_params[:, :, n], baseline_params[:, :, n])
            print(f'   - baseline: {np.mean(baseline_err_r):.3f} ± {np.std(baseline_err_r):.3f}')

        if add_nodimred_comparison:
            tuned_params_path_room_nodimred = os.path.join(tuned_params_path_nodimred, room)
            # tuned_audio_path_room_nodimred = os.path.join(tuned_audio_path_nodimred, room, 'SDN', f'{pos}_SDN.wav')
            tuned_audio_path_room_nodimred = os.path.join(tuned_audio_path_nodimred, f'{room}.wav')

            conditions_rev[room][nodimred_name] = extract_rev_features(tuned_aurora_path_nodimred, tuned_audio_path_room_nodimred, room,
                                                source=acoustic_features_source)

            tuned_params_nodimred_r_raw = load_parameters(os.path.join(tuned_params_path_room_nodimred, pos, 'SDN', 'SDN.yml'))
            tuned_params_nodimred_r = tuned_params_nodimred_r_raw['parameters']
            tuned_params_nodimred[:, :, n] = extract_rir_params_matrix(tuned_params_nodimred_r)

            pred_nodimred_err_r = compute_error(actual_params[:, :, n], tuned_params_nodimred[:, :, n])
            print(f'   - No dim red: {np.mean(pred_nodimred_err_r):.3f} ± {np.std(pred_nodimred_err_r):.3f}')

        if add_pca_classic_comparison:
            tuned_params_path_room_pcaclassic = os.path.join(tuned_params_path_pcaclassic, room)
            tuned_audio_path_room_pcaclassic = os.path.join(tuned_audio_path_pcaclassic, f'{room}.wav')

            conditions_rev[room][pcaclassic_name] = extract_rev_features(tuned_aurora_path_pcaclassic, tuned_audio_path_room_pcaclassic, room,
                                                source=acoustic_features_source)

            tuned_params_pcaclassic_r_raw = load_parameters(os.path.join(tuned_params_path_room_pcaclassic, pos, 'SDN', 'SDN.yml'))
            tuned_params_pcaclassic_r = tuned_params_pcaclassic_r_raw['parameters']
            tuned_params_pcaclassic[:, :, n] = extract_rir_params_matrix(tuned_params_pcaclassic_r)

            pred_pcaclassic_err_r = compute_error(actual_params[:, :, n], tuned_params_pcaclassic[:, :, n])
            print(f'   - PCA classic: {np.mean(pred_pcaclassic_err_r):.3f} ± {np.std(pred_pcaclassic_err_r):.3f}')

        tuned_audio_path_room = os.path.join(tuned_audio_path, f'{room}.wav')
        conditions_rev[room][ourmethod_name] = extract_rev_features(tuned_aurora_path, tuned_audio_path_room, room,
                                                source=acoustic_features_source)

        pred_err_r = compute_error(actual_params[:, :, n], tuned_params[:, :, n])
        print(f'   - our method: {np.mean(pred_err_r):.3f} ± {np.std(pred_err_r):.3f}')

    actual_params_nozero = actual_params.copy()
    actual_params_nozero[actual_params_nozero==0] = 0.001
    if add_baseline_comparison:
        baseline_err = compute_error(actual_params, baseline_params)
        baseline_err_perc = baseline_err / actual_params_nozero * 100

        print(f'Mean absolute error (baseline): {np.mean(baseline_err):.3f} ({np.nanmean(baseline_err_perc):.1f}%) ± {np.std(baseline_err):.3f}')

    if add_nodimred_comparison:
        pred_nodimred_err = compute_error(actual_params, tuned_params_nodimred)
        pred_nodimred_err_perc = pred_nodimred_err / actual_params_nozero * 100
        print(f'Mean absolute error (No dim red): {np.mean(pred_nodimred_err):.3f} ({np.nanmean(pred_nodimred_err_perc):.1f}%) ± {np.std(pred_nodimred_err):.3f}')

    if add_pca_classic_comparison:
        pred_pcaclassic_err = compute_error(actual_params, tuned_params_pcaclassic)
        pred_pcaclassic_err_perc = pred_pcaclassic_err / actual_params_nozero * 100
        print(f'Mean absolute error (PCA classic): {np.mean(pred_pcaclassic_err):.3f} ({np.nanmean(pred_nodimred_err_perc):.1f}%) ± {np.std(pred_pcaclassic_err):.3f}')

    pred_err = compute_error(actual_params, tuned_params)
    pred_err_perc = pred_err / actual_params_nozero * 100

    print(f'Mean absolute error (our method): {np.mean(pred_err):.3f} ({np.nanmean(pred_err_perc):.1f}%) ± {np.std(pred_err):.3f}')

    errors = {}
    errors_perc = {}

    if add_baseline_comparison:
        errors[baseline_name] = baseline_err
        errors_perc[baseline_name] = baseline_err_perc
    if add_nodimred_comparison:
        errors[nodimred_name] = pred_nodimred_err
        errors_perc[nodimred_name] = pred_nodimred_err_perc
    if add_pca_classic_comparison:
        errors[pcaclassic_name] = pred_pcaclassic_err
        errors_perc[pcaclassic_name] = pred_pcaclassic_err_perc

    errors[ourmethod_name] = pred_err
    errors_perc[ourmethod_name] = pred_err_perc

    plot_err_bands_conditions(cond_names, errors, show=True, title='', y_name='Mean absolute error', filename='abserr_per_band')

    plot_matrix_err(cond_names, errors)

    try:
        statistical_analysis(cond_names, errors)
        # statistical_analysis(cond_names, errors_perc)
    except:
        Warning('Not enough conditions for statistical analysis')

    # EXAMPLES
    # cond_names_trg = ['Target'] + cond_names
    # for r, room in enumerate(rooms_to_compare):
    #     for w, wall in enumerate(walls_lbl):
    #         coeff = {}
    #         coeff['Target'] = actual_params[w, :, r]
    #         if add_baseline_comparison:
    #             coeff[baseline_name] = baseline_params[w, :, r]
    #         if add_nodimred_comparison:
    #             coeff[nodimred_name] = tuned_params_nodimred[w, :, r]
    #         if add_pca_classic_comparison:
    #             coeff[pcaclassic_name] = tuned_params_pcaclassic[w, :, r]
    #
    #         coeff[ourmethod_name] = tuned_params[w, :, r]
    #
    #         plot_err_bands_conditions(cond_names_trg, coeff, show=False, title=f'{room} - {wall}',
    #                                   y_name='Absorption coefficient', filename=f'ex_{room}_{wall}')

    # REVERB FEATURES
    compare_reverb_features(actual_rev, conditions_rev)

    pass
