import os

import numpy as np
import pandas as pd

from scripts.parameters_learning import *
from scripts.utils.json_functions import *
from scripts.utils.plot_functions import *

from scripts.audio.audio_manipulation import *
from scripts.audio.signal_generation import *
from scripts.audio.reverb_features import *
from scripts.audio.DSPfunc import *

rir_path = '../audio/input/chosen_rirs/'
rir_file = os.listdir(rir_path)
rir_folder = os.listdir('../audio/results')

rir, sr = sf.read(rir_path + rir_file[0])
rir = rir.T

test_sound = create_impulse(sr * 6)
test_sound = np.stack([test_sound, test_sound])

#rir_eq_coeffs = np.load('audio/armodels/rir_eq_coeffs_ms.npy', allow_pickle=True)[()]
rir_eq_coeffs = np.load('../audio/armodels/rir_eq_coeffs_kl_s.npy', allow_pickle=True)[()]

rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
rev_param_names_ex.pop('fdn_size_internal')
rev_param_ranges = rev_param_ranges_ex[:-1]

df = pd.DataFrame(
    columns=['Ref_RT', 'Ref_EDT', 'Ref_TS', 'Ref_G', 'Ref_C', 'Ref_LS', 'Ref_CS',
             'Fv_RT',  'Fv_EDT',  'Fv_TS',  'Fv_G',  'Fv_C',  'Fv_LS',  'Fv_CS',
             'Fdn_RT', 'Fdn_EDT', 'Fdn_TS', 'Fdn_G', 'Fdn_C', 'Fdn_LS', 'Fdn_CS'],
    index=rir_folder)

num_features = int(len(df.columns) / 3)

convolution = []
reference_audio = []
reference_norm = []

mix = 1.0

rirs = os.listdir(rir_path)

for idx, rir_file in enumerate(rirs):
    convolution.append(pedalboard.Convolution(rir_path + rir_file, mix))
    reference_audio.append(convolution[idx](test_sound, sr))
    reference_norm.append(normalize_audio(reference_audio[idx]))

for ref_idx, ref in enumerate(reference_norm):

    ref_rt, ref_edt, ref_g, ref_c, ref_lf, ref_ts, ref_sc = get_rev_features(ref, sr)

    sf.write('audio/revparams_rirs/' + rir_folder[ref_idx] + '/reference.wav', ref.T, sr)

    rir_eq = list(rir_eq_coeffs.values())[ref_idx]

    test_mid = filters([1], rir_eq[0], test_sound[0])
    test_eq = np.stack([test_mid, test_mid])

    params_path = '../audio/params/'
    params_folder = os.listdir(params_path)

    current_param_path = params_path + params_folder[ref_idx] + '/'

    for effect_params in os.listdir(current_param_path):

        model_path = current_param_path + effect_params + '/'

        for model in os.listdir(model_path):

            params = json_load(model_path + model)

            if effect_params == 'fdn':
                #print('fdn ---', params)
                external_vst3_set_params(params, rev_external)

                reverb_audio_external = plugin_process(rev_external, test_eq, sr)
                reverb_audio_external = pd_highpass_filter(reverb_audio_external, 3, sr)

                rev_ms = ms_matrix(reverb_audio_external)
                rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
                reverb_audio_external = ms_matrix(rev_ms)

                reverb_norm_external = normalize_audio(reverb_audio_external)

                sf.write('audio/revparams_rirs/' + rir_folder[ref_idx] + '/' + effect_params + '.wav',
                         reverb_norm_external.T, sr)

                fdn_rt, fdn_edt, fdn_g, fdn_c, fdn_lf, fdn_ts, fdn_sc = get_rev_features(reverb_norm_external, sr)


            elif effect_params == 'fv':
                #print('fv ---', params)
                opt_rev_native = native_reverb_set_params(params)

                reverb_audio_native = plugin_process(opt_rev_native, test_eq, sr)
                reverb_audio_native = pd_highpass_filter(reverb_audio_native, 3, sr)

                rev_ms = ms_matrix(reverb_audio_native)
                rev_ms[1] = filters([1], rir_eq[1], rev_ms[1])
                reverb_audio_native = ms_matrix(rev_ms)

                reverb_norm_native = normalize_audio(reverb_audio_native)

                sf.write('audio/revparams_rirs/' + rir_folder[ref_idx] + '/' + effect_params + '.wav',
                         reverb_norm_native.T, sr)

                fv_rt, fv_edt, fv_g, fv_c, fv_lf, fv_ts, fv_sc = get_rev_features(reverb_norm_native, sr)

    results = [ref_rt, ref_edt, ref_ts, ref_g, ref_c, ref_lf, ref_sc,
               fv_rt, fv_edt, fv_ts, fv_g, fv_c, fv_lf, fv_sc,
               fdn_rt, fdn_edt, fdn_ts, fdn_g, fdn_c, fdn_lf, fdn_sc]

    row = [round(r, 1) for r in results]
    df.iloc[ref_idx] = row

index = ['Auditorium - Atherton Hall (M)', 'Outdoors - Knts Halls Courtyard',
         'Recording Studio - The Masterfonics', 'Small Room - JamSync Florida Room',
         'Simulation_A1_01_XY', 'REVelation']

df = df.reindex(index)

df = df.rename(index={'Auditorium - Atherton Hall (M)': 'Auditorium Hall',
                 'Outdoors - Knts Halls Courtyard': 'Outdoors Courtyard',
                 'Recording Studio - The Masterfonics': 'Recording Studio',
                 'Small Room - JamSync Florida Room' : 'Small Room',
                 'Simulation_A1_01_XY': 'Living Room',
                 'REVelation': 'Plate Reverb'})

text_file = open("../images/rev_feats_klm.txt", "w")

fv_mape = np.zeros(num_features)
fdn_mape = np.zeros(num_features)

for i in range(0, len(rir_folder)):

    ref_v = []
    fv_v = []
    fdn_v = []

    multirow = '\multirow{3}{*}{' + df.iloc[i].name + '} & '

    ref = "Reference & "
    for j in df.iloc[i][0:num_features]:
        ref_v.append(j)
        ref = ref + str(j) + " & "

    ref = ref[:-3] + ' \\\\ '

    fv = "& Freeverb & "
    for j in df.iloc[i][num_features:(2 * num_features)]:
        fv_v.append(j)
        fv = fv + str(j) + " & "

    fv = fv[:-3] + ' \\\\ '

    fdn = "& FDN & "
    for j in df.iloc[i][(2 * num_features):(3 * num_features)]:
        fdn_v.append(j)
        fdn = fdn + str(j) + " & "

    fdn = fdn[:-3] + ' \\\\ ' + '\midrule '

    rf_string = multirow + ref + fv + fdn
    print(rf_string)

    fv_diff = [round(v, 1) for v in (list(np.array(ref_v) - np.array(fv_v)) / np.array(ref_v))]
    fv_mape += np.abs(fv_diff)

    fdn_diff = [round(v, 1) for v in (list(np.array(ref_v) - np.array(fdn_v)) / np.array(ref_v))]
    fdn_mape += np.abs(fdn_diff)

    text_file.write(rf_string + '\n')

fv_e = (fv_mape / len(rir_folder)) * 100
fdn_e = (fdn_mape / len(rir_folder)) * 100

multirow_mape = '\multirow{2}{*}{MAPE (\%)} & '

fv_m = "Freeverb & "

for i in fv_e:
    fv_m = fv_m + str(round(i, 1)) + " & "

fv_m = fv_m[:-3] + ' \\\\ '

fdn_m = "& FDN & "

for i in fdn_e:
    fdn_m = fdn_m + str(round(i, 1)) + " & "

fdn_m = fdn_m[:-3] + ' \\\\ ' + '\midrule '

mape_string = multirow_mape + fv_m + fdn_m

text_file.write(mape_string + '\n')

text_file.close()


df.to_csv('images/reverb_features_klm.csv')




