import os

import numpy as np
import pandas as pd

from scripts.parameters_learning import *
from scripts.utils.json_functions import *
from scripts.utils.plot_functions import *

from scripts.audio_functions.DSPfunc import *
from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.signal_generation import *
from scripts.audio_functions.pedalboard_process import *
from scripts.audio_functions.rir_functions import *


def vst_reverb_process(params, input, sr, scale_factor=1.0, hp_cutoff=None, rev_external=None):
    if rev_external is not None:
        rev_ex = external_vst3_set_params(params, rev_external)
        reverb_norm = process_external_reverb(rev_ex, sr, input, hp_cutoff=hp_cutoff, norm=True)

    else:
        rev_native = native_reverb_set_params(params)
        reverb_norm = process_native_reverb(rev_native, sr, input, hp_cutoff=hp_cutoff, norm=True)

    reverb_norm *= scale_factor

    return reverb_norm


def merge_er_tail_rir(er, tail, sr, fade_length=128, trim=None):

    fade_length = len(er.T)

    padded_er_rir = pad_signal(er, len(er), len(tail.T) - fade_length)

    fade_in_tail = tail * cosine_fade(len(tail.T), fade_length, False)

    start_point = len(er.T) - fade_length
    padded_er_rir[:, start_point:] += fade_in_tail

    if trim is not None:
        padded_er_rir = padded_er_rir[:, :(trim * sr)]
        padded_er_rir *= cosine_fade(len(padded_er_rir.T), fade_length)

    return padded_er_rir


def batch_generate_vst_rir(params_path, input_audio, sr, max_dict, rev_name='fv',
                           hp_cutoff=None, rev_external=None, save_path=None):
    if rev_external is not None and rev_name == 'fv':
        raise Exception("Attention! Reverb name is the default native reverb one but you loaded an external reverb!")

    effect_params = rev_name

    for rir_idx, rir in enumerate(os.listdir(params_path)):

        current_param_path = params_path + rir + '/'
        model_path = current_param_path + effect_params + '/'

        dp_scale_factor = 1.0 #max_dict[rir + '.wav']

        # scaled_input = input_audio * dp_scale_factor

        for model in os.listdir(model_path):

            params = model_load(model_path + model)

            reverb_norm = vst_reverb_process(params, input_audio, sr, scale_factor=dp_scale_factor,
                                             hp_cutoff=hp_cutoff, rev_external=rev_external)

            # if rev_external is not None:
            #     reverb_norm = process_external_reverb(params, rev_external, sr, input_audio, hp_cutoff=20, norm=True)
            #
            # else:
            #     reverb_norm = process_native_reverb(params, sr, input_audio, hp_cutoff=20, norm=True)
            #
            # reverb_norm *= dp_scale_factor

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(reverb_norm[0])

            if save_path is not None:
                sf.write(save_path + rir + '/' + rir + '_' + effect_params + '.wav', reverb_norm.T, sr)


def batch_merge_er_tail_rir(er_path, tail_path, fade_length=128, trim=None, save_path=None):
    er_files = os.listdir(er_path)
    tail_files = os.listdir(tail_path)

    for idx, rir in enumerate(tail_files):

        effect_path = tail_path + rir + '/'

        for effect_rir in os.listdir(effect_path):

            er_rir, er_sr = sf.read(er_path + rir + '.wav')
            er_rir = er_rir.T

            tail_rir, tail_sr = sf.read(effect_path + effect_rir)
            tail_rir = tail_rir.T

            if er_sr != tail_sr:
                raise Exception("Warning! ER and tail sampling rate doesn't match!")

            merged_rir = merge_er_tail_rir(er_rir, tail_rir, er_sr, fade_length, trim)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(merged_rir[0])

            if save_path is not None:
                #+ er_files[idx].replace(".wav", "/")
                sf.write(save_path + effect_rir, merged_rir.T, er_sr)


if __name__ == "__main__":
    rir_path = 'audio/trimmed_rirs/'
    rir_files = os.listdir(rir_path)

    sr = 44100

    impulse = create_impulse(sr * 6, stereo=True)

    rev_param_ranges_nat = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    rev_param_names_nat = {'room_size': 0.0, 'damping': 0.0, 'wet_level': 0.0, 'dry_level': 0.0, 'width': 0.0}

    rev_external = pedalboard.load_plugin("vst3/FdnReverb.vst3")
    rev_param_names_ex, rev_param_ranges_ex = retrieve_external_vst3_params(rev_external)
    rev_param_names_ex.pop('fdn_size_internal')
    rev_param_ranges = rev_param_ranges_ex[:-1]

    params_path = 'audio/params/'
    params_folder = os.listdir(params_path)

    vst_rir_save_path = 'audio/vst_rirs/'

    rir_max = rir_maximum(rir_path)

    batch_generate_vst_rir(params_path, impulse, sr, rir_max, 'fv',
                           hp_cutoff=20, rev_external=None, save_path=vst_rir_save_path)

    batch_generate_vst_rir(params_path, impulse, sr, rir_max, 'fdn',
                           hp_cutoff=20, rev_external=rev_external, save_path=vst_rir_save_path)

    er_path = 'audio/trimmed_rirs/'
    tail_path = 'audio/vst_rirs/'
    merged_rirs_path = 'audio/merged_rirs/'

    fade_in = int(5 * sr * 0.001)

    batch_merge_er_tail_rir(er_path, tail_path, fade_length=fade_in, trim=3, save_path=merged_rirs_path)

    input_sound_path = 'audio/input/sounds/'
    batch_input_sound = prepare_batch_input_stereo(input_sound_path)
    batch_convolution = prepare_batch_convolve(merged_rirs_path)

    merged_final_path = 'audio/merged_final/'
    input_file_names = os.listdir(input_sound_path)

    batch_convolve(batch_input_sound, batch_convolution, input_file_names, merged_rirs_path, 44100, scale_factor=0.70,
                   save_path=merged_final_path)

    path1 = 'audio/input/chosen_rirs/'
    path2 = 'audio/merged_rirs/'

    for rir in os.listdir(path1):
        merged_path = path2 + rir.replace('.wav', "") + '/'
        for merged in os.listdir(merged_path):
            plot_rir_pair(path1 + rir, merged_path + merged, 'images/rirs/generated/' + merged.replace('.wav', '.pdf'))

