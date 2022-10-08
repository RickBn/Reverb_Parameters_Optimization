from scripts.parameters_learning import *
from scripts.utils.plot_functions import *

from scripts.audio_functions.signal_generation import *
from scripts.audio_functions.pedalboard_functions import *
from scripts.audio_functions.rir_functions import *


def process_reverb(rev, sr, input_audio, scale_factor: float = 1.0,
                   hp_cutoff: float = None, norm: bool = True):
    reverb = plugin_process(rev, input_audio, sr)

    if hp_cutoff is not None:
        reverb = pd_highpass_filter(reverb, 3, sr, hp_cutoff)

    if norm:
        if reverb.ndim == 1:
            reverb = normalize_audio(reverb, nan_check=True, by_row=False)
        else:
            reverb = normalize_audio(reverb, nan_check=True)

    return reverb * scale_factor


def vst_reverb_process(params, input, sr, scale_factor: float = 1.0, hp_cutoff=None, norm=False, rev_external=None):
    if rev_external is not None:
        rev = external_vst3_set_params(params, rev_external)
        print(rev.parameters)

    else:
        rev = native_reverb_set_params(params)
        print(rev)

    rev_audio = process_reverb(rev, sr, input, hp_cutoff=hp_cutoff, norm=norm)
    rev_audio *= scale_factor

    return rev_audio


def merge_er_tail_rir(er, tail, sr, fade_length=128, trim=None, offset=0, fade=True):

    if len(tail.T) > abs(len(er.T) - offset):
        er_rir = pad_signal(er, len(er), len(tail.T) - abs(len(er.T) - offset))
    else:
        er_rir = er

    if fade:
        tail = tail * cosine_fade(len(tail.T), abs(len(er.T) - offset), False)

    start_point = offset
    print(f'Start point: {start_point}')
    print(er_rir.shape)
    print(tail.shape)
    er_rir[:, start_point:] += tail

    if trim is not None:
        er_rir = er_rir[:, :(trim * sr)]
        er_rir *= cosine_fade(len(er_rir.T), fade_length)

    return er_rir


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
