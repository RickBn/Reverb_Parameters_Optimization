from scripts.parameters_learning import *
from scripts.utils.plot_functions import *

from scripts.audio.signal_generation import *
from scripts.audio.pedalboard_functions import *
from scripts.audio.rir_functions import *


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
        # print(rev.parameters)

    else:
        rev = native_reverb_set_params(params)
        # print(rev)

    print(rev.render_line_of_sight)

    rev_audio = process_reverb(rev, sr, input, hp_cutoff=hp_cutoff, norm=norm)
    rev_audio *= scale_factor

    return rev_audio


def merge_er_tail_rir(er, tail, sr, fade_length=256, trim=None, offset=0, fade=True):
    er_rir = np.zeros(tail.shape)

    for ch in range(er.shape[0]):
        offset[ch] = int(offset[ch])
        if len(tail.T) > abs(len(er.T) - offset[ch]):
            er_rir[ch,:] = pad_signal(np.expand_dims(er[ch,:], axis=0), n_channels=1,
                                      pad_length=len(tail[ch,:].T) - len(er[ch,:].T))
                                      # pad_length=len(tail[ch,:].T) - abs(len(er[ch,:].T) - offset[ch]))
        else:
            er_rir[ch,:] = er[ch,:]

        if fade:
            cos_fade = np.concatenate([np.zeros(offset[ch] - round(fade_length/2)), cosine_fade(len(tail[ch, :].T) - offset[ch] - round(fade_length/2), fade_length, False)])
            tail[ch,:] = tail[ch,:] * cos_fade
            er_rir[ch,:] = er_rir[ch,:] * (cos_fade * (-1) + 1)

        # start_point = offset[ch]
        # print(f'Start point: {start_point}')
        # print(er_rir.shape)
        # print(tail.shape)
        # er_rir[ch, start_point:] += tail[ch,:]
        er_rir[ch,:] += tail[ch,:]

        if trim is not None:
            er_rir[ch,:] = er_rir[ch, :(trim * sr)]
            er_rir[ch,:] *= cosine_fade(len(er_rir[ch,:].T), fade_length)

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
