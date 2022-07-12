import pedalboard

from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.pedalboard_functions import *
from scripts.audio_functions.audio_metrics import *
from scripts.audio_functions.signal_generation import create_impulse
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.dict_functions import exclude_keys


def reverb_distance_native(params, params_dict, input_audio, ref_audio, sample_rate, pre_norm=False):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    reverb_to_match = native_reverb_set_params(params_dict)
    audio_to_match = plugin_process(reverb_to_match, input_audio, sample_rate)

    audio_to_match = pd_highpass_filter(audio_to_match, 3, sample_rate)

    loss = 0.0

    if pre_norm:
        ref_audio = ref_audio / np.max(abs(ref_audio))
        audio_to_match = audio_to_match / np.max(abs(audio_to_match))
        ref_audio[np.isnan(ref_audio)] = 0
        audio_to_match[np.isnan(audio_to_match)] = 0

    loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
                    mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])

    return loss


def reverb_distance_external(params, vst3, params_dict, input_audio, ref_audio, sample_rate, pre_norm=False):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    external_vst3_set_params(params_dict, vst3)
    audio_to_match = plugin_process(vst3, input_audio, sample_rate)

    audio_to_match = pd_highpass_filter(audio_to_match, 3, sample_rate)

    loss = 0.0

    if pre_norm:
        ref_audio = ref_audio / np.max(abs(ref_audio))
        audio_to_match = audio_to_match / np.max(abs(audio_to_match))
        ref_audio[np.isnan(ref_audio)] = 0
        audio_to_match[np.isnan(audio_to_match)] = 0

    loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
                    mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])

    return loss


def merged_rir_distance(params, params_dict, input_audio, ref_audio, er_path, sample_rate, vst3=None, pre_norm=False):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    impulse = create_impulse(sample_rate * 6, stereo=True)

    scale = params_dict['scale']
    par = exclude_keys(params_dict, 'scale')

    rir_tail = vst_reverb_process(par, impulse, sample_rate, scale_factor=scale, hp_cutoff=20, rev_external=vst3)

    rir_er, sr_er = sf.read(er_path)
    rir_er = rir_er.T

    fade_in = int(5 * sample_rate * 0.001)
    merged_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, fade_length=fade_in, trim=3)

    sf.write('audio/conv/current_rir.wav', merged_rir.T, sample_rate)

    conv = pedalboard.Convolution('audio/conv/current_rir.wav', mix=1.0)

    audio_to_match = conv(input_audio, sample_rate)

    if pre_norm:
        ref_audio = normalize_audio(ref_audio, nan_check=True)
        audio_to_match = normalize_audio(audio_to_match, nan_check=True)

    loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
                    mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])

    return loss


# def merged_rir_distance_native(params, params_dict, input_audio, ref_audio, er_path, sample_rate, pre_norm=False):
#
#     for idx, par in enumerate(params_dict):
#         params_dict[par] = params[idx]
#
#     impulse = create_impulse(sample_rate * 6, stereo=True)
#
#     rir_tail = vst_reverb_process(params_dict, impulse, sample_rate, scale_factor=1.0, hp_cutoff=20, rev_external=None)
#
#     rir_er, sr_er = sf.read(er_path)
#     rir_er = rir_er.T
#
#     fade_in = int(5 * sample_rate * 0.001)
#     merged_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, fade_length=fade_in, trim=3)
#
#     sf.write('audio/conv/current_rir.wav', merged_rir.T, sample_rate)
#
#     conv = pedalboard.Convolution('audio/conv/current_rir.wav', mix=1.0)
#
#     audio_to_match = conv(input_audio, sample_rate)
#
#     if pre_norm:
#         ref_audio = normalize_audio(ref_audio, nan_check=True)
#         audio_to_match = normalize_audio(audio_to_match, nan_check=True)
#
#     loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
#                     mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])
#
#     return loss
#
#
# def merged_rir_distance_external(params, vst3, params_dict, input_audio, ref_audio,
#                                  er_path, sample_rate, pre_norm=False):
#
#     for idx, par in enumerate(params_dict):
#         params_dict[par] = params[idx]
#
#     impulse = create_impulse(sample_rate * 6, stereo=True)
#
#     rir_tail = vst_reverb_process(params_dict, impulse, sample_rate, scale_factor=1.0, hp_cutoff=20, rev_external=vst3)
#
#     rir_er, sr_er = sf.read(er_path)
#     rir_er = rir_er.T
#
#     fade_in = int(5 * sample_rate * 0.001)
#     merged_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, fade_length=fade_in, trim=3)
#
#     sf.write('audio/conv/current_rir.wav', merged_rir.T, sample_rate)
#
#     conv = pedalboard.Convolution('audio/conv/current_rir.wav', mix=1.0)
#
#     audio_to_match = conv(input_audio, sample_rate)
#
#     if pre_norm:
#         ref_audio = normalize_audio(ref_audio, nan_check=True)
#         audio_to_match = normalize_audio(audio_to_match, nan_check=True)
#
#     loss = np.mean([mel_spectrogram_l1_distance(ref_audio[0], audio_to_match[0], sample_rate),
#                     mel_spectrogram_l1_distance(ref_audio[1], audio_to_match[1], sample_rate)])
#
#     return loss


