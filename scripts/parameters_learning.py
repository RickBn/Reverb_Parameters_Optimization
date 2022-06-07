from scripts.audio_functions.audio_manipulation import *
from scripts.audio_functions.pedalboard_functions import *
from scripts.audio_functions.audio_metrics import *


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


def merged_rir_distance_native(params, params_dict, input_audio, ref_audio, sample_rate, pre_norm=False):

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