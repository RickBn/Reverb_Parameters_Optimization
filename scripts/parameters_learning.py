import pedalboard

from scripts.audio.audio_manipulation import *
from scripts.audio.pedalboard_functions import *
from scripts.audio.audio_metrics import *
from scripts.audio.signal_generation import create_impulse
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.dict_functions import exclude_keys


def rir_distance(params, params_dict, input_audio, ref_audio, rir_er, offset, sample_rate,
                 vst3=None, merged=False, pre_norm=False):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    n_channels = 2 if input_audio.shape[0] == 2 else 1

    impulse = create_impulse(sample_rate * 3, n_channels=n_channels)

    scale = params_dict['scale']
    par = exclude_keys(params_dict, 'scale')

    rir_tail = vst_reverb_process(par, impulse, sample_rate, scale_factor=scale, hp_cutoff=20, rev_external=vst3)

    if merged:
        if rir_tail.ndim == 1 and rir_tail.ndim != rir_er.ndim:
            rir_tail = np.stack([rir_tail] * rir_er.shape[0])
        final_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, trim=3, offset=offset)
    else:
        # rir_tail = rir_tail * cosine_fade(len(impulse.T), abs(len(rir_er.T) - offset), False)
        final_rir = pad_signal(rir_tail, n_channels, np.max(offset), pad_end=False)

    if input_audio.ndim == 1 and input_audio.ndim != ref_audio.ndim:
        input_audio = np.stack([input_audio] * ref_audio.shape[0])

    audio_to_match = scipy.signal.fftconvolve(input_audio, final_rir, mode='full', axes=1)

    audio_to_match = pd_highpass_filter(audio_to_match, order=3, sr=sample_rate, cutoff=20.0)

    if ref_audio.ndim > 1:
        ref_audio = ref_audio[:, :len(input_audio.T)]
        audio_to_match = audio_to_match[:, :len(input_audio.T)]
    else:
        ref_audio = ref_audio[:len(input_audio.T)]
        audio_to_match = audio_to_match[:len(input_audio.T)]

    print(ref_audio.shape)
    print(audio_to_match.shape)

    if pre_norm:
        ref_audio = normalize_audio(ref_audio, nan_check=True)
        audio_to_match = normalize_audio(audio_to_match, nan_check=True)

    loss = mel_spectrogram_l1_distance(ref_audio, audio_to_match, sample_rate)

    return loss


def merged_rir_distance_1D(params, params_dict, input_audio, ref_audio, rir_er, offset,
                           sample_rate, vst3=None, pre_norm=False):

    for idx, par in enumerate(params_dict):
        params_dict[par] = params[idx]

    impulse = create_impulse(sample_rate * 3, n_channels=1)

    scale = params_dict['scale']
    par = exclude_keys(params_dict, 'scale')

    print(scale)

    rir_tail = vst_reverb_process(par, impulse, sample_rate,
                                  scale_factor=scale, hp_cutoff=20, norm=False, rev_external=vst3)

    rir_er = np.array([rir_er])
    rir_tail = np.array([rir_tail])

    print(rir_er.shape)
    print(rir_tail.shape)

    merged_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, trim=3, offset=offset)

    audio_to_match = scipy.signal.fftconvolve(input_audio, merged_rir[0], mode='full', axes=0)

    audio_to_match = pd_highpass_filter(audio_to_match, order=3, sr=sample_rate, cutoff=20.0)

    ref_audio = ref_audio[:len(input_audio)]
    audio_to_match = audio_to_match[:len(input_audio)]

    print(ref_audio.shape)
    print(audio_to_match.shape)

    if pre_norm:
        ref_audio = normalize_audio(ref_audio, nan_check=True, by_row=False)
        audio_to_match = normalize_audio(audio_to_match, nan_check=True, by_row=False)

    loss = mel_spectrogram_l1_distance(ref_audio, audio_to_match, sample_rate)

    return loss
