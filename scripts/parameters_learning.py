# import pedalboard
from pedalboard_change_channel_limit import pedalboard

from scripts.audio.audio_manipulation import *
from scripts.audio.pedalboard_functions import *
from scripts.audio.audio_metrics import *
from scripts.audio.signal_generation import create_impulse
from scripts.vst_rir_generation import vst_reverb_process, merge_er_tail_rir
from scripts.utils.dict_functions import exclude_keys
from scripts.params_dim_reduction import reconstruct_original_params
from scripts.audio.rir_functions import remove_direct_from_rir

n_wall_bands = 8


def pol2cart(pol):
    # pol: [rho, phi]
    rho = pol[0]
    phi = pol[1]

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    if x < 0 or y < 0:
        pass

    return [x, y]


def rir_distance(params, params_dict, input_audio, ref_audio, rir_er, offset, sample_rate,
                 vst3=None, merged=False, pre_norm=False, fixed_params: dict = None, fixed_params_bool: list = [],
                 match_only_late: bool = True, dim_red_mdl = None, same_coef_walls: bool = False,
                 force_last2_bands_equal: bool = False, fade_length: int = 256, polar_coords: bool = False, impulse = []):

    # for idx, par in enumerate(params_dict):
    #     params_dict[par] = params[idx]

    # import random
    # # params = [random.random() for p in params]
    # params[0] = random.random()
    # params[1] = random.uniform(0, np.pi/2)

    if dim_red_mdl is not None:
        if polar_coords:
            params = pol2cart(params)

        params = reconstruct_original_params(dim_red_mdl, params)

    if force_last2_bands_equal:
        new_params = []
        for i, p in enumerate(params):
            new_params.append(p)
            if i % (n_wall_bands-2) == 5:
                new_params.append(p)
                new_params.append(p)

        params = new_params
        del new_params

    params_count = 0
    for idx, par in enumerate(params_dict):
        if fixed_params_bool[idx]:
            params_dict[par] = fixed_params[par]
        else:
            params_dict[par] = params[params_count]
            if same_coef_walls:
                params_count = (params_count + 1) % n_wall_bands
            else:
                params_count = params_count + 1

    if 'scale' in params_dict.keys():
        scale = params_dict['scale']
        par = exclude_keys(params_dict, 'scale')
    else:
        scale = 1
        par = params_dict

    rir_tail = vst_reverb_process(par, impulse, sample_rate, scale_factor=scale, hp_cutoff=20, rev_external=vst3)
    # if np.isnan(rir_tail).any():
    #     np.nan_to_num(rir_tail, copy=False, nan=0)

    rir_tail = remove_direct_from_rir(rir_tail)

    # if merged:
    if match_only_late:
        if rir_tail.ndim == 1 and rir_tail.ndim != rir_er.ndim:
            rir_tail = np.stack([rir_tail] * rir_er.shape[0])

        # V1: attaccare con cross-fade prima della loss er originali e late matchata e poi calcolare la loss
        final_rir = merge_er_tail_rir(rir_er, rir_tail, sample_rate, fade_length=fade_length, trim=3, offset=offset)

        # # V2: fade-in prima della loss di late matchata e della RIR originale (per togliere er), poi calcolare la loss
        # for ch in range(n_channels):
        #     rir_tail[ch,:] = rir_tail[ch,:] * np.concatenate([np.zeros(int(offset[ch])),
        #                                                       cosine_fade(int(len(rir_tail[ch, :].T) - offset[ch]),
        #                                                                   fade_length, False)])
        # final_rir = rir_tail

    else:
        # rir_tail = rir_tail * cosine_fade(len(impulse.T), abs(len(rir_er.T) - offset), False)
        # if match_only_late:
        #     final_rir = pad_signal(rir_tail, n_channels, np.max(offset), pad_end=False)
        # else:
        final_rir = rir_tail

    if input_audio.ndim == 1 and input_audio.ndim != ref_audio.ndim:
        input_audio = np.stack([input_audio] * ref_audio.shape[0])
    # Convolve reverberator current output with sweep
    audio_to_match = scipy.signal.fftconvolve(input_audio, final_rir, mode='full', axes=1)
    if np.isnan(audio_to_match).any():
        # for ch in range(audio_to_match.shape[0]):
        #     audio_to_match[ch, :] = np.convolve(input_audio[ch, :], final_rir[ch, :], mode='full')
        np.nan_to_num(audio_to_match, copy=False, nan=0)

    audio_to_match = pd_highpass_filter(audio_to_match, order=3, sr=sample_rate, cutoff=20.0)

    if ref_audio.ndim > 1:
        ref_audio = ref_audio[:, :len(input_audio.T)]
        audio_to_match = audio_to_match[:, :len(input_audio.T)]
    else:
        ref_audio = ref_audio[:len(input_audio.T)]
        audio_to_match = audio_to_match[:len(input_audio.T)]

    # print(ref_audio.shape)
    # print(audio_to_match.shape)

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
