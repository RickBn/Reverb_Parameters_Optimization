# import pedalboard
from pedalboard_change_channel_limit import pedalboard
from scipy.fft import rfft

from scripts.audio.audio_manipulation import *
from scripts.audio.pedalboard_functions import *
from scripts.audio.audio_metrics import *
from scripts.audio.signal_generation import create_impulse
from scripts.audio.rir_functions import beaforming_ambisonic
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

def beamforming_loss(target_rir, matched_rir, sr: int = 48000):
    # TODO: controllare, gestire caso omni e efficientamento fft con zeri alla fine
    target_rir_f = rfft(target_rir)
    matched_rir_f = rfft(matched_rir)
    # target_rir_f = librosa.power_to_db(target_rir_f, ref=np.max)
    # matched_rir_f = librosa.power_to_db(matched_rir_f, ref=np.max)
    target_rir_f = np.fft.rfft(target_rir, axis=0)/target_rir.shape[0]

    loss = np.mean(np.abs(np.abs(target_rir_f) - np.abs(matched_rir_f)))

    return loss

def rir_distance(params, params_dict, input_sweep, target_rir, rir_er, offset, sample_rate,
                 vst3=None, merged=False, pre_norm=False, fixed_params_bool: list = [],
                 match_only_late: bool = True, dim_red_mdl = None, same_coef_walls: bool = False,
                 force_last2_bands_equal: bool = False, fade_length: int = 256, polar_coords: bool = False, impulse = [],
                 remove_direct: bool = False, wall_idx_ambisonic: int = None,
                 beamformer=None, engine=None, playback=None, wall_order=None
                 ):

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

    if wall_idx_ambisonic is None:
        for idx, par in enumerate(params_dict):
            if not fixed_params_bool[idx]:
                params_dict[par] = params[params_count]
                if same_coef_walls:
                    params_count = (params_count + 1) % n_wall_bands
                else:
                    params_count = params_count + 1
    # Ambisonic
    else:
        params_count = 0
        for idx, par in enumerate(params_dict):
            if not fixed_params_bool[idx]:

                # If not omni, set only to the current wall coeffs
                if wall_idx_ambisonic > 0:
                     if par.split('_wall_')[1] == wall_order[wall_idx_ambisonic]:
                        params_dict[par] = params[params_count]
                        if params_count == n_wall_bands - 1:
                            break
                        params_count = params_count + 1

                # If omni, set to all coeffs of all walls
                else:
                    params_dict[par] = params[params_count]
                    params_count = (params_count + 1) % n_wall_bands

    # params_count = 0
    # params_count_omni = 0
    # for idx, par in enumerate(params_dict):
    #     if fixed_params_bool[idx]:
    #         params_dict[par] = fixed_params[par]
    #     else:
    #         if wall_idx_ambisonic > 0:
    #             if par in params_already_fitted.keys():# Parameter in wall already fittes
    #                 params_dict[par] = params_already_fitted[par]
    #             elif par.split('_wall_')[1] == wall_order[wall_idx_ambisonic]:# Parameter in wall currently fitting
    #                 params_dict[par] = params[params_count]
    #                 params_count = params_count + 1
    #             else:# Parameter in wall not fitted
    #                 params_dict[par] = params_omni[params_count_omni]
    #                 params_count_omni = (params_count_omni + 1) % n_wall_bands
    #         else:
    #             params_dict[par] = params[params_count]
    #             if same_coef_walls or wall_idx_ambisonic == 0:
    #                 params_count = (params_count + 1) % n_wall_bands
    #             else:
    #                 params_count = params_count + 1

    if 'scale' in params_dict.keys():
        scale = params_dict['scale']
        par = exclude_keys(params_dict, 'scale')
    else:
        scale = 1
        par = params_dict

    matched_rir = vst_reverb_process(par, impulse, sample_rate, scale_factor=scale, hp_cutoff=None, rev_external=vst3,
                                     norm=True)
    # if np.isnan(matched_rir).any():
    #     np.nan_to_num(matched_rir, copy=False, nan=0)

    if remove_direct:
        matched_rir = remove_direct_from_rir(matched_rir)

    if wall_idx_ambisonic is not None:
        if wall_idx_ambisonic == 0:
            matched_rir = matched_rir[0, :]

        else:
            playback.set_data(matched_rir)
            matched_rir = beaforming_ambisonic(beamformer, engine, fixed_params=params_dict, wall_idx_ambisonic=wall_idx_ambisonic, wall_order=wall_order,
                                               length=matched_rir.shape[1] / sample_rate, window=True)
            matched_rir = matched_rir[0, :]

    # if merged:
    if match_only_late:
        if matched_rir.ndim == 1 and matched_rir.ndim != rir_er.ndim:
            matched_rir = np.stack([matched_rir] * rir_er.shape[0])

        # V1: attaccare con cross-fade prima della loss er originali e late matchata e poi calcolare la loss
        matched_rir = merge_er_tail_rir(rir_er, matched_rir, sample_rate, fade_length=fade_length, trim=3, offset=offset)

        # # V2: fade-in prima della loss di late matchata e della RIR originale (per togliere er), poi calcolare la loss
        # for ch in range(n_channels):
        #     matched_rir[ch,:] = matched_rir[ch,:] * np.concatenate([np.zeros(int(offset[ch])),
        #                                                       cosine_fade(int(len(matched_rir[ch, :].T) - offset[ch]),
        #                                                                   fade_length, False)])
        # final_rir = matched_rir

    # else:
        # matched_rir = matched_rir * cosine_fade(len(impulse.T), abs(len(rir_er.T) - offset), False)
        # if match_only_late:
        #     final_rir = pad_signal(matched_rir, n_channels, np.max(offset), pad_end=False)
        # else:
        # final_rir = matched_rir

    if input_sweep.ndim == 1 and input_sweep.ndim != target_rir.ndim:
        input_sweep = np.stack([input_sweep] * target_rir.shape[0])

    if wall_idx_ambisonic is None:
        # Convolve reverberator current output with sweep
        matched_rir = scipy.signal.fftconvolve(input_sweep, matched_rir, mode='full', axes=1)
        if np.isnan(matched_rir).any():
            # for ch in range(audio_to_match.shape[0]):
            #     audio_to_match[ch, :] = np.convolve(input_audio[ch, :], final_rir[ch, :], mode='full')
            np.nan_to_num(matched_rir, copy=False, nan=0)
    # else:
    #     audio_to_match = final_rir

    # matched_rir = pd_highpass_filter(matched_rir, order=3, sr=sample_rate, cutoff=20.0)

        if target_rir.ndim > 1:
            target_rir = target_rir[:, :len(input_sweep.T)]
            matched_rir = matched_rir[:, :len(input_sweep.T)]
        else:
            target_rir = target_rir[:len(input_sweep.T)]
            matched_rir = matched_rir[:len(input_sweep.T)]

    # print(ref_audio.shape)
    # print(audio_to_match.shape)

    if pre_norm:
        target_rir = normalize_audio(target_rir, nan_check=True)
        matched_rir = normalize_audio(matched_rir, nan_check=True)

    if wall_idx_ambisonic is None:
        loss = mel_spectrogram_l1_distance(target_rir, matched_rir, sample_rate)
    else:
        loss = beamforming_loss(target_rir[0,:], matched_rir, sr=sample_rate)

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
