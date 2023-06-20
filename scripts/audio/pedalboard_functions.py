import numpy as np
import pedalboard
import pedalboard_native
import skopt
from pedalboard import Pedalboard, Reverb, load_plugin


def retrieve_external_vst3_params(vst3: pedalboard.VST3Plugin) -> (dict, list):
    params = vst3.parameters

    param_names = list(params.keys())
    param_names.remove('bypass')

    parameters = {}
    ranges = []

    for p in param_names:
        parameters[p] = vst3.__getattr__(p)
        ranges.append(skopt.space.space.Real(params[p].range[0], params[p].range[1], transform='identity'))
    # ranges.append((params[p].range[0], params[p].range[1]))

    return parameters, ranges


def external_vst3_set_params(params: dict, vst3: pedalboard.VST3Plugin) \
        -> pedalboard.VST3Plugin:
    for p in params:
        vst3.__setattr__(p, params[p])

    print(params)

    return vst3


def native_reverb_set_params(params: dict, full_wet=True) -> pedalboard_native.Reverb:
    #r = Reverb(freeze_mode=0)

    if full_wet:
        r = Reverb(freeze_mode=0, dry_level=0.0, wet_level=1.0)

    else:
        r = Reverb(freeze_mode=0)

    for p in params:
        r.__setattr__(p, params[p])

    print(r)

    return r


def plugin_process(vst3, audio, sr):
    # if audio.shape[0] > 2:
    #     effected = [vst3(audio[0], sample_rate=sr)]
    #     for ch in range(1, audio.shape[0]):
    #         effected = np.concatenate((effected, [vst3(audio[ch], sample_rate=sr)]), axis=0)
    # else:
    effected = vst3(audio, sample_rate=sr)

    return effected


def board_process(board, audio):
    effected = board(audio)

    return effected


def pd_highpass_filter(audio: np.ndarray, order: int, sr: int, cutoff=20.0):
    filter = np.array([pedalboard.HighpassFilter(cutoff_frequency_hz=cutoff)] * audio.ndim)

    if audio.ndim == 1:
        audio = audio = filter[0](audio, sr)

    elif audio.ndim == len(filter):
        for ch, fil in enumerate(filter):
            for i in range(0, order):
                audio[ch] = fil(audio[ch], sr)

    return audio

