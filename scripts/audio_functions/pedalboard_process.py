from scripts.audio_functions.audio_manipulation import *


def process_external_reverb(params, rev_external, sr, input_audio, hp_cutoff: float = None, norm: bool = True):

	external_vst3_set_params(params, rev_external)

	reverb_audio_external = plugin_process(rev_external, input_audio, sr)

	if hp_cutoff is not None:
		reverb_audio_external = pd_highpass_filter(reverb_audio_external, 3, sr, hp_cutoff)

	if norm:
		reverb_audio_external = normalize_audio(reverb_audio_external)

	return reverb_audio_external


def process_native_reverb(params, sr, input_audio, hp_cutoff: float = None, norm: bool = True):

	opt_rev_native = native_reverb_set_params(params)

	reverb_audio_native = plugin_process(opt_rev_native, input_audio, sr)

	if hp_cutoff is not None:
		reverb_audio_native = pd_highpass_filter(reverb_audio_native, 3, sr, hp_cutoff)

	if norm:
		reverb_audio_native = normalize_audio(reverb_audio_native)

	return reverb_audio_native
