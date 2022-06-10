from scripts.audio_functions.audio_manipulation import *


def process_external_reverb(rev_external, sr, input_audio, scale_factor: float = 1.0,
                            hp_cutoff: float = None, norm: bool = True):

	reverb_audio_external = plugin_process(rev_external, input_audio, sr)

	if hp_cutoff is not None:
		reverb_audio_external = pd_highpass_filter(reverb_audio_external, 3, sr, hp_cutoff)

	if norm:
		reverb_audio_external = normalize_audio(reverb_audio_external)

	return reverb_audio_external * scale_factor


def process_native_reverb(rev_native, sr, input_audio, scale_factor: float = 1.0,
                          hp_cutoff: float = None, norm: bool = True):

	reverb_audio_native = plugin_process(rev_native, input_audio, sr)

	if hp_cutoff is not None:
		reverb_audio_native = pd_highpass_filter(reverb_audio_native, 3, sr, hp_cutoff)

	if norm:
		reverb_audio_native = normalize_audio(reverb_audio_native)

	return reverb_audio_native * scale_factor
