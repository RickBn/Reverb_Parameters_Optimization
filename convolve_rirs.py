from scripts.direct_path_removal import *


def convolve_rir(rir_name: str):
    input_path = 'audio/input/sounds/48/speech/_trimmed/loudnorm/_todo/'
    input_name = 'speech'

    rir = rir_name

    rir_path = f'audio/input/chosen_rirs/HOA/{rir}/_todo/'
    rir_file_names = directory_filter(rir_path)
    input_file_names = os.listdir(input_path)

    result_path = f'audio/results/HOA/{rir}/{input_name}/'

    result_file_names = [x.replace(".wav", '_ref.wav') for x in input_file_names]

    batch_fft_convolve(input_path, result_file_names, rir_path, save_path=result_path,
                       return_convolved=False, scale_factor=1.0, norm=False)

    # FREEVERB LATE ///////////////////////////////////////////////////////////////

    vst_rir_path = f'audio/vst_rirs/stereo/{rir}/'
    vst_rir_names = [x.replace(".wav", '_Freeverb.wav') for x in rir_file_names]

    result_path = f'audio/results/stereo/{rir}/{input_name}/late_only/'
    result_file_names = [x.replace(".wav", '_late_fv.wav') for x in input_file_names]

    batch_fft_convolve(input_path, result_file_names, vst_rir_path, vst_rir_names, result_path,
                       return_convolved=False, scale_factor=1.0, norm=False)

    # 90 SHIFT ////////////////////////////////////////////////////////////////////

    # vst_rir_path = f'audio/vst_rirs/stereo/{rir}/shifted/'
    # vst_rir_names = os.listdir(vst_rir_path)
    #
    # result_path = f'audio/results/stereo/{rir}/{input_name}/shifted/'
    # result_file_names = [x.replace(".wav", '_shifted_fv.wav') for x in input_file_names]
    #
    # batch_fft_convolve(input_path, result_file_names, vst_rir_path, vst_rir_names, result_path,
    #                    return_convolved=False, scale_factor=1.0, norm=False)

    # HOA ER //////////////////////////////////////////////////////////////////////

    trimmed_rir_path = f'audio/trimmed_rirs/HOA/{rir}/'
    trimmed_rir_names = rir_file_names

    result_path = f'audio/results/HOA/{rir}/{input_name}/early_only/'
    result_file_names = [x.replace(".wav", '_early_fv.wav') for x in input_file_names]

    batch_fft_convolve(input_path, result_file_names, trimmed_rir_path, trimmed_rir_names, result_path,
                       return_convolved=False, scale_factor=1.0, norm=False)

    # HOA/BIN LATE /////////////////////////////////////////////////////////////////

    late_rir_path = f'audio/trimmed_rirs/bin/{rir}/'
    late_rir_names = rir_file_names

    result_path = f'audio/results/bin/late_only/{rir}/{input_name}/'
    result_file_names = [x.replace(".wav", '_late_bin.wav') for x in input_file_names]

    batch_fft_convolve(input_path, result_file_names, late_rir_path, late_rir_names, result_path,
                       return_convolved=False, scale_factor=1.0, norm=False)

    print(0)


if __name__ == "__main__":
    convolve_rir("spergair")
