from scripts.audio.rir_functions import *

if __name__ == "__main__":
    frame_size = 512
    sr = 48000
    fade_factor = 4
    early_trim = 500

    rir_name = 'MARCo'
    folder = f'stereo/{rir_name}/'

    rir_path = f'audio/input/chosen_rirs/{folder}/_todo/'
    armodel_path = 'audio/armodels/' + folder

    a_a, p_a, l_a = rir_psd_metrics(rir_path, sr, frame_size, fade_factor, early_trim, direct_offset=True,
                                  ms_encoding=False, save_path=armodel_path)

    knee_save_path = 'images/lsd/' + folder

    # arm_dict = np.load('audio/armodels/arm_dict_ms.npy', allow_pickle=True)[()]
    lsd_dict = np.load('audio/armodels/' + folder + 'lsd_dict.npy', allow_pickle=True)[()]

    cut_dict, offset_dict = rir_er_detection(rir_path, lsd_dict, img_path=None, cut_dict_path=armodel_path)

    trim_rir_save_path = 'audio/trimmed_rirs/' + folder
    trim_rir_dict = rir_trim(rir_path, cut_dict, fade_length=128, save_path=trim_rir_save_path)

    print(0)
