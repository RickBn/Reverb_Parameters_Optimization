from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":

    folder = 'MARCo/bf4/'

    rir_path = 'audio/input/chosen_rirs/HOA/' + folder
    er_path = 'audio/trimmed_rirs/HOA/' + folder
    armodel_path = 'audio/armodels/HOA/' + folder
    merged_rir_path = 'audio/merged_rirs/HOA/' + folder
    vst_rir_path = 'audio/vst_rirs/HOA/' + folder
    result_path = 'audio/results/HOA/' + folder
    input_path = 'audio/input/sounds/48/speech/'

    find_params_merged(rir_path, er_path, armodel_path, merged_rir_path, vst_rir_path, result_path, input_path,
                       generate_references=True, pre_norm=False)



