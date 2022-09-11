from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":

    # folder = 'HOA/MARCo/bf4/'
    folder = 'stereo/spergair/'

    rir_path = f'audio/input/chosen_rirs/{folder}'
    er_path = f'audio/trimmed_rirs/{folder}'
    armodel_path = f'audio/armodels/{folder}'
    merged_rir_path = f'audio/merged_rirs/{folder}'
    vst_rir_path = f'audio/vst_rirs/{folder}'
    result_path = f'audio/results/{folder}'
    input_path = f'audio/input/sounds/48/mozart/_trimmed/'

    find_params_merged(rir_path, er_path, armodel_path, merged_rir_path, vst_rir_path, result_path, input_path,
                       generate_references=True, pre_norm=False)



