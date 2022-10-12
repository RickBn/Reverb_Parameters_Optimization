from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":

    #folder = 'HOA/sdn_project/bf4/'
    rir_name = 'Living Room'
    folder = f'stereo/{rir_name}/'

    rir_path = f'audio/input/chosen_rirs/{folder}/_todo/'
    er_path = f'audio/trimmed_rirs/{folder}'
    armodel_path = f'audio/armodels/{folder}'
    merged_rir_path = f'audio/merged_rirs/{folder}'
    vst_rir_path = f'audio/vst_rirs/{folder}'
    params_path = f'audio/params/{folder}'
    result_path = f'audio/results/{folder}'
    input_path = f'audio/input/sounds/48/speech/_trimmed/loudnorm/'

    find_params_merged(rir_path,
                       er_path,
                       armodel_path,
                       merged_rir_path,
                       vst_rir_path,
                       params_path,
                       result_path,
                       input_path,
                       generate_references=False,
                       pre_norm=False)

    # rir_path = f'audio/trimmed_rirs/bin/{rir_name}/'
    # er_path = f'audio/trimmed_rirs/{folder}/_done/'
    # armodel_path = f'audio/armodels/{folder}'
    # merged_rir_path = f'audio/merged_rirs/temp/{folder}'
    # vst_rir_path = f'audio/vst_rirs/temp/{folder}'
    # params_path = f'audio/params/temp/{folder}'
    # result_path = f'audio/results/temp/{folder}'
    # input_path = f'audio/input/sounds/48/speech/_trimmed/loudnorm/'
    #
    # find_params_late(rir_path,
    #                  er_path,
    #                  armodel_path,
    #                  merged_rir_path,
    #                  vst_rir_path,
    #                  params_path,
    #                  result_path,
    #                  input_path,
    #                  generate_references=True,
    #                  pre_norm=False)
    #




