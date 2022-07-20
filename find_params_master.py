from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":
    # rir_path = 'audio/_SIM/'
    # er_path = 'audio/_SIM_TRIMMED/'

    rir_path = 'audio/input/chosen_rirs/'
    er_path = 'audio/trimmed_rirs/'
    result_path = 'audio/results/'
    input_path = 'audio/input/sounds/'

    find_params_merged(rir_path, er_path, result_path, input_path, generate_references=True, pre_norm=False)



