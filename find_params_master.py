from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":

    rir_path = 'audio/input/chosen_rirs/HOA/MARCo/bf4/'
    er_path = 'audio/trimmed_rirs/HOA/MARCo/bf4/'
    result_path = 'audio/results/HOA/MARCo/bf4/'
    input_path = 'audio/input/sounds/48/'

    find_params_merged(rir_path, er_path, result_path, input_path, generate_references=True, pre_norm=False)



