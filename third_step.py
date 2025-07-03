from scripts.reverb_parameters_optimize import match_coef_scale, get_param_error_statistic, plot_abscoeff_comparison, get_rt_err
from scripts.audio.signal_generation import *
from scripts.vst_rir_generation import vst_reverb_process
import soundfile as sf
import pedalboard
import yaml

rir_names = ['SDN050']

# Number of iterations of gp_minimize
n_iterations = 5#150  # 250

# Number of initial points used by gp_minimize
n_initial_points = 4#50

#  gp_minimize, forest_minimize, gbrt_minimize
optimizer = 'gp_minimize'
# Controls how much of the variance in the predicted values should be taken into account. If set to be very high,
# then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
# Non sembra influire
optimizer_kappa = 1.96  # 1.96

# Controls how much improvement one wants over the previous best values. Used when the acquisition is either "EI" or
# "PI".
optimizer_xi = 0.01,

# Number of cores to run in parallel. If n_jobs=-1, then number of jobs is set to number of cores.
n_jobs = -1

# Whether to remove direct. Set to False because the direct is already removed when SDN generates the RIR with the parameter line_of_sight
remove_direct = False

vst_path = "vst3/Real time SDN_25ch.vst3"
vst_name = 'SDN'


if __name__ == "__main__":

    rev_plugin = pedalboard.load_plugin(vst_path)

    for rir_name in rir_names:

        optimized_params_dict2 = {}
        optimized_params_dict3 = {}

        print(f'FITTING ROOM: {rir_name}')

        folder = f'stereo/{rir_name}/'
        rir_path = f'audio/input/chosen_rirs/{folder}_todo/pos01.wav'

        target_rir, sr = sf.read(rir_path)
        target_rir = target_rir.T

        impulse = create_impulse(target_rir.shape[1], n_channels=target_rir.shape[0])

        # Load parameters
        pred2_params_path = rf'audio\params\stereo\dim_red_150_hypercardioid_20log\omni_starting _point\{rir_name}\pos01\SDN\SDN.yml'
        with open(pred2_params_path, "r") as stream:
            try:
                optimized_params_dict2['pos01'] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        optimized_params_dict2['pos01'] = optimized_params_dict2['pos01']['parameters']

        target_params_path = f'audio/input/chosen_rirs/{folder}/parameters.yml'
        with open(target_params_path, "r") as stream:
            try:
                target_params_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print('3° STEP OF MATCHING (scale to match T60)')
        optimized_params_dict3['pos01'], loss_end_scale_coef = match_coef_scale(target_rir, sr, optimized_params_dict2['pos01'].copy(),
                                                                      rev_plugin, impulse, remove_direct, n_iterations,
                                                                      n_initial_points, optimizer_kappa, optimizer,
                                                                      n_jobs, optimizer_xi)

        mae_overall2 = get_param_error_statistic(['pos01.wav'], target_params_dict, optimized_params_dict2, 'prediction 2',
                                                params_path=None, effect_folder=None, venv_name=rir_name)
        mae_overall3 = get_param_error_statistic(['pos01.wav'], target_params_dict, optimized_params_dict3, 'prediction 3',
                                                params_path=None, effect_folder=None, venv_name=rir_name)

        plot_abscoeff_comparison(['pos01.wav'], target_params_dict, optimized_params_dict2, optimized_params_dict3,
                                 params_path=None, effect_folder=None, venv_name=rir_name,
                                 names=['Original', 'Prediction 2', 'Prediction 3'])

        # Generate SRIRs with direct path
        target_params_dict['pos01']['render_line_of_sight'] = True
        target_rir_direct = vst_reverb_process(target_params_dict['pos01'], impulse, sr, scale_factor=1,
                                               rev_external=rev_plugin, norm=False)

        optimized_params_dict2['render_line_of_sight'] = True
        rir_tail_direct2 = vst_reverb_process(optimized_params_dict2, impulse, sr, scale_factor=1,
                                             rev_external=rev_plugin, norm=False)

        optimized_params_dict3['render_line_of_sight'] = True
        rir_tail_direct3 = vst_reverb_process(optimized_params_dict3, impulse, sr, scale_factor=1,
                                             rev_external=rev_plugin, norm=False)

        print('ACOUSTICAL PARAMETERS ERROR - PREDICTION 2:')
        get_rt_err(target_rir_direct, rir_tail_direct2, sr)
        print('ACOUSTICAL PARAMETERS ERROR - PREDICTION 3:')
        get_rt_err(target_rir_direct, rir_tail_direct3, sr)