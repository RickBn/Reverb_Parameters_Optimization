from scripts.reverb_parameters_optimize import *
from generate_sdn_rir import generate_rirs
import datetime
import warnings
warnings.filterwarnings("ignore")

plt.switch_backend('agg')

if __name__ == "__main__":

    start = datetime.datetime.now()

    #  gp_minimize, forest_minimize, gbrt_minimize
    optimizer = 'gp_minimize'
    # Controls how much of the variance in the predicted values should be taken into account. If set to be very high,
    # then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
    # Non sembra influire
    optimizer_kappa = 1.96#1.96
    # Controls how much improvement one wants over the previous best values. Used when the acquisition is either "EI" or
    # "PI".
    optimizer_xi = 0.01,

    # Whether match only the late reverberation or the entire RIR
    match_only_late = False

    # Whether to apply the dimensionality reduction to the walls coefficients
    # Path for pre-computed values of the 2PS
    apply_dim_red = {'pts_2d': r'.\wall_coeff_dim_reduction\PCA_data\324000_iterations\2d_projection_data.csv',
                     'pts_original': r'.\wall_coeff_dim_reduction\PCA_data\324000_iterations\filters_data.csv'}
    # Classic PCA
    # apply_dim_red = 'pca'
    # No dim reduction
    # apply_dim_red = False

    # Whether interpolate to return to the original space. Used only when apply_dim_red = True
    inv_interp = True

    # Whether to force the points in the unit circle. Used only when apply_dim_red = True and inv_interp = True
    unit_circle = False

    # Whether the optimizator works on polar coordinates instead of cartesian ones. Used only when apply_dim_red = True and n_dims_red = 2
    polar_coords = False

    # Whether all the walls have the same absorption coefficients. Not used if RIR Ambisonic
    same_coef_walls = True

    # Whether to set the absorption coef of the last 2 bands equal to the third to last
    force_last2_bands_equal = True

    # Number of iterations of gp_minimize
    n_iterations = 150#250

    # Number of initial points used by gp_minimize
    n_initial_points = 50

    # Sample length of the fade used in peak windowing [ms]
    fade_length = 20#256

    # Whether to remove direct. Set to False because the direct is already removed when SDN generates the RIR with the parameter line_of_sight
    remove_direct = False

    # Whether to apply the windows to isolate the 1st reflection when using Ambisonics RIRs
    window = True

    # Possibilities for the third step of matching
    # False: no third step
    # third_matching_step = False
    # rerun_matching_with_prev_par: Rifare un giro di matching come è ora usando come coef di partenza quelli calcolati nel 2° step
    # third_matching_step = 'rerun_matching_with_prev_par'

    # scale_coef_t60: Scalando i coefficienti per matchare t60 -> usando gp_minimize senza windowing per prima riflessione. 3 opzioni:
    # 		1 fattore di scala tunato da gp_minimize. Loss: Err di T60 sull'omni
    third_matching_step = 'scale_coef_t60'
    # 		6 fattori di scala (1 per parate) tunato da gp_minimize. Loss: err T60 per parete
    # third_matching_step = 'scale_coef_t60_perwall'
    # 		TODO: 2 fattori di scala per parete per trovare un tilt (tilt eq) dei coefficienti. Loss: erro T60 per bande per ogni parete

    n_iterations_third_step = 150

    # rerun_match_all_coef_together: Rifare un giro di matching considerando i coefficienti di tutte le pareti insieme (6*2=12 parametri che gp_minimize deve tunare). Loss: Mel spectrogram dell'EDR
    # third_matching_step = 'rerun_match_all_coef_together'

    # Used to perform the third matching step without doing the second step
    # load_previous_res_from = None
    if isinstance(apply_dim_red, dict):
        load_previous_res_from = r'audio/params/stereo/dim_red_150_hypercardioid_20log/omni_starting _point/'
        # load_previous_res_from = r'audio/params/stereo/dim_red_150_hypercardioid_20log_3stepConParametriPrecedenti/omni_starting _point/'
    elif apply_dim_red == 'pca':
        load_previous_res_from = r'audio/params/stereo/dim_red_pca_classic_150_hypercardioid_20log/omni_starting _point/'
    elif apply_dim_red == False:
        load_previous_res_from = r'audio/params/stereo/no_dim_red_150_hypercardioid_20log/omni_starting _point/'
    else:
        load_previous_res_from = None

    # Number of cores to run in parallel. If n_jobs=-1, then number of jobs is set to number of cores.
    n_jobs = -1

    # n_rirs = 1
    # rir_names = []
    # for r in range(n_rirs):
    #     rir_names.append(generate_rirs(param_type='sample', same_coef_per_wall=False))

    rir_names = ['SDN050']
        # , 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058',
        #          'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
    # rir_names = ['SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058',
    #              'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
    # rir_names = ['SDN056', 'SDN057', 'SDN058', 'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
    # rir_names = ['SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058', 'SDN059', 'SDN060', 'SDN061', 'SDN062', 'SDN063', 'SDN064', 'SDN065']
    # rir_names = ['SDN062', 'SDN063', 'SDN064']#['SDN059', 'SDN060', 'SDN061',
    # rir_names = ['SDN050', 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055', 'SDN056', 'SDN057', 'SDN058']#['SDN050', 'SDN051', 'SDN052', 'SDN053', 'SDN054', 'SDN055']
    # rir_names = ['SDN046', 'SDN047']#, 'SDN048', 'SDN049']

    err = dict()
    t60_err = dict()

    # Set the path of the reverberator (vst3):
    # - 'vst3/Real time SDN_25ch.vst3'
    # - 'vst3/FdnReverb.vst3'
    vst_path = "vst3/Real time SDN_25ch.vst3"
    vst_name = 'SDN'

    for rir_name in rir_names:
        print(f'FITTING ROOM: {rir_name}')

        # Set the name of the room:
        # - 'Living Room'
        # - 'MARCo'
        # - 'METu'
        # rir_name = 'SDN000'
        folder = f'stereo/{rir_name}/'

        rir_path = f'audio/input/chosen_rirs/{folder}_todo/'
        er_path = f'audio/trimmed_rirs/{folder}'
        armodel_path = f'audio/armodels/{folder}'
        merged_rir_path = f'audio/merged_rirs/{folder}'
        vst_rir_path = f'audio/vst_rirs/{folder}'
        baseline_rir_path = f'audio/vst_rirs/stereo/baseline/{rir_name}'
        params_path = f'audio/params/{folder}'
        original_params_path=f'audio/input/chosen_rirs/{folder}/parameters.yml'
        result_path = f'audio/results/{folder}'
        input_path = f'audio/input/sounds/48/speech/_trimmed/loudnorm/_todo/'
        fixed_params_path = f'fixed_parameters/{vst_name}/{rir_name}.yml'

        err[rir_name], t60_err[rir_name] = find_params(rir_path,
                                    er_path,
                                    armodel_path,
                                    merged_rir_path,
                                    vst_rir_path,
                                    baseline_rir_path,
                                    params_path,
                                    original_params_path,
                                    result_path,
                                    input_path,
                                    optimizer=optimizer,
                                    optimizer_kappa=optimizer_kappa,
                                    optimizer_xi=optimizer_xi,
                                    fixed_params_path=fixed_params_path,
                                    generate_references=False,
                                    original_er=False,
                                    pre_norm=False,
                                    vst_path=vst_path,
                                    n_iterations=n_iterations,
                                    match_only_late=match_only_late,
                                    apply_dim_red=apply_dim_red,
                                    same_coef_walls=same_coef_walls,
                                    force_last2_bands_equal=force_last2_bands_equal,
                                    n_initial_points=n_initial_points,
                                    inv_interp=inv_interp,
                                    unit_circle=unit_circle,
                                    polar_coords=polar_coords,
                                    fade_length=fade_length,
                                    window=window,
                                    third_matching_step=third_matching_step,
                                    n_iterations_third_step=n_iterations_third_step,
                                    load_previous_res_from=load_previous_res_from,
                                    n_jobs=n_jobs)

    print(f'MAE RIRs {list(err.values())}')
    print(f'Mean MAE for the RIRs {rir_names}: {np.mean(list(err.values())):.3f} ± {np.std(list(err.values())):.3f}')

    print()
    print(f'T60 RIRs [%] {list(t60_err.values())}')
    print(f'Mean T60 error [%] for the RIRs {rir_names}: {np.mean(list(t60_err.values())):.3f}% ± {np.std(list(t60_err.values())):.3f}%')


    stop = datetime.datetime.now()

    elapsed = stop - start

    print(f'Total elapsed time: {elapsed}')

    pass
# mae_sv = [0.074,0.086,0.123,0.118,0.069,0.109,0.061,0.087,0.056,0.074,0.054,0.089,0.126,0.119,0.102,0.066]
# t60_sv = [124.21,67.76,37.84,40.66,49.04,13.16,36.55,6.01,2.87,61.43,4.79,8.33,426.09,29.53,118.81,21.35]
# mae_2 = [0.055,0.074,0.093,0.101,0.076,0.091,0.061,0.109,0.055,0.063,0.049,0.095,0.091,0.088,0.127,0.056]
# t60_2 = [64.99,57.65,75.13,18.21,9.37,8.30,160.76,263.37,6.78,82.26,5.92,57.14,30.41,30.36,252.23,19.64]