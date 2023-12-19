from scripts.reverb_parameters_optimize import *
import datetime

plt.switch_backend('agg')

if __name__ == "__main__":

    start = datetime.datetime.now()

    #  gp_minimize, forest_minimize, gbrt_minimize
    optimizer = 'forest_minimize'
    # Controls how much of the variance in the predicted values should be taken into account. If set to be very high,
    # then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
    # Non sembra influire
    optimizer_kappa = 1.96#1.96
    # Controls how much improvement one wants over the previous best values. Used when the acquisition is either "EI" or
    # "PI".
    optimizer_xi = 0.01,

    # Whether match only the late reverberation or the entire RIR
    match_only_late = False

    # Whether to apply the dimensionality reduction to the walls coefficients. Path for pre-computed values
    apply_dim_red = {'pts_2d': r'.\wall_coeff_dim_reduction\PCA_data\20000_iterations\2d_projection_data.csv',
                     'pts_original': r'.\wall_coeff_dim_reduction\PCA_data\20000_iterations\filters_data.csv'}

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
    n_iterations = 11

    # Number of initial points used by gp_minimize
    n_initial_points = 10

    # Sample length of the fade
    fade_length = 256

    # Whether to remove direct. Set to False because the direct is already removed when SDN generates the RIR with the parameter line_of_sight
    remove_direct = False

    # Number of cores to run in parallel. If n_jobs=-1, then number of jobs is set to number of cores.
    n_jobs = 1

    rir_names = ['SDN044']

    # Set the path of the reverberator (vst3):
    # - 'vst3/Real time SDN.vst3'
    # - 'vst3/FdnReverb.vst3'
    vst_path = "vst3/Real time SDN.vst3"
    vst_name = 'SDN'

    for rir_name in rir_names:
        print(f'FITTING ENVIRONMENT: {rir_name}')

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
        params_path = f'audio/params/{folder}'
        original_params_path=f'audio/input/chosen_rirs/{folder}/parameters.yml'
        result_path = f'audio/results/{folder}'
        input_path = f'audio/input/sounds/48/speech/_trimmed/loudnorm/_todo/'
        fixed_params_path = f'fixed_parameters/{vst_name}/{rir_name}.yml'

        find_params(rir_path,
                    er_path,
                    armodel_path,
                    merged_rir_path,
                    vst_rir_path,
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
                    n_jobs=n_jobs)

    stop = datetime.datetime.now()

    elapsed = stop - start

    print(f'Total elapsed time: {elapsed}')

    pass
