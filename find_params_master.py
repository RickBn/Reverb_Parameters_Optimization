from scripts.reverb_parameters_optimize import *
import datetime

plt.switch_backend('agg')

if __name__ == "__main__":

    start = datetime.datetime.now()

    # Whether match only the late reverberation or the entire RIR
    match_only_late = True

    # Whether to apply the dimensionality reduction to the walls coefficients
    apply_dim_red = True

    # Whether interpolate to return to the original space. Used only when apply_dim_red = True
    inv_interp = True

    # Whether to force the points in the unit circle. Used only when apply_dim_red = True and inv_interp = True
    unit_circle = False

    # Whether the optmizator works on polar coordinates instead of cartesian ones. Used only when apply_dim_red = True and n_dims_red = 2
    polar_coords = False

    # Whether all the walls have the same absorption coefficients
    same_coef_walls = True

    # Whether to set the absorption coef of the last 2 bands equal to the third to last
    force_last2_bands_equal = True

    # Number of iterations of gp_minimize
    n_iterations = 500

    # Number of initial points used by gp_minimize
    n_initial_points = 10

    # Sample length of the fade
    fade_length = 256

    rir_names = ['SDN037']

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
                    fade_length=fade_length)

    stop = datetime.datetime.now()

    elapsed = stop - start

    print(f'Total elapsed time: {elapsed}')

    pass
