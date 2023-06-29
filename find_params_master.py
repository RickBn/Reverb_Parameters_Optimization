from scripts.reverb_parameters_optimize import *

plt.switch_backend('agg')

if __name__ == "__main__":

    # Whether match only the late reverberation or the entire RIR. Set to False for SDN
    match_only_late = False

    # Set the name of the room:
    # - 'Living Room'
    # - 'MARCo'
    # - 'METu'
    rir_name = 'SDN000'
    folder = f'stereo/{rir_name}/'

    # Set the path of the reverberator (vst3):
    # - 'vst3/Real time SDN.vst3'
    # - 'vst3/FdnReverb.vst3'
    vst_path = "vst3/Real time SDN.vst3"
    vst_name = 'SDN'

    n_iterations = 25

    rir_path = f'audio/input/chosen_rirs/{folder}_todo/'
    er_path = f'audio/trimmed_rirs/{folder}'
    armodel_path = f'audio/armodels/{folder}'
    merged_rir_path = f'audio/merged_rirs/{folder}'
    vst_rir_path = f'audio/vst_rirs/{folder}'
    params_path = f'audio/params/{folder}'
    result_path = f'audio/results/{folder}'
    input_path = f'audio/input/sounds/48/speech/_trimmed/loudnorm/_todo/'
    fixed_params_path = f'fixed_parameters/{vst_name}/{rir_name}.yml'

    find_params(rir_path,
                er_path,
                armodel_path,
                merged_rir_path,
                vst_rir_path,
                params_path,
                result_path,
                input_path,
                fixed_params_path=fixed_params_path,
                generate_references=False,
                original_er=False,
                pre_norm=False,
                vst_path=vst_path,
                n_iterations=n_iterations,
                match_only_late=match_only_late)

    pass
