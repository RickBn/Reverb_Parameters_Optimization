import os
import numpy as np
import soundfile as sf
from pysofaconventions import *
import matplotlib.pyplot as plt
import warnings

plt.switch_backend('TkAgg')


def sofa_fir_to_wav(sofa_path: str, verbose: bool = False, save_path=None):
    conventionFile = SOFAGeneralFIR(sofa_path, 'r')

    # Check validity
    if not conventionFile.isValid():
        raise Exception(sofa_path + ' is _NOT_ a valid SOFA file')

    files = conventionFile.getDataIR()

    listenerPosUnits, listenerPosCoordinates = conventionFile.getListenerPositionInfo()
    listenerPos = conventionFile.getListenerPositionValues()

    sourcePosUnits, sourcePosCoordinates = conventionFile.getSourcePositionInfo()
    sourcePos = conventionFile.getSourcePositionValues()

    if verbose:
        conventionFile.printSOFAGlobalAttributes()
        print("- ListenerPosition:Type = " + listenerPosCoordinates)
        print("- ListenerPosition:Units = " + listenerPosUnits)
        print("- ListenerPosition = " + str(listenerPos))
        print("- SourcePosition:Type = " + sourcePosCoordinates)
        print("- SourcePosition:Units = " + sourcePosUnits)
        print("- SourcePosition = " + str(sourcePos))

    for idx, file in enumerate(files):
        if listenerPos.shape != sourcePos.shape:
            raise Exception('Shape mismatch between listener position and source position!')

        current_sp = listenerPos[idx]
        current_lp = listenerPos[idx]
        file_name = 'S' + str(idx % 3) + '_R' + str(idx // 3) + '.wav'
        #file_name = "_".join(str(current_sp).split()) + "_".join(str(current_lp).split()) + '.wav'

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            sf.write(save_path + file_name, file.T, int(conventionFile.getSamplingRate()[0]))


if __name__ == "__main__":
    sofa_path = 'audio/input/chosen_rirs/HOA/SRIRs/B_format_4th/sofa/6DoF_SRIRs_eigenmike_SH_0percent_absorbers_enabled.sofa'
    save_path = 'audio/input/chosen_rirs/HOA/SRIRs/B_format_4th/'

    sofa_fir_to_wav(sofa_path=sofa_path, verbose=False, save_path=save_path)