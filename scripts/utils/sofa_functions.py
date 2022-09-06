import os
import numpy as np
import soundfile as sf
from pysofaconventions import *
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
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

    import os
    import numpy as np
    import soundfile as sf
    from pysofaconventions import *
    import matplotlib.pyplot as plt
    import scipy.io
    import scipy.signal
    import warnings

    plt.switch_backend('TkAgg')
    plt.switch_backend('agg')

    mat = scipy.io.loadmat('audio/input/chosen_rirs/HOA/TUT/mat/Tietotalo_RIR.mat')
    tut_path = 'audio/input/chosen_rirs/HOA/TUT/e32/'

    rir_db = mat['rir_DB']

    n_dist = rir_db.shape[0]
    n_elev = rir_db.shape[1]
    n_azim = rir_db.shape[3]
    n_block = rir_db.shape[4]
    n_channel = rir_db.shape[5]

    e32 = np.array([])
    for channel in range(n_channel):
        ifft = np.array([])
        for i in range(1025):
            for block in range(n_block):
                if block == 0:
                    bins = np.fft.ifft([rir_db[0][0][i][block][0][channel]]).real
                else:
                    bins = np.concatenate((bins, np.fft.ifft([rir_db[0][0][i][block][0][channel]]).real))

            if i == 0:
                ifft = bins
            else:
                ifft = np.concatenate((ifft, bins), axis=0)

        if channel == 0:
            e32 = [ifft]
        else:
            e32 = np.concatenate((e32, [ifft]))

    sf.write(tut_path + 'c.wav', e32.T, 48000)

    for distance in range(n_dist):
        for elevation in range(n_elev):
            for azimuth in range(n_azim):
                e32 = np.array([])
                for channel in range(n_channel):
                    for block in range(n_block):
                        if block == 0:
                            ifft = np.fft.ifft([rir_db[0][0][:].T[0][block][0]]).real
                        else:
                            ifft = np.concatenate((ifft, np.fft.ifft([rir_db[0][0][:].T[0][block][0]]).real))

                    if channel == 0:
                        e32 = [ifft]
                    else:
                        e32 = np.concatenate((e32, [ifft]))

                sf.write(tut_path + 'b.wav', e32[0].T, 48000)
                sf.write(tut_path + str(distance) + str(elevation) + str(azimuth) + '.wav', e32.T, 48000)