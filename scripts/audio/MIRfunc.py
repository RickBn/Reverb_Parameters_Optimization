from __future__ import division
from scripts.audio.DSPfunc import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''

MIR Functions

Should work on both Python 2.7.x and 3.x
Look at the test() function for some examples...

This collection is based upon the DSPfunc collection and uses the following packages:
  - numpy
  - scipy
  - matplotlib (for visualization)

'''

__version__ = "0.01"
__author__  = "G.Presti"


'''
------------------------
'''

def f2mel(f):
    return 2595.0 * np.log10(1+f/700.0)

def mel2f(mel):
    return 700.0 * ( ( 10.0**(np.asarray(mel)/2595) ) - 1 )

def f2bark(f):
    return 6 * np.arcsinh(f/600)

def bark2f(brk):
    return 600 * np.sinh(brk/6)

def envelope(x, method='rms', param=512):
    '''Extract envelope using 'rms','smooth','peak' or 'hilbert'''
    def pk(x,r):
        y = np.zeros(x.shape)
        rel = np.exp(-1/r)
        y[0] = np.abs(x[0])
        for n,e in enumerate(np.abs(x[1:])):
            y[n+1] = np.maximum(e,y[n]*rel)
        return y

    def rms(x,w,mode):
        return np.sqrt(np.convolve(np.power(x,2), w/np.sum(w), mode)) # Not very efficient...
            
    m = {
        'rms': lambda x,param: rms(x,np.ones(param),'full')[:1-param],
        'smooth': lambda x,param: rms(x,np.hanning(param),'same'),
        'peak': pk,
        'hilbert': lambda x,param: np.abs(sp.signal.hilbert(x))
    }.get(method.lower())
    return m(x,param)


def brightness(x, Fs = 1, envType = 'rms', envParam = 600):
    '''Calculates Equivalent Brightness Frequency (CoBE algo.)'''
    envFun = lambda x: envelope(x, envType, envParam)
    ex = envFun(x)
    dx = np.hstack((0,np.diff(x)))
    edx = envFun(dx)
    silence = ex <= eps
    ex[silence] = 1
    b = edx / ex
    b[silence] = 0
    return (float(Fs)/np.pi)*np.arcsin(b/2)

def getFilterbank(F, bw = 100, scale = 'mel', wfunc = np.bartlett):
    '''Return a matrix of overlapping masks of size bw*2+1 and center freq. array'''
    '''Masks have wfunc shape and are equally spaced in scale-space'''
    '''scale can be: 'mel', 'bark', 'st' '''
    findx = lambda array,x: (np.abs(array-x)).argmin()
    f2x, x2f = {
        'mel': (f2mel,mel2f),
        'bark': (f2bark,bark2f),
        'st': (linf2logf,logf2linf)
    }.get(scale.lower())
    mstrt = f2x(F[0]) if F[0]>0 else f2x(F[1])
    mstop = f2x(F[-1])
    nband = int((mstop-mstrt)//bw - 1)
    cnt = np.linspace(1,nband,nband)*bw
    low = x2f(cnt-bw)
    hig = x2f(cnt+bw)
    # cnt = x2f(cnt)
    fbank = np.zeros((len(F),nband))
    for b in range(nband):
        il = findx(F,low[b])
        ih = findx(F,hig[b])
        fbank[il:ih,b] = wfunc(ih-il)
    return fbank, cnt

def applyFilterbank(psd,fbank):
    '''given n filters in fbank, reduces psd to n lines'''
    fltc = lambda ps,fb: [ np.sum(ps*fb[:,f]) for f in range(fb.shape[1]) ]
    return np.asarray([ fltc(psd[:,c],fbank) for c in range(psd.shape[1]) ]).T

def mfcc(stft, F, n = -1, melbw = 100, upsample = None):
    '''Computes MFCC from STFT input'''
    '''Returns MFCC indexes in range [0:n]'''
    x = np.abs(stft)
    scale = 1
    if upsample is not None:
        nF = np.linspace(F[0],F[-1],upsample*x.shape[0]-1)
        x, F = (interpc(nF, F, x), nF)
        scale = upsample
    fbank = getFilterbank(F,melbw,'mel',np.bartlett)[0]
    psd = np.power(x,2) #/scale ?
    mpsd = applyFilterbank(psd,fbank)
    mpsd = amp2db(mpsd,-96)
    mfc = sp.fftpack.dct(mpsd, axis=0, norm='ortho')
    return mfc[0:n,:]

def imfcc(mfc, F):
    '''Inverse MFCC (creates a mask for a stft matrix)'''
##    if sigPow is None                   ## MFCC are returned WITH MFCC[0] coeff!
##        sigPow = np.ones(mfc.shape[1])
##    mpsd = sp.fftpack.idct(np.vstack((amp2db(sigPow),mfc)), axis=0, norm='ortho')
    mpsd = sp.fftpack.idct(mfc, axis=0, norm='ortho')
    mpsd = db2amp(mpsd)
    n = mpsd.shape[0]
    psd = interpc(F,mel2f(np.linspace(f2mel(F[0]),f2mel(F[-1]),n)),mpsd)
    return np.sqrt(psd)

'''
------------------------
'''

def test( filename, frames=-1, start=0 ):
    '''Test function'''

    '''Load sample file (convert to mono if necessary)'''
    x, Fs = audioread(filename, frames, start)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x/peak(x)
    
    '''Setup buffering variables and windowing function'''
    l = x.size
    fsiz = 1024
    hsiz = fsiz // 4
    win = np.hanning(fsiz)

    '''Split input into cunks (columns) and window them'''
    buf = buffer(x,fsiz,hsiz)
    buf = buf * win[:,None]

    '''Get the FT of the input chunks (with time and freq axes)'''
    X, F, T = stft(buf,hsiz,Fs)



    '''Start timing...'''
    start = time.clock()

    '''Extract brightness'''
    EBF = brightness(x,Fs,'smooth',800)

    '''MFCC'''
    nmc = -1
    melbw = 100
    upsample = 2
    MFCC = mfcc(X,F,nmc,melbw,upsample)

    '''Inverse MFCC'''
    iMFCC = imfcc(MFCC,F)

    '''Mess with MFCCs (reverse MFCC in time)'''
    Y = X/iMFCC
    Y = Y * np.fliplr(iMFCC)

    '''Check how long it took'''
    sec = time.clock() - start
    print('Time: {} sec'.format(sec))
    print('MFCC shape: {}'.format(MFCC.shape))

    '''Save output'''
    y = istft(Y)
    y = y * rms(buf)/(rms(y)+eps)
    y = y * win[:,None]
    y = unbuffer(y, hsiz, w=win**2, ln=l)
    y[0:hsiz] = 0
    audiowrite('Alien_Voice.wav',y/peak(y),Fs,subtype='PCM_24')

    


    '''Now let's plot some data...'''
    f, axarr = plt.subplots(3, sharex=True)
    Tx = np.linspace(0,(l-1)/Fs,l)

    '''Resamples the STFT in log scale (mels)'''
    nA = mel2f(np.linspace(30,f2mel(4*Fs/10),1024))
    lX = interpc(nA,F,amp2db(np.abs(X),-40),method='slinear')

    '''Plot spectrum in mel-scale with EBF ontop of it'''
    axarr[0].imshow(lX,cmap="inferno",interpolation="nearest",origin="lower",aspect="auto",extent=[T[0],T[-1],f2mel(nA[0]),f2mel(nA[-1])])
    axarr[0].plot(Tx,f2mel(EBF))
    axarr[0].title.set_text ('Spectrum (mel scale) with EBF')
    axarr[0].set_ylabel ('Mels')
    axarr[0].grid('on')

    '''Plot mel filterbank output'''
    fbank = getFilterbank(F,100,'mel',np.bartlett)[0]
    mfpsd = amp2db( applyFilterbank( np.power(np.abs(X),2) ,fbank ) ,-48)
    axarr[1].imshow(mfpsd,cmap="inferno",interpolation="nearest",origin="lower",aspect="auto",extent=[T[0],T[-1],f2mel(nA[0]),f2mel(nA[-1])])
    axarr[1].title.set_text ('Filterbank output')
    axarr[1].set_ylabel ('Mels')
    
    '''Plot MFCC'''
    axarr[2].imshow(MFCC,cmap="inferno",interpolation="nearest",origin="lower",aspect="auto",extent=[T[0],T[-1],1,nmc])
    axarr[2].title.set_text ('MFCC')
    axarr[2].set_ylabel ('MFCC coeff.')
    axarr[2].set_xlabel ('Time (sec)')
    axarr[2].grid('on')
    
##    '''Plot waveform and envelope'''
##    axarr[2].plot(Tx,x,'0.75')
##    axarr[2].plot(Tx,env)
##    axarr[2].plot(Tx,EBF)
##    axarr[2].grid('on')

##    '''Compare frequency scales'''
##    hz = np.linspace(10,20000,200)
##    mel = f2mel(hz)
##    brk = f2bark(hz)
##    st = linf2logf(hz)

##    f3 = plt.figure()
##    ax3 = f3.add_subplot(111)
##    ax3.plot(F,fb)
##    p1, = ax3.plot(hz/hz.max(),label='Hz')
##    p2, = ax3.plot(mel/mel.max(),label='Mel')
##    p3, = ax3.plot(brk/brk.max(),label='Bark')
##    p4, = ax3.plot(st/st.max(),label='ST')
##    ax3.legend(handles=[p1, p2, p3, p4])
##    ax3.grid('on')
    
    plt.show()

    

if __name__ == '__main__':
    '''Run the test case scenario'''
    '''First, select a file to analize'''
    testfile = './Samples/voice.wav'
    '''Run test'''
    test(testfile)
