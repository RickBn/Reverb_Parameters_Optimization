import pedalboard as pb
import numpy as np
import matplotlib.pyplot as plt

sr = 48000
duration = 5
y = np.zeros((2, sr * duration))
y[0][0] = 1
y[1][0] = 1

sdn = pb.load_plugin('vst3/Real time SDN.vst3')

sdn.output_mode = 'Stereo'

effected = sdn(y, sample_rate=sr)

print(effected)

time = np.linspace(0, duration, num=sr * duration)

plt.figure(figsize=(15, 5))
plt.plot(time, effected[1])
plt.title('Audio Plot')
plt.ylabel(' signal wave')
plt.xlabel('time (s)')
plt.show()

pass