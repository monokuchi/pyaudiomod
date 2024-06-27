
import numpy as np
import matplotlib.pyplot as plt

from time_stretching import OLA, WSOLA, FrameShiftBoundaries



time_samples = np.arange(0, 5, 0.00001)
original_signal = np.sin(2 * np.pi * time_samples)

# imp = signal.unit_impulse(100, 'mid')
# b, a = signal.butter(4, 0.2)
# original_signal = signal.lfilter(b, a, imp)




wsola = WSOLA(speed_factor=2.0)
output_signal = wsola.run(original_signal)


print(original_signal.shape[0])
print(output_signal.shape[0])
print("Check Speed Factor:", original_signal.shape[0]/output_signal.shape[0])

# num_figure = 1
# for frame in analysis_frames:
#     plt.figure(num_figure)
#     num_figure += 1
#     plt.plot(frame)
# plt.show()

plt.figure(0)
plt.plot(original_signal, "r--")
plt.plot(output_signal, "b--")
plt.show()
