
import numpy as np
import matplotlib.pyplot as plt

from time_stretching import OLA, WSOLA



time_samples = np.arange(0, 3.53, 1e-5)
original_signal = np.sin(2 * np.pi * time_samples)



ola = OLA(frame_size=64, speed_factor=2)
output_signal = ola.run(original_signal)

# wsola = WSOLA(speed_factor=2.0)
# output_signal = wsola.run(original_signal)


print("Original Signal Length:", original_signal.shape[0])
print("TSM Signal Length:", output_signal.shape[0])
print("Check Speed Factor:", original_signal.shape[0]/output_signal.shape[0])


plt.figure(0)
plt.plot(original_signal, "r--", label="Original")
plt.plot(output_signal, "b--", label="TSM")
plt.legend()
plt.show()
