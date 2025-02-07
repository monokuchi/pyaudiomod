
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from time_stretching import OLA, WSOLA



sample_rate, audio_data = wavfile.read("./audio_samples/C_Chord.wav")
length = audio_data.shape[0] / sample_rate
print(f"Number of Channels: {audio_data.shape[1]}")
print(f"Sample Rate: {sample_rate}")
print(f"Length: {length}s")

# Take just the left channel for testing
original_signal = audio_data[:, 1]



ola = OLA(frame_size=256, speed_factor=2)
output_signal = ola.run(original_signal)

# wsola = WSOLA(speed_factor=2.0)
# output_signal = wsola.run(original_signal)


print("Original Signal Length:", original_signal.shape[0])
print("TSM Signal Length:", output_signal.shape[0])
print("Check Speed Factor:", original_signal.shape[0]/output_signal.shape[0])


# Generate audio file from output signal
wavfile.write("./audio_samples/C_Chord_OLA.wav", sample_rate, output_signal.astype(np.float64))


plt.figure(0)
plt.plot(original_signal, "r--", label="Original")
plt.plot(output_signal, "b--", label="TSM")
plt.legend()
plt.show()
