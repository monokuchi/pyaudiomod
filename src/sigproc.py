
import numpy as np
import numpy.typing as npt



def split_into_frames(signal: npt.NDArray, frame_size: int, hopsize: int) -> npt.NDArray:
    """
    Splits signal into multiple frames, each having a fixed frame_size and spaced apart by hopsize
    """
    signal_length = signal.shape[0]
    signal_length_without_last_frame = signal_length - frame_size
    
    # First check if signal_length_without_last_frame is a multiple of hopsize and pad accordingly if not
    if signal_length_without_last_frame % hopsize != 0:
        # signal_length_without_last_frame is not a multiple of hopsize, pad zeros until it is
        residual = signal_length - (hopsize*((signal_length_without_last_frame // hopsize) + 1))
        zeros = np.zeros(frame_size - residual)
        signal = np.concatenate((signal, zeros))
        # Update signal_length
        signal_length = signal.shape[0]

    # Extract the frames from the signal
    # Should be of dimension (num_frames x frame_size)
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hopsize]

    return frames
    
def reconstruct_from_frames(frames: npt.NDArray, hopsize: int) -> npt.NDArray:
    """
    Constructs a signal by iteratively combining frames that are spaced apart by hopsize
    frames is a numpy array with dimensions representing ("Number of Frames" x "Frame Size")
    """

    num_frames = frames.shape[0]
    frame_size = frames.shape[1]

    signal = np.zeros(((num_frames-1)*hopsize) + frame_size)

    for frame_idx in range(num_frames):
        start_idx = frame_idx * hopsize
        end_idx = start_idx + frame_size
        signal[start_idx:end_idx] += frames[frame_idx]

    return signal






def hann_window(window_length: int=64, symmetric_flag: bool=True) -> npt.NDArray:
    """
    Implements a Hann (Hanning) window similar to the corresponding matlab function
    https://www.mathworks.com/help/signal/ref/hann.html

    Note: Returned window will be causal instead of centered around zero
    """

    hann = np.empty(window_length)

    if symmetric_flag:
        # Symmetric window 
        # Generates points that are in [0, window_length-1]
        numerator = np.linspace(0, window_length-1, window_length)
        denominator = window_length - 1
    else:
        # Periodic window
        # The left zero endpoint is included in the window, while the one on the right lies one sample outside to the right
        # Generates points that are in [0, window_length-1]
        numerator = np.linspace(0, window_length-1, window_length)
        denominator = window_length

    hann = 0.5 * (1 - np.cos(((2*np.pi)*numerator) / denominator))

    return hann
