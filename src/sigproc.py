
import numpy as np
import numpy.typing as npt

from time_stretching import FrameShiftBoundaries



def split_into_frames(signal: npt.NDArray, 
                      frame_size: int, 
                      synthesis_hopsize: int, 
                      analysis_hopsize: int, 
                      frame_shift_boundaries: type[FrameShiftBoundaries]) -> npt.NDArray:
    """
    Splits signal into multiple frames, each having a fixed frame_size and spaced apart by hopsize
    frame can be shifted by some amount of samples within frame_shift_boundaries (negative value indicates a backwards shift)
    """
    signal_length = signal.shape[0]
    signal_length_without_last_frame = signal_length - frame_size

    # num_frames = 100000000
    # frames = np.empty([num_frames, frame_size])
    frames = []

    
    end_of_signal = False

    adjusted_analysis_frame_start_idx = 0
    extended_frame_region_start_idx = analysis_hopsize
    optimal_shift_idx = 0

    while not end_of_signal:
        adjusted_analysis_frame = signal[adjusted_analysis_frame_start_idx:adjusted_analysis_frame_start_idx+frame_size]
        # Insert our adjusted_analysis_frame into the frames list
        frames.append(adjusted_analysis_frame)
        natural_progression = signal[adjusted_analysis_frame_start_idx+synthesis_hopsize:adjusted_analysis_frame_start_idx+synthesis_hopsize+frame_size]
        extended_frame_region = signal[extended_frame_region_start_idx+frame_shift_boundaries.min_shift:extended_frame_region_start_idx+frame_size+frame_shift_boundaries.max_shift]
        optimal_shift_idx = np.argmax(np.correlate(natural_progression, extended_frame_region))

        # Check if we reached end of signal
        if signal_length < 1000000:
            end_of_signal = True


        # Update our start indices
        adjusted_analysis_frame_start_idx = (len(frames)*analysis_hopsize) + optimal_shift_idx
        extended_frame_region_start_idx = (len(frames)+1) * analysis_hopsize



    return np.array(frames)

# def split_into_frames(signal: npt.NDArray, frame_size: int, hopsize: int) -> npt.NDArray:
#     """
#     Splits signal into multiple frames, each having a fixed frame_size and spaced apart by hopsize
#     """
#     signal_length = signal.shape[0]
#     signal_length_without_last_frame = signal_length - frame_size
    
#     # First check if signal_length_without_last_frame is a multiple of hopsize and pad accordingly if not
#     if signal_length_without_last_frame % hopsize != 0:
#         # signal_length_without_last_frame is not a multiple of hopsize, pad zeros until it is
#         residual = signal_length - (hopsize*((signal_length_without_last_frame // hopsize) + 1))
#         zeros = np.zeros(frame_size - residual)
#         signal = np.concatenate((signal, zeros))
#         # Update signal_length
#         signal_length = signal.shape[0]

#     # Extract the frames from the signal
#     # Should be of dimension (num_frames x frame_size)
#     frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hopsize]

#     return frames
    
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
