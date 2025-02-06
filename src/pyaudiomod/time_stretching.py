
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from utils import FrameShiftBoundaries, hann_window



class TSM(ABC):
    """
    Abstract Base Class (ABC) for Time Scale Modification (TSM) algorithms
    """

    def __init__(self,
                 frame_size: int=256,
                 speed_factor: float=1.0,
                 synthesis_hopsize: int=None,
                 analysis_hopsize: int=None) -> None:
        
        self.frame_size: int = int(frame_size) # Should be a power of 2
        self.speed_factor: float = float(speed_factor) # speed_factor<1 for slower speed, speed_factor=1 for original speed, speed_factor>1 for faster speed
        self.synthesis_hopsize: int = int(self.frame_size // 4) if synthesis_hopsize is None else int(synthesis_hopsize)
        if not 10 <= self.synthesis_hopsize <= self.frame_size:
            # TODO: Set appropiate bounds for synthesis hopsize so analysis_hopsize does not round to 0
            raise ValueError(f"Input Synthesis Hopsize: {self.synthesis_hopsize}, must be within [10:{self.frame_size}]")
        self.analysis_hopsize: int = int(self.synthesis_hopsize * self.speed_factor) if analysis_hopsize is None else int(analysis_hopsize)

    @abstractmethod
    def split_into_frames(self) -> npt.NDArray:
        ...

    @abstractmethod
    def reconstruct_from_frames(self) -> npt.NDArray:
        ...

    @abstractmethod
    def run(self) -> npt.NDArray:
        ...
        



class OLA(TSM):
    def __init__(self, 
                 frame_size: int=256, # TODO: Make the frame_size equivalent to around ~10 ms of the input audio to reduce the transient doubling effect
                 speed_factor: float=1.0, 
                 synthesis_hopsize: int=None, 
                 analysis_hopsize: int=None,
                 synthesis_window: npt.NDArray=None,
                 analysis_window: npt.NDArray=None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)

        self.synthesis_window = hann_window(frame_size, False) if synthesis_window is None else synthesis_window
        self.analysis_window = hann_window(frame_size, False) if analysis_window is None else analysis_window
        

    def split_into_frames(self, signal: npt.NDArray, frame_size: int, hopsize: int) -> npt.NDArray:
        """
        Splits signal into multiple frames, each having a fixed frame_size and spaced apart by hopsize
        """
        signal_length = signal.shape[0]
        signal_length_without_last_frame = signal_length - frame_size

        # First check if signal_length_without_last_frame is a multiple of hopsize and pad accordingly if not
        if signal_length_without_last_frame % hopsize != 0:
            # signal_length_without_last_frame is not a multiple of hopsize, pad (frame_size-residual) amount of zeros until it is
            if hopsize < frame_size:
                residual = signal_length - (hopsize*((signal_length_without_last_frame // hopsize) + 1))
            elif hopsize > frame_size:
                residual = signal_length % hopsize
            elif hopsize == frame_size:
                residual = frame_size-(signal_length % hopsize)

            zeros = np.zeros(frame_size - residual)
            signal = np.concatenate((signal, zeros))

            # print("Number of Zero Padding:", zeros.shape[0])

        # Extract the frames from the signal, should be of dimension (num_frames_extracted x frame_size)
        frames = np.lib.stride_tricks.sliding_window_view(signal, window_shape=frame_size)[::hopsize]
        # print("(num_frames_extracted x frame_size):", frames.shape)

        return frames 
    

    def reconstruct_from_frames(self, frames: npt.NDArray, hopsize: int) -> npt.NDArray:
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


    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Overlap-Add algorithm
        This method is suitable for percussive and transient sounds
        """
        # Split the input signal into seperate analysis frames
        analysis_frames = self.split_into_frames(signal, self.frame_size, self.analysis_hopsize)

        # Generate the synthesis_frames
        synthesis_frames = analysis_frames * self.synthesis_window

        # Reconstruct the output signal from the synthesis frames
        output_signal = self.reconstruct_from_frames(synthesis_frames, self.synthesis_hopsize)

        # We need to normalize the output signal by the sum of the overlapped window functions
        # so we don't get any amplitude fluctuations caused by the overlapping and adding
        if self.synthesis_hopsize == int(self.frame_size // 2):
            # In this case the windows are spaced exactly 50% of frame size apart so they overlap perfectly to sum to a constant 1.0
            # This is called COLA (Constant Overlap-Add Constraint)
            overlapping_windows = 1.0
        else:
            # Stack the synthesis window vertically by the number of analysis frames
            stacked_synthesis_window = np.tile(self.synthesis_window, (analysis_frames.shape[0], 1))
            # Construct the overlapping window signal from the vertically stacked synthesis windows
            overlapping_windows = self.reconstruct_from_frames(stacked_synthesis_window, self.synthesis_hopsize)

        # Replace any zeros with 1.0 to avoid divide by zeros
        overlapping_windows = np.where(overlapping_windows == 0, 1.0, overlapping_windows)

        # Element wise divide by overlapping_windows in order to normalize the output signal
        output_signal /= overlapping_windows

        # Reconstruct our signal by using the synthesis_frames
        return output_signal
    
    

class WSOLA(TSM):
    def __init__(self, 
                 frame_size: int=256, 
                 speed_factor: float=1.0, 
                 synthesis_hopsize: int=None, 
                 analysis_hopsize: int=None,
                 synthesis_window: npt.NDArray=None,
                 analysis_window: npt.NDArray=None,
                 frame_shift_boundaries: type[FrameShiftBoundaries]=None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)

        self.synthesis_window = hann_window(frame_size, False) if synthesis_window is None else synthesis_window
        self.analysis_window = hann_window(frame_size, False) if analysis_window is None else analysis_window
        self.frame_shift_boundaries = FrameShiftBoundaries() if frame_shift_boundaries is None else frame_shift_boundaries


    def split_into_frames(self,
                          signal: npt.NDArray, 
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
    

    def reconstruct_from_frames(self, frames: npt.NDArray, hopsize: int) -> npt.NDArray:
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


    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Waveform Similarity Overlap-Add algorithm
        This method is an improvement on the OLA algorithm by attempting to address 
        the lack of signal sensitivity
        """

        analysis_frames = self.split_into_frames(signal, self.frame_size, self.synthesis_hopsize, self.analysis_hopsize, self.frame_shift_boundaries)


        # We need to normalize our windowed frames by the sum of the overlapped window functions
        # so we don't get any amplitude fluctuations caused by the overlapping and adding
        if self.synthesis_hopsize == int(self.frame_size // 2):
            normalization = 1.0
        else:
            normalization = self.synthesis_window.sum()

        # Generate our synthesis_frames
        synthesis_frames = (analysis_frames * self.synthesis_window) / normalization

        # Reconstruct our signal by using the synthesis_frames
        return self.reconstruct_from_frames(synthesis_frames, self.synthesis_hopsize)
    

class PV(TSM):
    def __init__(self, 
                 frame_size: int=256, 
                 speed_factor: float=1.0, 
                 synthesis_hopsize: int=None, 
                 analysis_hopsize: int=None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)


    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Phase Vocoder algorithm
        This method is suitable for harmonic sounds
        """
        return signal
    

class HPS(TSM):
    def __init__(self, 
                 frame_size: int=256, 
                 speed_factor: float=1.0, 
                 synthesis_hopsize: int=None, 
                 analysis_hopsize: int=None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)


    def _hps(self, signal: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Helper function for the Harmonic-Percussive Seperation algorithm
        Seperates out the harmonic and percussive components of a given signal
        """
        harmonic = signal
        percussive = signal

        return harmonic, percussive
    

    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Harmonic-Percussive Seperation algorithm
        """
        harmonic_component, percussive_component = self._hps(signal)

        time_stretched_harmonic_component: npt.NDArray = PV.run(harmonic_component)
        time_stretched_percussive_component: npt.NDArray = OLA.run(percussive_component)
        
        return (time_stretched_harmonic_component + time_stretched_percussive_component)
