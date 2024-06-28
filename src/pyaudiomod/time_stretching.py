
import numpy as np
import numpy.typing as npt

from sigproc import hann_window, split_into_frames, reconstruct_from_frames
from custom import FrameShiftBoundaries



class TSM:
    """
    Base Class for Time Scale Modification (TSM) techniques
    """

    def __init__(self,
                 frame_size: int = 256,
                 speed_factor: float = 1.0,
                 synthesis_hopsize: int = None,
                 analysis_hopsize: int = None) -> None:
        
        self.frame_size: int = frame_size # Should be a power of 2
        self.speed_factor: float = speed_factor # speed_factor<1 for slower speed, speed_factor=1 for original speed, speed_factor>1 for faster speed
        self.synthesis_hopsize: int = int(self.frame_size // 4) if synthesis_hopsize is None else synthesis_hopsize
        self.analysis_hopsize: int = int(self.synthesis_hopsize * self.speed_factor) if analysis_hopsize is None else analysis_hopsize
        


        

class OLA(TSM):
    def __init__(self, 
                 frame_size: int = 256, # TODO: Make the frame_size equivalent to around ~10 ms of the input audio to reduce the transient doubling effect
                 speed_factor: float = 1, 
                 synthesis_hopsize: int = None, 
                 analysis_hopsize: int = None,
                 synthesis_window: npt.NDArray = None,
                 analysis_window: npt.NDArray = None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)

        self.synthesis_window = hann_window(frame_size, False) if synthesis_window is None else synthesis_window
        self.analysis_window = hann_window(frame_size, False) if analysis_window is None else analysis_window
        




    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Overlap-Add algorithm
        This method is suitable for percussive and transient sounds
        """

        analysis_frames = split_into_frames(signal, self.frame_size, self.analysis_hopsize)


        # We need to normalize our windowed frames by the sum of the overlapped window functions
        # so we don't get any amplitude fluctuations caused by the overlapping and adding
        if self.synthesis_hopsize == int(self.frame_size // 2):
            # In this case the windows are spaced exactly 50% of frame size apart so they overlap perfectly to sum to a constant 1.0
            # This is called the COLA constraint (Constant Overlap-Add Constraint)
            normalization = 1.0
        else:
            normalization = self.synthesis_window[0:self.analysis_hopsize].sum() * self.frame_size

        # Generate our synthesis_frames
        print(normalization)
        # Need element wise division by normalization?
        synthesis_frames = (analysis_frames * self.synthesis_window) / normalization

        # Reconstruct our signal by using the synthesis_frames
        return reconstruct_from_frames(synthesis_frames, self.synthesis_hopsize)
    

class WSOLA(TSM):
    def __init__(self, 
                 frame_size: int = 256, 
                 speed_factor: float = 1, 
                 synthesis_hopsize: int = None, 
                 analysis_hopsize: int = None,
                 synthesis_window: npt.NDArray = None,
                 analysis_window: npt.NDArray = None,
                 frame_shift_boundaries: type[FrameShiftBoundaries] = None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)

        self.synthesis_window = hann_window(frame_size, False) if synthesis_window is None else synthesis_window
        self.analysis_window = hann_window(frame_size, False) if analysis_window is None else analysis_window
        self.frame_shift_boundaries = FrameShiftBoundaries() if frame_shift_boundaries is None else frame_shift_boundaries



    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Waveform Similarity Overlap-Add algorithm
        This method is an improvement on the OLA algorithm by attempting to address 
        the lack of signal sensitivity
        """

        analysis_frames = split_into_frames(signal, self.frame_size, self.synthesis_hopsize, self.analysis_hopsize, self.frame_shift_boundaries)


        # We need to normalize our windowed frames by the sum of the overlapped window functions
        # so we don't get any amplitude fluctuations caused by the overlapping and adding
        if self.synthesis_hopsize == int(self.frame_size // 2):
            normalization = 1.0
        else:
            normalization = self.synthesis_window.sum()

        # Generate our synthesis_frames
        synthesis_frames = (analysis_frames * self.synthesis_window) / normalization

        # Reconstruct our signal by using the synthesis_frames
        return reconstruct_from_frames(synthesis_frames, self.synthesis_hopsize)
    

class PV(TSM):
    def __init__(self, 
                 frame_size: int = 256, 
                 speed_factor: float = 1, 
                 synthesis_hopsize: int = None, 
                 analysis_hopsize: int = None) -> None:
        super().__init__(frame_size, speed_factor, synthesis_hopsize, analysis_hopsize)


    def run(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Implements a TSM based on the Phase Vocoder algorithm
        This method is suitable for harmonic sounds
        """
        return signal
    

class HPS(TSM):
    def __init__(self, 
                 frame_size: int = 256, 
                 speed_factor: float = 1, 
                 synthesis_hopsize: int = None, 
                 analysis_hopsize: int = None) -> None:
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
