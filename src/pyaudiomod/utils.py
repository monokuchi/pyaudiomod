
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass



@dataclass
class FrameShiftBoundaries:
    min_shift: int = -10
    max_shift: int = 10

    def __repr__(self) -> str:
        return f"Min Shift: {self.min_shift} Max Shift: {self.max_shift}"
    


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
