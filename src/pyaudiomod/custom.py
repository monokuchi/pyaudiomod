
from dataclasses import dataclass



@dataclass
class FrameShiftBoundaries:
    min_shift: int = -10
    max_shift: int = 10

    def __repr__(self) -> str:
        return f"Min Shift: {self.min_shift} Max Shift: {self.max_shift}"
    