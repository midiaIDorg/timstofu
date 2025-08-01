import numpy as np

from numpy.typing import NDArray

from timstofu.sort_and_pepper import increases


def deduce_shift_and_spacing(ms1_frames: NDArray) -> tuple[int, int]:
    assert increases(ms1_frames)
    shift = ms1_frames[0]
    spacing = np.unique(np.diff(ms1_frames - shift))
    assert len(spacing) == 1
    return int(shift), int(spacing[0])
