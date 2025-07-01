"""
%load_ext autoreload
%autoreload 2
"""
import numpy as np

from opentimspy import OpenTIMS

from timstofu.numba_helper import decount
from timstofu.pivot import Pivot
from timstofu.stats import get_index

from timstofu.numba_helper import is_permutation
from timstofu.sort_and_pepper import is_lex_nondecreasing

from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort

paranoid = True
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)
# LexSortedDataset.from_tdf(folder_dot_d)

urt2frame = raw_data.ms1_frames
cols = raw_data.query(urt2frame, columns=("scan", "tof", "intensity"))
scans, tofs, intensities = cols.values()
urt_counts = raw_data.frames["NumPeaks"][urt2frame - 1]
urt_max = len(urt_counts)
urts = decount(
    np.arange(urt_max, dtype=scans.dtype),
    urt_counts,
)

# provide counts and indices if available.
tof_scan_urt = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
)
tof_scan_urt.is_sorted()
tof_scan_urt.sort()
tof_scan_urt.is_sorted()
