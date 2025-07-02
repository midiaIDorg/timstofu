"""
%load_ext autoreload
%autoreload 2
"""
import matplotlib.pyplot as plt
import numba
import numpy as np

from numba_progress import ProgressBar
from opentimspy import OpenTIMS

from timstofu.numba_helper import decount
from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.pivot import Pivot
from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.stats import count1D
from timstofu.stats import get_index

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


# urt_tof_scan = Pivot.new(
#     urt=urts,
#     tof=tofs,
#     scan=scans,
# )
# urt_tof_scan.is_sorted()
# urt_tof_scan.sort()
# assert urt_tof_scan.is_sorted()


tof_scan_urt = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
)
tof_scan_urt.sort()
assert tof_scan_urt.is_sorted()

radii = np.array((1, 1, 1))
shape = 2 * radii + 1

indices = np.zeros(shape, dtype=get_min_int_data_type(len(tof_scan_urt)))

weights = intensities[:100]
data = tof_scan_urt.array[:1000]

def foo(indices, current_idx,)
    



# Iterate over starts end positions.
chunk_ends = divide_indices(len(tof_scan_urt))

@numba.njit(parallel=True)
def nbmap(chunk_ends, indices, progress_proxy):
    for i in numba.prange(len(chunk_ends)):
        s, e = chunk_ends[i]
        local_indices = indices.copy()
        local_indices[:] = s
        progress_proxy.update(e-s)

with ProgressBar(
    total=len(tof_scan_urt),
    desc="Counting scans per tof",
) as progress_proxy:
    nbmap(
        chunk_ends,
        indices,
        progress_proxy,
    )





