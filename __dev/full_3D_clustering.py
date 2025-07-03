"""
%load_ext autoreload
%autoreload 2
"""
import matplotlib.pyplot as plt
import numba
import numpy as np

from dictodot import DotDict
from numba_progress import ProgressBar
from opentimspy import OpenTIMS

from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import moving_window
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

dintensity_counts = np.zeros(dtype=np.int32, shape=256)
dintensity_counts = discretize(intensities, transform=log2)

pivot = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
    dintensity=dintensity_counts,  # perhaps this should go out???
)
pivot.sort()
assert pivot.is_sorted()
radii = dict(
    tof=1,
    scan=5,
    urt=4,
)

# shape = np.append(2 * radii + 1, 2)
# N = 1_000
intensities = pivot.extract("dintensity")
chunk_ends = divide_indices(len(pivot), k=16 * 100)


diffs_dct = pivot.get_stencil_diffs(**radii)
diffs = np.array(list(diffs_dct.values()))
# diffs = diffs[diffs >= 0]


@numba.njit(boundscheck=True)
def local_max(current_idx, start_idx, end_idx, intensities, results):
    results[current_idx] = max(
        results[current_idx], intensities[start_idx:end_idx].max()
    )


updater = local_max
updater_results = DotDict(max_intensity=np.zeros(len(pivot), intensities.dtype))

with ProgressBar(
    total=len(pivot),
    desc=f"Getting stats in window {radii}",
) as progress_proxy:
    moving_window(
        chunk_ends,
        diffs,
        pivot.array,
        updater,
        (intensities, *tuple(updater_results.values())),
        progress_proxy,
    )


updater_results.max_intensity
count1D(updater_results.max_intensity)


np.unique(updater_results.max_intensity == intensities, return_counts=True)
