"""
%load_ext autoreload
%autoreload 2
"""
import itertools
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numba_progress import ProgressBar
from opentimspy import OpenTIMS


# from timstofu.math import moving_window
from timstofu.math import _minmax
from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import merge_intervals
from timstofu.misc import filtering_str
from timstofu.misc import get_max_count
from timstofu.numba_helper import decount
from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import filter_nb
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.pivot import Pivot
from timstofu.plotting import plot_discrete_marginals
from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort
from timstofu.sort_and_pepper import increases
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import is_nondecreasing
from timstofu.stats import count1D
from timstofu.stats import count2D_marginals
from timstofu.stats import get_index
from timstofu.timstofmisc import deduce_shift_and_spacing

paranoid = True
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)
# LexSortedDataset.from_tdf(folder_dot_d)

index_data = raw_data[:]

ms1_frames = urt2frame = raw_data.ms1_frames
ms1_shift, ms1_spacing = deduce_shift_and_spacing(ms1_frames)

index_data[:,0] -= ms1_shift
index_data[:,0] //= ms1_spacing

discretize(index_data[:,3], index_data[:,3], transform=log2)
maxes = index_data.max(axis=0)

# skoro to trzymam, to po co mi to zamieniać w liczby? po to, żeby sortować?
# nie otwierajmy tego na razie... jestem idiotą... :(




urt_counts = raw_data.frames["NumPeaks"][urt2frame - 1]
urt_max = len(urt_counts)




cols = raw_data.query(urt2frame, columns=("scan", "tof", "intensity"))
scans, tofs, intensities = cols.values()
urts = decount(
    np.arange(urt_max, dtype=np.uint32),
    urt_counts,
)

# dintensity_counts = np.zeros(dtype=np.int32, shape=256)
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
    scan=20,
    urt=10,
)

leading_index = pivot.get_index(pivot.leading_column)
chunk_ends = divide_indices(len(pivot), k=16)

chunk_ends_avoiding_lead_dim_splits = pivot.get_chunk_ends(number_of_chunks=16)

full_data_shape = pivot.maxes[1:-1]
div = np.uint64(np.prod(pivot.maxes[1:]))


# should we put an extra event at the end of the array???


@numba.njit(parallel=True, boundscheck=True)
def check_how_long(
    data,
    chunk_ends,
    leading_index,
    full_data_shape,
    div,
    progress_proxy=None,
):
    for chunk_idx in numba.prange(len(chunk_ends)):
        chunk_s, chunk_e = chunk_ends[chunk_idx]
        zeros = np.zeros(shape=full_data_shape, dtype=np.uint32)

        s = data[chunk_s] // div
        e = data[chunk_e] // div
        leading_dim_idx = leading_index[s:e]

        for i in range(e-s-1):
            for idx in range(leading_dim_idx[i], leading_dim_idx[i+1]):

            zeros[:] = 0
            
