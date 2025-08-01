"""
%load_ext autoreload
%autoreload 2
"""
import functools
import itertools
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numba_progress import ProgressBar
from numpy.typing import NDArray
from opentimspy import OpenTIMS


# from timstofu.math import moving_window
from timstofu.math import _minmax
from timstofu.math import bit_width
from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import merge_intervals
from timstofu.misc import filtering_str
from timstofu.misc import get_max_count
from timstofu.misc import split_array
from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import filter_nb
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.numba_helper import permute_inplace
from timstofu.numba_helper import repeat
from timstofu.pivot import Pivot
from timstofu.plotting import df_to_plotly_scatterplot3D
from timstofu.plotting import plot_counts
from timstofu.plotting import plot_discrete_marginals
from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort
from timstofu.sort_and_pepper import increases
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import is_nondecreasing
from timstofu.sort_array_ops import is_lex_increasing
from timstofu.stats import _count1D
from timstofu.stats import count1D
from timstofu.stats import count2D_marginals
from timstofu.stats import get_index
from timstofu.timstofmisc import deduce_shift_and_spacing

discretize_intensity = False
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
radii = DotDict(tof=1, scan=20, urt=10)

# TODO: ask Michal to allow to get the order we want: will save only a bit of RAM.
data_handler = OpenTIMS(folder_dot_d)
data = data_handler[data_handler.ms1_frames]
ms1_shift, ms1_spacing = deduce_shift_and_spacing(data_handler.ms1_frames)
data[:, 0] -= ms1_shift
data[:, 0] //= ms1_spacing

if discretize_intensity:
    discretize(data[:, 3], data[:, 3], transform=log2)

data_pd = pd.DataFrame(data, copy=False, columns=["urt", "scan", "tof", "dintensity"])
assert np.shares_memory(
    data_pd.to_numpy(), data
), "Stupid pandas fucked us over again..."

assert is_lex_increasing(data, strictly=False), "Data is not urt-scan-tof sorted."

maxes = data_pd.max()
tof_counts = count1D(data_pd.tof)
tof_index = get_index(tof_counts)

##### SORTING
# This is tof_frame_scan_order as data was frame-scan-tof ordered and our sort is stable.
tof_frame_scan_order = argcountsort(data[:, 2], tof_counts)
data[:, 3] = data[tof_frame_scan_order, 3]  # satellite data: can stay where it is
data[:, 2] = data[tof_frame_scan_order, 1]
data[:, 1] = data[tof_frame_scan_order, 0]
repeat(
    counts=tof_counts, results=data[:, 0]
)  # direct application on dims and construction.
data_pd.columns = ["tof", "urt", "scan", "intensity"]

assert is_lex_increasing(data), "Data was not lex sorted by tof-frame-scan."
######
# plot_counts(counts.tof)

leading_index = tof_index
leading_index_splits = split_array(leading_index, 16, right_buffer=1)
empty_inner_index_count = np.zeros(maxes.urt + 1, data.dtype)


@numba.njit(parallel=True, boundscheck=True)
def moving_widow(
    data,
    leading_index_splits,
    empty_inner_index_count,
    # foo,
    # foo_args=(),
    progress_proxy=None,
):
    for split_idx in numba.prange(len(leading_index_splits)):
        leading_index = leading_index_splits[split_idx]
        inner_index_count = empty_inner_index_count.copy()
        for leading_idx in range(len(leading_index) - 1):
            # current tof range
            s0 = leading_index[leading_idx]
            o0 = leading_index[leading_idx + 1]

            # count tof specific urts
            # build urt index
            inner_index_count[:] = 0
            _count1D(data[s0:o0, 1], inner_index_count)

            # for idx in range(s0, o0):
            #     tof, urt, scan, intensity = data[idx]

            # foo(idx, thr_densifier, *foo_args)

            # for idx in range(start_idx, end_idx):  # thr_densifier[:] = 0 equivalent.
            #     tof, urt, scan, intensity = data[idx]
            #     thr_densifier[urt, scan] = 0

        if progress_proxy is not None:
            progress_proxy.update(index[-1] - index[0])


@numba.njit
def test(idx, thr_densifier, counts):
    counts[idx] = np.count_nonzero(thr_densifier)


test_args = DotDict(counts=np.zeros(len(data), np.uint32))


with ProgressBar(
    total=len(data),
    desc=f"checking_local_densification",
) as progress_proxy:
    moving_widow(
        data,
        leading_index_splits,
        densifier,
        test,
        tuple(*test_args.values()),
        progress_proxy,
    )


# most_intense_tof = np.argmax(counts.tof)
# tof = most_intense_tof
# k = 4
# X = data[indices.tof[tof - k] : indices.tof[tof + k + 1], :]
# X_pd = pd.DataFrame(X, copy=False, columns=["tof", "urt", "scan", "intensity"])
# df_to_plotly_scatterplot3D(X_pd, s=1).show()


# get_index
