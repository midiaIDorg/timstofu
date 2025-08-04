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
from timstofu.misc import iter_array_splits
from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import filter_nb
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.numba_helper import overwrite
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
from timstofu.stats import count2D
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

##### SORTING: data was frame-scan-tof ordered and our sort is stable, so ...
tof_frame_scan_order = argcountsort(data[:, 2], tof_counts)

data[:, 3] = data[tof_frame_scan_order, 3]  # satellite data: can stay where it is
data[:, 2] = data[tof_frame_scan_order, 1]  # moving scans down
data[:, 1] = data[tof_frame_scan_order, 0]  # moving urts down
repeat(counts=tof_counts, results=data[:, 0])  # re-constructing tofs
data_pd.columns = ["tof", "urt", "scan", "intensity"]
assert is_lex_increasing(data), "Data was not lex sorted by tof-frame-scan."
# plot_counts(counts.tof)


tof_urt_counts, *_ = count2D(data_pd.tof, data_pd.urt, get_min_int_data_type(len(data)))

scans_no, scans_cnt = np.unique(tof_urt_counts, return_counts=True)

plt.scatter(scans_no, scans_cnt)
plt.xlabel("Number of scans per tof-urt")
plt.ylabel("count")
plt.yscale("log")
plt.show()

# TODO on Monday: try to prepare some operations on it.


dim_names = data_pd.columns[:3]
x_index = tof_index# I have to split those pairs smaller ones.



def split_contiguous_by_total_count(counts, k):
    counts = np.asarray(counts)
    total = counts.sum()
    target = total / k
    cumsum = np.cumsum(counts)

    split_indices = [0]
    last_split = 0

    for _ in range(1, k):
        # Find the split point that brings the sum closest to the next multiple of target
        ideal = split_indices[-1] + np.searchsorted(cumsum[last_split:], cumsum[last_split] + target)
        # Clip to ensure it's strictly increasing
        ideal = min(len(counts), max(ideal, last_split + 1))
        split_indices.append(ideal)
        last_split = ideal

    split_indices.append(len(counts))  # Ensure the last segment ends at the array end

    # Slice the array
    chunks = [counts[split_indices[i]:split_indices[i+1]] for i in range(k)]

    return chunks, np.array(split_indices)


len(tof_counts)
tof_index
chunks, split_points = split_contiguous_by_total_count(tof_counts, 16)

tof_index[split_points]


for i, (chunk, start, end) in enumerate(zip(chunks, split_points[:-1], split_points[1:])):
    print(f"Chunk {i}: counts[{start}:{end}] = {chunk}, sum = {chunk.sum()}")
    print(tof_index[start:end])
    print(np.diff(tof_index[start:end]))
    print()


get_index(np.array(list(map(np.sum, chunks))))
tof_index

tof_counts
x_index[len(x_index) - 2]

size, remainder = divmod(len(x_index) - 2, 16)
k = 16
N = len(x_index)-1
q, r = divmod(N, k)

starts = []
start = 0
for i in range(k):
    end = start + q + (1 if i < r else 0)
    starts.append(start)
    start = end


min_x = 0
max_x =
len(x_index)-1


x_start_end_tuples = np.array(list(iter_array_splits(x_index, k=16)))
x_start_end_tuples[:,1] += 1



x_index[:-1]
N = len(x_index)-1
k = 16


size, remainder = divmod(N,k)
np.arange(0, N, size)




N // 16
N / 16
N % 16 to first those add 1




for s, e in x_start_end_tuples:
    print(s, e, x_index[s:e])
# what do I need splits for then? I only need start ends.


w = x_indices[0][1]
w[len(w) : len(w) + 1]


@numba.njit(parallel=True, boundscheck=True)
def moving_widow(
    data,
    x_index,
    x_start_end_tuples,
    x_radius,
    y_radius,
    z_radius,
    x_max,
    y_max,
    z_max,
    # foo,
    # foo_args=(),
    progress_proxy=None,
):
    """event = (x,y,z,intensity)"""
    yy = data[:, 1]
    for split_s, split_e in numba.prange(len(x_start_end_tuples)):
        y_counts = np.zeros(y_max + 1, data.dtype)
        y_indices = np.zeros((x_radius * 2 + 1, len(y_counts) + 1), data.dtype)

        events_visited = 0
        for x in range(split_s, split_e):
            sx = x_index[split_idx]  # start of x in data
            ox = x_index[split_idx + 1]  # end of x in data

            y_counts[:] = 0
            _count1D(yy[sx:ox], y_counts)  # x conditioned y counts

            y_indices[:, 0] = 0
            get_index(y_counts, dx_y_index[0])  # x conditioned y index

            for idx in range(sx, ox):  # idx are where tof is fixed
                pass

            events_visited += ox - sx

        if progress_proxy is not None:
            progress_proxy.update(events_visited)


with ProgressBar(
    total=len(data),
    desc=f"moving widow",
) as progress_proxy:
    moving_widow(
        data,
        x_index_splits,
        *(radii[dim] for dim in dim_names),
        *(maxes[dim] for dim in dim_names),
        progress_proxy,
    )


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
        x_indices_splits,
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
