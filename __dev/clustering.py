"""
%load_ext autoreload
%autoreload 2
"""
import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import shutil

from collections import Counter
from math import inf

from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from numba_progress import ProgressBar
from opentimspy import OpenTIMS

from dictodot import DotDict
from pathlib import Path
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_arange
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import melt
from timstofu.numba_helper import permute_inplace
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.tofu import LexSortedDataset

from mmapuccino import empty
from timstofu.sort_and_pepper import rank_array
from timstofu.stats import count1D
from timstofu.stats import count2D
from timstofu.stats import cumsum
from timstofu.stats import get_index
from timstofu.stats import get_unique_cnts_in_groups
from timstofu.stats import max_intensity_in_window
from tqdm import tqdm

from timstofu.sort_and_pepper import grouped_lexargcountsort


folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)

# rd1 = pd.DataFrame(
#     raw_data.query(1, columns=("frame", "scan", "tof", "intensity")), copy=False
# )
# rd1.sort_values(["frame","tof","scan"])
# Counter(rd1.groupby(["frame","tof"]).size())
chosen_frames = raw_data.ms1_frames

# do we need frames at the beginning? nope. others yes, sort of.
cols = raw_data.query(chosen_frames, columns=("frame", "scan", "tof", "intensity"))
frames, scans, tofs, intensities = cols.values()
frame_max, scan_max, tof_max, intensity_max = (
    v.max() + 1 for col, v in cols.items()
)  # +1 needed to not go out of scope
paranoid: bool = False

frame_counts = raw_data.frames["NumPeaks"][chosen_frames - 1]
frame_index = get_index(frame_counts)
event_count = frame_index[-1]
if paranoid:
    frame_scan_tof_order = grouped_lexargcountsort(
        arrays=(scans, tofs),
        group_index=frame_index,
        array_maxes=(scan_max, tof_max),
    )
    assert is_lex_nondecreasing(frame_scan_tof_order)
    assert frame_scan_tof_order[0] == 0
    assert frame_scan_tof_order[-1] == event_count - 1

frame_tof_scan_order = grouped_lexargcountsort(
    arrays=(tofs, scans),
    group_index=frame_index,
    array_maxes=(tof_max, scan_max),
    # order=?, RAM OPTIMIZATION POSSIBILITY?
)
if paranoid:
    assert is_permutation(frame_tof_scan_order)


# reorder arrays at small RAM price, but only once.
_visited = permute_inplace(scans, frame_tof_scan_order)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(tofs, frame_tof_scan_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(intensities, frame_tof_scan_order, visited=_visited)
if paranoid:
    np.all(_visited)

if paranoid:
    assert is_lex_nondecreasing(frames, tofs, scans)  # huzzzaah! sorted


# from timstofu.numba_helper import test_foo_for_map_onto_lexsorted_indexed_data
# from timstofu.stats import max_around

# now, another approach: do local 3D peak counts and intensity sums.

# how many frame-scans per tof?
# tof_counts = count1D(tofs)
# tof_cnts, tof_cnts_cnts = np.unique(tof_counts, return_counts=True)
# plt.scatter(tof_cnts, tof_cnts_cnts)
# plt.xlabel("number of (frame,scan) pairs per tof")
# plt.ylabel("count")
# plt.xscale("log")
# plt.yscale("log")
# plt.show()


# too many points of change for frame,tof = 90M
# frame_tof_to_change, tofs_per_frame = get_index_2D(frame_index, tofs)
# from timstofu.numba_helper import map_onto_lexsorted_indexed_data
# from timstofu.stats import get_window_borders

import math


@numba.njit(boundscheck=True, parallel=True)
def map_onto_lexsorted_indexed_data(
    xx_index: npt.NDArray,
    yy_by_xx: npt.NDArray,
    foo,
    foo_args,
    progress_proxy: ProgressBar | None = None,
) -> npt.NDArray:
    """Here we iterate over indexed xx but in groups by yy.

    Notes
    -----
    Data (xx,yy) must be lexicographically sorted.
    `foo` must be
    """
    unique_y_per_x = np.zeros(shape=len(xx_index) - 1, dtype=np.uint32)
    for i in numba.prange(len(xx_index) - 1):
        j_s = xx_index[i]
        j_e = xx_index[i + 1]
        j_prev = j_s
        y_prev = yy_by_xx[j_prev]
        for j in range(j_s + 1, j_e):
            y = yy_by_xx[j]
            if y != y_prev:
                unique_y_per_x[i] += 1
                foo(j_prev, j, *foo_args)
                y_prev = y
                j_prev = j
        foo(j_prev, j_e, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(1)
    return unique_y_per_x


@numba.njit
def foo1(s, e, res):
    left = s
    right = s
    for i in range(s, e):
        res[i] = e - s


res = np.zeros(dtype=np.uint32, shape=len(scans))
N_max = 100
with ProgressBar(total=N_max - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index[:N_max],
        tofs,
        foo1,
        (res,),  # foo1 args
        progress_proxy,
    )


@numba.njit(boundscheck=True)
def get_window_borders(i, i_max, xx, radius, left=0, right=0):
    x = xx[i]
    while left < i and xx[left] + radius < x:
        left += 1
    right = max(i, right)
    while right < i_max and xx[right] <= x + radius:
        right += 1
    return left, right


@numba.njit
def foo2(s, e, zz, radius, res, borders):
    left = s
    right = s
    for i in range(s, e):
        res[i] = e - s
        # zz sorted -> can update left and right
        left, right = get_window_borders(i, e, zz, radius, left, right)
        borders[i, 0] = left
        borders[i, 1] = right


N = 10000
res = np.zeros(dtype=np.uint32, shape=len(scans))
borders = np.zeros(dtype=np.uint32, shape=(len(scans), 2))
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index[:N],
        tofs,
        foo2,
        (  # foo args
            scans,
            10,
            res,
            borders,
        ),  # foo args
        progress_proxy,
    )


frame_tof_cnt, cnt = np.unique(res, return_counts=True)
# How many scans per frame,tof?
# plt.scatter(frame_tof_cnt, cnt)
# plt.xlabel("number of scans per (frame,tof)")
# plt.ylabel("count")
# # plt.xscale("log")
# plt.yscale("log")
# plt.show()


# seems we are not vising all.
len(np.nonzero(res == 0)[0])
frame_index
# should I have changed frame index? no, why?


@numba.njit(boundscheck=True)
def get_window_borders(i, xx, radius, left=0, right=0):
    """
    Parameters
    ----------
    i (int): Current index in xx.
    xx (np.array): A sorted array (non-decrasing).
    left (int): Currently explored left end.
    right (int): Currently explored right end.
    radius (int): A positive integer.

    Returns
    -------
    tuple: updated values of left and right.
    """
    N = len(xx)
    x = xx[i]
    # Move left pointer to ensure xx[i] - xx[left] <= radius
    while left < N and radius + xx[left] < x:
        left += 1
    right = max(i, right)
    # Move right pointer to ensure xx[right] - xx[i] <= radius
    while right < N and xx[right] <= x + radius:
        right += 1
    return left, right


@numba.njit
def foo(s, e, yy, intensities, radius, boundries):
    for i in range(s, e):
        s, e = get_window_borders(i, yy, radius, s, e)
        boundries[i, 0] = s
        boundries[i, 1] = e


results = DotDict(
    scans_boundries=np.empty(dtype=scans.dtype, shape=(len(scans), 2)),
)
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index,
        tofs,
        foo,  # foo
        # test_foo_for_map_onto_lexsorted_indexed_data,
        (  # foo args
            scans,
            intensities,
            10,
            results.scans_boundries,
        ),
        progress_proxy,
    )


@numba.njit(boundscheck=True)
def find_bounds(xx, s, e, i, x_min):
    i = np.int64(i)
    # Search left
    left = i
    while left >= s and xx[left] <= x_min:
        left -= 1
    # Search right
    right = i
    while right <= e and xx[right] <= x_min:
        right += 1
    # Clamp to boundaries
    left = max(left, s)
    right = min(right, e)
    return left, right


@numba.njit(boundscheck=True)
def window_stats(
    left,
    right,
    # real args
    zz,
    intensities,
    radius,
    min_intensity,
    # results
    is_local_max,
    boundries,
):
    s = left
    e = left
    for i in range(left, right):
        s, e = get_window_borders(i, zz, radius, s, e)
        s0, e0 = find_bounds(intensities, s, e, i, min_intensity)
        boundries[i, 0] = s
        boundries[i, 1] = e
        # boundries[i, 0] = s0
        # boundries[i, 1] = e0
        neighborhood_max = max_around(intensities, i, s, e) if e - s > 0 else 0
        is_local_max[i] = neighborhood_max == intensities[i]


results = DotDict(
    is_local_max=np.empty(dtype=np.bool_, shape=intensities.shape),
    scans_boundries=np.empty(dtype=scans.dtype, shape=(len(scans), 2)),
)
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index,
        tofs,
        window_stats,  # foo
        # test_foo_for_map_onto_lexsorted_indexed_data,
        (  # foo args
            scans,
            intensities,
            10,
            0,
            results.is_local_max,
            results.scans_boundries,
        ),
        progress_proxy,
    )

local_maxes_cnt = np.sum(is_local_max)
f"{local_maxes_cnt / len(is_local_max) * 100:.4f}%"
# what is needed: results allocator and stats function and its parameters.


@numba.njit(boundscheck=True)
def area_stats(
    left: int,
    right: int,
    # real args
    zz: npt.NDArray,
    intensities: npt.NDArray,
    radius: int,
    # results
    left_border: npt.NDArray,
    right_border: npt.NDArray,
    cnts: npt.NDArray,
    tics: npt.NDArray,
):
    s = left
    e = left
    for i in range(left, right):
        s, e = get_window_borders(i, zz, radius, s, e)  # indices of the moving window
        left_border[i] = s
        right_border[i] = e
        cnts[i] = e - s
        tics[i] = intensities[s:e].sum()


results = DotDict(
    left=np.zeros(dtype=np.uint32, shape=intensities.shape),
    right=np.zeros(dtype=np.uint32, shape=intensities.shape),
    cnts=np.zeros(dtype=np.uint32, shape=intensities.shape),
    tics=np.zeros(dtype=np.uint32, shape=intensities.shape),
)
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index,
        tofs,
        area_stats,  # foo
        # test_foo_for_map_onto_lexsorted_indexed_data,
        (  # foo args
            scans,
            intensities,
            100,
            results.left,
            results.right,
            results.cnts,
            results.tics,
        ),
        progress_proxy,
    )

np.all(results.right - results.left == results.cnts)

funny = results.cnts == 24
funny.nonzero()

results.left[51336746]
results.right[51336746]

results.left
# look like the last value is not considered at all and every left window borded is always the same as before... annoying. Can it be true?
# Perhaps?

np.unique(results.cnts, return_counts=True)
