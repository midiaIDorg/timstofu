"""
%load_ext autoreload
%autoreload 2
"""
import math
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
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
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
from timstofu.stats import get_window_borders
from timstofu.stats import max_around
from timstofu.stats import max_intensity_in_window
from tqdm import tqdm

from timstofu.numba_helper import test_foo_for_map_onto_lexsorted_indexed_data
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


# trivial example
def test():
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
    return res


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


@numba.njit
def foo2(
    s,
    e,
    zz,
    radius,
    zz_total_span,
    event_count,
    total_ion_current,
    is_max,
    left_nonzero,
    right_nonzero,
):
    left = s
    right = s
    for i in range(s, e):
        zz_total_span[i] = e - s
        # zz sorted -> can update left and right
        left, right = get_window_borders(i, e, zz, radius, left, right)
        event_count[i] = right - left
        total_ion_current[i] = intensities[left:right].sum()
        is_max[i] = max_around(intensities, i, left, right) == intensities[i]


scan_neighborhood_size = 100

zz_total_span = np.zeros(dtype=np.uint32, shape=len(scans))
event_count = np.zeros(dtype=np.uint32, shape=len(scans))
total_ion_current = np.zeros(dtype=np.uint32, shape=len(scans))
is_max = np.zeros(dtype=np.bool_, shape=len(scans))
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index,
        tofs,
        foo2,
        (  # foo args
            scans,
            100,
            zz_total_span,
            event_count,
            total_ion_current,
            is_max,
        ),  # foo args
        progress_proxy,
    )

event_count_size, event_count_cnt = np.unique(event_count, return_counts=True)

plt.scatter(event_count_size, event_count_cnt)
plt.xlabel(
    f"Number of nonzero events in a scan neighborhood of size {2*scan_neighborhood_size+1}."
)
plt.yscale("log")
plt.ylabel("count")
plt.show()


tics, tics_cnt = np.unique(total_ion_current, return_counts=True)

plt.scatter(tics, tics_cnt, s=1)
plt.xscale("log")
plt.xlabel("TOTAL ION CURRENT")
plt.yscale("log")
plt.ylabel("COUNT")
plt.show()

np.sum(is_max)


tof_scan_frame_order = grouped_lexargcountsort(
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
