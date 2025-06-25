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
from timstofu.numba_helper import decount
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_arange
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.numba_helper import permute_inplace
from timstofu.sort_and_pepper import is_lex_nondecreasing

from mmapuccino import empty
from timstofu.sort_and_pepper import rank
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

from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset

from mmapuccino import MmapedArrayValuedDict
from pathlib import Path
from shutil import rmtree


SIMULATED = FALSE

if SIMULATED:
    simulated_precursors_path = Path("/home/matteo/tmp/simulated_precursors.mmappet")
    try:
        simulated_precursors = LexSortedDataset.from_tofu(simulated_precursors_path)
    except Exception:
        simulated_sorted_clusters_path = Path(
            "/home/matteo/tmp/simulated_sorted_clusters.mmappet"
        )
        rmtree(simulated_sorted_clusters_path, ignore_errors=True)
        simulated_sorted_clusters_path.mkdir(parents=True)
        rmtree(simulated_precursors_path, ignore_errors=True)
        simulated_precursors_path.mkdir(parents=True)

        mmap_sorted_clusters = MmapedArrayValuedDict(simulated_sorted_clusters_path)
        mmap_simulated_precursors = MmapedArrayValuedDict(simulated_precursors_path)
        sorted_clusters = LexSortedClusters.from_df(
            df=open_dataset_dct("/home/matteo/tmp/test1.mmappet"),
            _empty=mmap_sorted_clusters.empty,
        )
        simulated_precursors = sorted_clusters.deduplicate(
            _empty=mmap_simulated_precursors.empty,
            _zeros=mmap_simulated_precursors.zeros,
        )

    frames, scans, frame_scan_to_count = melt(simulated_precursors.counts)
    assert frame_scan_to_count.sum() == len(simulated_precursors)

    frames = decount(frames.astype(np.uint32), frame_scan_to_count)
    scans = decount(scans.astype(np.uint32), frame_scan_to_count)
    tofs = sorted_clusters.columns.tof
    intensities = sorted_clusters.columns.intensity
    frame_counts = count1D(frames)
else:
    folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
    raw_data = OpenTIMS(folder_dot_d)
    chosen_frames = raw_data.ms1_frames

    # do we need frames at the beginning? nope. others yes, sort of.
    cols = raw_data.query(chosen_frames, columns=("frame", "scan", "tof", "intensity"))
    frames, scans, tofs, intensities = cols.values()
    frame_counts = raw_data.frames["NumPeaks"][chosen_frames - 1]


frame_max, scan_max, tof_max, intensity_max = map(
    lambda v: v.max() + 1, (frames, scans, tofs, intensities)
)
# +1 needed to not go out of scope


paranoid: bool = False

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
def neighbors_above_threshold_left_and_right(
    intensities,
    s,
    e,
    i,
    min_fraction_of_intensity=0,
):
    i = np.intp(i)
    left = i
    min_intensity = min_fraction_of_intensity * intensities[i]
    while left >= s and intensities[left] > min_intensity:
        left -= 1
    right = i
    while right <= e and intensities[right] > min_intensity:
        right += 1
    # Clamp to boundaries
    left = max(left, s)
    right = min(right, e)
    return left, right


@numba.njit
def get_total_ion_current(intensities, left, right):
    tic = 0
    for i in range(left, right):
        tic += intensities[i]
    return tic


@numba.njit
def foo2(
    s,
    e,
    zz,
    intensities,
    radius,
    zz_total_span,
    event_count,
    total_ion_current,
    is_max,
    left_direct,
    right_direct,
    min_fraction_of_intensity,
):
    left = s
    right = s
    for i in range(s, e):
        zz_total_span[i] = e - s
        # zz sorted -> can update left and right at will without errors
        left, right = get_window_borders(i, e, zz, radius, left, right)
        event_count[i] = right - left
        total_ion_current[i] = get_total_ion_current(intensities, left, right)
        is_max[i] = max_around(intensities, i, left, right) == intensities[i]
        left_direct[i], right_direct[i] = neighbors_above_threshold_left_and_right(
            intensities, left, right, i, min_fraction_of_intensity
        )


scan_neighborhood_size = 100
min_fraction_of_intensity = 0.5

zz_total_span = np.zeros(dtype=np.uint32, shape=len(scans))
event_count = np.zeros(dtype=np.uint32, shape=len(scans))
total_ion_current = np.zeros(dtype=np.uint32, shape=len(scans))
is_max = np.zeros(dtype=np.bool_, shape=len(scans))
left_direct = np.zeros(dtype=np.uint32, shape=len(scans))
right_direct = np.zeros(dtype=np.uint32, shape=len(scans))
with ProgressBar(total=len(frame_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_frame = map_onto_lexsorted_indexed_data(
        frame_index,
        tofs,
        foo2,
        (  # foo args
            scans,
            intensities,
            100,
            zz_total_span,
            event_count,
            total_ion_current,
            is_max,
            left_direct,
            right_direct,
            min_fraction_of_intensity,
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

consecutive_size = right_direct - left_direct
consecutive_sizes, consecutive_sizes_cnt = np.unique(
    consecutive_size, return_counts=True
)
plt.scatter(event_count_size, event_count_cnt, label="nonzero events")
plt.scatter(
    consecutive_sizes, consecutive_sizes_cnt, label="consecutive nonzero events"
)
# plt.xscale("log")
plt.xlabel("SIZE IN SCAN DIM")
plt.yscale("log")
plt.ylabel("COUNT")
plt.legend()
plt.show()


tof_counts = count1D(tofs)
tof_index = get_index(tof_counts)
tof_scan_frame_order = grouped_lexargcountsort(
    arrays=(scans, frames),
    group_index=tof_index,
    array_maxes=(scan_max, frame_max),
    # order=?, RAM OPTIMIZATION POSSIBILITY?
)
if paranoid:
    assert is_permutation(tof_scan_frame_order)


tof_scan_frame_to_frame_tof_scan_perm = rank(
    tof_scan_frame_order
)  # can give array now too for RAM savings.

# reorder arrays at small RAM price, but only once.
_visited = permute_inplace(scans, tof_scan_frame_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(tofs, tof_scan_frame_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(intensities, tof_scan_frame_order, visited=_visited)
if paranoid:
    np.all(_visited)

if paranoid:
    assert is_lex_nondecreasing(frames, tofs, scans)  # huzzzaah! sorted


# problem: we will need to likely try to extend the dims by one observation to each scan and frame

# problem: how to actually choose the peak tops?
