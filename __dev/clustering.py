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
from pathlib import Path

from kilograms import scatterplot_matrix
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from numba_progress import ProgressBar
from opentimspy import OpenTIMS
from shutil import rmtree
from tqdm import tqdm

from dictodot import DotDict
from mmapuccino import MmapedArrayValuedDict
from mmapuccino import empty

from timstofu.numba_helper import decount
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_arange
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.numba_helper import permute_inplace
from timstofu.numba_helper import test_foo_for_map_onto_lexsorted_indexed_data
from timstofu.plotting import plot_discrete_marginals
from timstofu.sort_and_pepper import grouped_lexargcountsort
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import rank
from timstofu.stats import count1D
from timstofu.stats import count2D
from timstofu.stats import count2D_marginals
from timstofu.stats import cumsum
from timstofu.stats import get_index
from timstofu.stats import get_unique_cnts_in_groups
from timstofu.stats import get_window_borders
from timstofu.stats import max_around
from timstofu.stats import max_intensity_in_window
from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset


PRECURSOR_LEVEL = 1
FRAGMENT_LEVEL = 2

ms_level: int = PRECURSOR_LEVEL
paranoid: bool = False


assert ms_level in (PRECURSOR_LEVEL, FRAGMENT_LEVEL)

if False:
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

        if ms_level == PRECURSOR_LEVEL:
            # urt = unit of retention time
            simulated_precursors_ur_based = simulated_precursors.cut_counts()
            (
                urts,
                scans,
            ), urt_scan_to_count = simulated_precursors_ur_based.melt_index(
                very_long=True
            )
        else:
            raise NotImplementedError
            # # TODO: we need to still do something here, or not?
            # urts, scans, urt_scan_to_count = melt(simulated_precursors.counts)

    # TODO: missing urt2frame.
    assert urt_scan_to_count.sum() == len(simulated_precursors)
    tofs = sorted_clusters.columns.tof
    intensities = sorted_clusters.columns.intensity

else:
    folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
    raw_data = OpenTIMS(folder_dot_d)
    # LexSortedDataset.from_tdf(folder_dot_d)

    if ms_level == PRECURSOR_LEVEL:
        urt2frame = raw_data.ms1_frames
    cols = raw_data.query(urt2frame, columns=("scan", "tof", "intensity"))
    scans, tofs, intensities = cols.values()
    urt_counts = raw_data.frames["NumPeaks"][urt2frame - 1]
    urt_max = len(urt_counts)
    urts = decount(
        np.arange(urt_max, dtype=get_min_int_data_type(urt_max, signed=False)),
        urt_counts,
    )

scan_max, tof_max, intensity_max = map(
    lambda v: v.max() + 1, (scans, tofs, intensities)
)
# +1 to remain in scope


urt_index = get_index(urt_counts)
event_count = urt_index[-1]
if paranoid:
    urt_scan_tof_order = grouped_lexargcountsort(
        arrays=(scans, tofs),
        group_index=urt_index,
        array_maxes=(scan_max, tof_max),
    )
    assert is_lex_nondecreasing(urt_scan_tof_order)
    assert urt_scan_tof_order[0] == 0
    assert urt_scan_tof_order[-1] == event_count - 1

urt_tof_scan_order = grouped_lexargcountsort(
    arrays=(tofs, scans),
    group_index=urt_index,
    array_maxes=(tof_max, scan_max),
    # order=?, RAM OPTIMIZATION POSSIBILITY?
)
if paranoid:
    assert is_permutation(urt_tof_scan_order)


# reorder arrays at small RAM price, but only once.
_visited = permute_inplace(scans, urt_tof_scan_order)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(tofs, urt_tof_scan_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(intensities, urt_tof_scan_order, visited=_visited)
if paranoid:
    np.all(_visited)

if paranoid:
    assert is_lex_nondecreasing(urts, tofs, scans)  # huzzzaah! sorted


# now, another approach: do local 3D peak counts and intensity sums.
# how many urt-scans per tof?
# tof_counts = count1D(tofs)
# tof_cnts, tof_cnts_cnts = np.unique(tof_counts, return_counts=True)
# plt.scatter(tof_cnts, tof_cnts_cnts)
# plt.xlabel("number of (urt,scan) pairs per tof")
# plt.ylabel("count")
# plt.xscale("log")
# plt.yscale("log")
# plt.show()


# too many points of change for urt,tof = 90M
# urt_tof_to_change, tofs_per_urt = get_index_2D(urt_index, tofs)


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
        unique_tofs_per_urt = map_onto_lexsorted_indexed_data(
            urt_index[:N_max],
            tofs,
            foo1,
            (res,),  # foo1 args
            progress_proxy,
        )
    return res


@numba.njit(boundscheck=True)
def max_nonzero_step_up(i, xx, radius):
    x_prev = xx[i]
    for j in range(0, radius):
        x = xx[i + j + 1]
        if x > x_prev + 1:
            break
        x_prev = x
    return j


@numba.njit(boundscheck=True)
def max_nonzero_step_down(i, xx, radius):
    i = np.intp(i)
    x_prev = xx[i]
    for j in range(0, radius):
        x = xx[i - j - 1]
        if x + 1 < x_prev:
            break
        x_prev = x
    return j


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
    # results
    zz_total_span,
    event_count,
    total_ion_current,
    is_max,
    left_direct,
    right_direct,
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
        left_direct[i] = max_nonzero_step_up(i, zz, radius)
        right_direct[i] = max_nonzero_step_down(i, zz, radius)


scan_neighborhood_radius = 10
scan_neighborhood_size = 2 * scan_neighborhood_radius + 1

counts_dtype = get_min_int_data_type(scan_neighborhood_size, signed=False)
stats = DotDict(
    zz_total_span=np.zeros(dtype=counts_dtype, shape=len(scans)),
    event_count=np.zeros(dtype=counts_dtype, shape=len(scans)),
    total_ion_current=np.zeros(dtype=np.uint32, shape=len(scans)),
    is_max=np.zeros(dtype=np.bool_, shape=len(scans)),
    left_direct=np.zeros(dtype=counts_dtype, shape=len(scans)),
    right_direct=np.zeros(dtype=counts_dtype, shape=len(scans)),
)
with ProgressBar(total=len(urt_index) - 1, desc="Getting stats") as progress_proxy:
    unique_tofs_per_urt = map_onto_lexsorted_indexed_data(
        urt_index,
        tofs,
        foo2,
        (  # foo args
            scans,
            intensities,
            scan_neighborhood_size,
            *tuple(stats.values()),  # results
        ),  # foo args
        progress_proxy,
    )

event_count_size, event_count_cnt = np.unique(stats.event_count, return_counts=True)
plt.scatter(event_count_size, event_count_cnt)
plt.xlabel(
    f"Number of nonzero events in a scan neighborhood of size {2*scan_neighborhood_size+1}."
)
plt.title(f"Summary for {len(tofs):_} events")
plt.yscale("log")
plt.ylabel("count")
plt.show()


tics, tics_cnt = np.unique(stats.total_ion_current, return_counts=True)

plt.scatter(tics, tics_cnt, s=1)
plt.title(f"Summary for {len(tofs):_} events")
plt.xscale("log")
plt.xlabel(f"TOTAL ION CURRENT IN SCAN ± {scan_neighborhood_size}")
plt.yscale("log")
plt.ylabel("COUNT")
plt.show()

np.sum(stats.is_max)

consecutive_size = stats.right_direct + stats.left_direct + 1
consecutive_sizes, consecutive_sizes_cnt = np.unique(
    consecutive_size, return_counts=True
)
right_sizes, right_sizes_cnt = np.unique(stats.right_direct, return_counts=True)
left_sizes, left_sizes_cnt = np.unique(stats.left_direct, return_counts=True)

plt.scatter(event_count_size, event_count_cnt, label="nonzero events")
plt.scatter(
    consecutive_sizes, consecutive_sizes_cnt, label="consecutive non-zero events"
)
plt.scatter(right_sizes, right_sizes_cnt, label="non-zero events right")
plt.scatter(left_sizes, left_sizes_cnt, label="non-zero events left")
plt.title(f"Summary for {len(tofs):_} events")
plt.xlabel(f"SCAN ± {scan_neighborhood_size} STRIPE")
plt.yscale("log")
plt.ylabel("COUNT")
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


plot_discrete_marginals(
    marginals=count2D_marginals(
        {
            c: stats[c]
            for c in [
                "event_count",
                "left_direct",
                "right_direct",
            ]
        }
    ),
    imshow_kwargs=dict(norm=LogNorm()),
)

# plot_discrete_marginals(marginals, m=3)
# plot_discrete_marginals(marginals, n=3)


# should this not use yet another data structure inheriting from CompactDataset?
tof_counts = count1D(tofs)
tof_index = get_index(tof_counts)
tof_scan_urt_order = grouped_lexargcountsort(
    arrays=(scans, urts),
    group_index=tof_index,
    array_maxes=(scan_max, urt_max),
    # order=?, RAM OPTIMIZATION POSSIBILITY?
)
if paranoid:
    assert is_permutation(tof_scan_urt_order)


tof_scan_urt_to_urt_tof_scan_perm = rank(
    tof_scan_urt_order
)  # can give array now too for RAM savings.

# reorder arrays at small RAM price, but only once.
_visited = permute_inplace(scans, tof_scan_urt_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(tofs, tof_scan_urt_order, visited=_visited)
if paranoid:
    np.all(_visited)
_visited = permute_inplace(intensities, tof_scan_urt_order, visited=_visited)
if paranoid:
    np.all(_visited)

if paranoid:
    assert is_lex_nondecreasing(urts, tofs, scans)  # huzzzaah! sorted


# problem: we will need to likely try to extend the dims by one observation to each scan and urt

# problem: how to actually choose the peak tops?
