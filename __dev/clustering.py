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
from timstofu.stats import max_intensity_in_window
from tqdm import tqdm

from timstofu.sort_and_pepper import grouped_argsorts
from timstofu.sort_and_pepper import lexargcountsort
from timstofu.sort_and_pepper import make_idx

grouped_argsort = grouped_argsorts[("safe", "multi_threaded")]

folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)

# rd1 = pd.DataFrame(
#     raw_data.query(1, columns=("frame", "scan", "tof", "intensity")), copy=False
# )
# rd1.sort_values(["frame","tof","scan"])
# Counter(rd1.groupby(["frame","tof"]).size())
chosen_frames = raw_data.ms1_frames

cols = raw_data.query(chosen_frames, columns=("frame", "scan", "tof", "intensity"))
frames, scans, tofs, intensities = cols.values()
F, S, T, I = (
    v.max() + 1 for col, v in cols.items()
)  # +1 needed to not go out of scope
paranoid: bool = False

frame_counts = raw_data.frames["NumPeaks"][chosen_frames - 1]
frame_index = get_index(frame_counts)
event_count = frame_index[-1]
if paranoid:
    scans_tofs = make_idx(
        (scans, tofs),
        (S, T),
        np.empty(shape=event_count, dtype=get_min_int_data_type(S * T)),
    )
    frame_scan_tof_order = grouped_argsort(scans_tofs, frame_index)
    assert is_lex_nondecreasing(frame_scan_tof_order)
    assert frame_scan_tof_order[0] == 0
    assert frame_scan_tof_order[-1] == event_count - 1

# Here ok: tofs and scans are ordered by frames.
tofs_scans = make_idx(
    (tofs, scans),
    (T, S),
    np.empty(shape=event_count, dtype=get_min_int_data_type(S * T)),
)
frame_tof_scan_order = grouped_argsort(tofs_scans, frame_index)  # need to test it too


permute_inplace(scans, frame_tof_scan_order)  # reorders at small RAM price
permute_inplace(tofs, frame_tof_scan_order)  # reorders at small RAM price
permute_inplace(intensities, frame_tof_scan_order)  # reorders at small RAM price
if paranoid:
    assert is_lex_nondecreasing(frames, tofs, scans)  # huzzzaah!


@numba.njit(boundscheck=True)
def max_intensity_in_window(results, xx, weights, radius):
    n = len(xx)
    left = 0
    right = 0
    for i in range(n):
        # Move left pointer to ensure xx[i] - xx[left] <= radius
        while left < n and xx[i] - xx[left] > radius:
            left += 1
        # Move right pointer to ensure xx[right] - xx[i] <= radius
        while right < n and xx[right] - xx[i] <= radius:
            right += 1

        max_val = 0.0
        found = False
        for j in range(left, right):
            if j != i:
                w = weights[j]
                if not found or w > max_val:
                    max_val = w
                    found = True
        results[i] = max_val if found else 0.0


# @numba.njit(boundscheck=True) # 3.95"
@numba.njit(boundscheck=True, parallel=True)  # 0.5"
def apply(
    group_index, yy, zz, weights, radius, results, unique_yy, progress_proxy=None
):
    for i in numba.prange(len(group_index) - 1):
        s = group_index[i]
        e = group_index[i + 1]
        prev_j = s
        for j in range(s + 1, e):
            if yy[j] != yy[prev_j]:
                max_intensity_in_window(
                    results[prev_j:j],
                    zz[prev_j:j],
                    weights[prev_j:j],
                    radius,
                )
                prev_j = j
                unique_yy[i] += 1
            if progress_proxy is not None:
                progress_proxy.update(1)


max_intensities = np.empty(dtype=intensities.dtype, shape=intensities.shape)
unique_tofs_per_frame = np.zeros(dtype=np.uint32, shape=len(frame_index) - 1)
apply(
    frame_index,
    tofs,
    scans,
    intensities,
    10,
    max_intensities,
    unique_tofs_per_frame,
)

np.sum(max_intensities == intensities)  # still a bit on the small side.

unique_max_intensities, max_intensity_counts = np.unique(
    max_intensities, return_counts=True
)
plt.plot(unique_max_intensities[1:], max_intensity_counts[1:])
plt.xlabel("max intensity")
plt.ylabel("count")
plt.yscale("log")
plt.show()


plt.plot(raw_data.ms1_frames, unique_tofs_per_frame)
plt.xlabel("ms1 frame")
plt.ylabel("unique tof values cnt")
plt.show()


plt.show()

# so now we have frame ordered things by frame, tof, and scan.

# so now we can run max_intensity in groups per frame.
# each time we can simply create a view into seperate (frame,tof) value. And go through scans.


max_intensity_in_window()


# Prooooblem: scans and frames are not ordered at all by tofs: need to order them first.
tof_counts = count1D(tofs)
tof_index = get_index(tof_counts)
tof_order = lexargcountsort(tofs, cumsum(tof_counts))  # 7.38"

scans_frames = make_idx(
    (scans, frames),
    (S, F),
    np.empty(shape=event_count, dtype=get_min_int_data_type(S * F)),
)
scans_frames = scans_frames[tof_order]
tof_scan_frame_order = grouped_argsort(scans_frames, tof_index)


# OK, we now have 2 more orders. what now?


##

dataset = LexSortedDataset.from_tdf(
    folder_dot_d=raw_data,
    level="precursor",
)
print(dataset)

dataset.counts
dataset.index
dataset.columns
dataset.columns.tof

from timstofu.sort_and_pepper import argcountsort3D as lexargcountsort3D
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import lexargcountsort2D
from timstofu.sort_and_pepper import rank_array
from timstofu.stats import get_precumsums

# rename to lexargcountsort3D
# def transpose_frame_scan

counts = dataset.counts.T
index = get_precumsums(counts)

# Now we need what? we have tofs. we need to sort them into (scan,frame) order.


scans, frames, counts = melt(counts)
scans = np.repeat(scans, counts)
frames = np.repeat(frames, counts)

scan_frame_order = lexargcountsort3D(scans, frames, dataset.columns.tof)
assert not is_lex_nondecreasing(scan_frame_order)  # almost impossible, unless empty.


scan_frame_order_invert = np.argsort(scan_frame_order, kind="stable")
scan_frame_order_invert2 = rank_array(scan_frame_order_invert)

np.testing.assert_equal(scan_frame_order_invert, scan_frame_order_invert2)


max_tof = dataset.columns.tof.max()


counts.sum()


counts = precursors.counts

frames, scans, nonzero_counts = melt(counts)
long_frames = np.repeat(frames, nonzero_counts)  # can we avoid that?
long_scans = np.repeat(scans, nonzero_counts)  # can we avoid that?


@njit
def correlate_sparse(tofs, intensities, kernel, results):
    n = len(tofs)
    left = 0
    right = 0
    kernel_size = len(kernel)
    K = kernel_size // 2 - 1

    for i in range(n):
        # Slide window: include all j such that |tofs[j] - tofs[i]| <= K
        while left < n and tofs[i] - tofs[left] > K:
            left += 1
        while right < n and tofs[right] - tofs[i] <= K:
            right += 1

        acc = 0.0
        for j in range(left, right):
            offset = np.int32(tofs[j]) - tofs[i]
            kernel_index = int(round(offset + K))  # shift offset to index in kernel
            if 0 <= kernel_index < kernel_size:
                acc += intensities[j] * kernel[kernel_index]
        result[i] = acc

    return result


@numba.njit
def tof_stripe_maxes(
    counts,
    index,
    tofs,
    intensities,
    window_size=5,
    progress_proxy=None,
    maxes=None,
):
    assert len(tofs) == len(intensities)
    if maxes is None:
        N = len(tofs)
        maxes = np.empty(shape=N, dtype=intensities.dtype)
    frames, scans, cnts = melt(counts)
    for i in numba.prange(len(frames)):
        f = frames[i]
        s = scans[i]
        start = index[f, s]
        end = start + counts[f, s]
        max_intensity_in_window(
            tofs[start:end], intensities[start:end], window_size, maxes[start:end]
        )
        if progress_proxy is not None:
            progress_proxy.update(1)
    return maxes


distinct_frame_scan_cnt = np.count_nonzero(precursors.counts)
tof_maxes = np.empty(len(precursors), dtype=precursors.columns.intensity.dtype)
with ProgressBar(total=distinct_frame_scan_cnt) as progress_proxy:
    tof_stripe_maxes(
        precursors.counts,
        precursors.index,
        precursors.columns.tof,
        precursors.columns.intensity,
        window_size=11,
        progress_proxy=progress_proxy,
        maxes=tof_maxes,
    )

np.sum(precursors.columns.intensity == tof_maxes) / len(tof_maxes)


# np.quantile(nonzero_counts)
# x, y = np.unique(nonzero_counts, return_counts=True)
# plt.plot(x, y)
# plt.show()
np.repeat([1, 2, 3], [4, 5, 6])

arr = np.array([50, 50, 10, 30])
sort_order = np.argsort(arr, stable=True)
reverse_order = np.argsort(sort_order, stable=True)


print("Original:", arr)
print("Sort order:", sort_order)
print("Reverse mapping:", reverse_order)
print("Ranks:", rank_array(sort_order))
print(arr[sort_order])
print(arr[sort_order][reverse_order])
print(arr[sort_order][reverse_order])

rank_array()
