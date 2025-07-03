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
from timstofu.pivot import Pivot
from timstofu.stats import count1D
from timstofu.stats import get_index

from timstofu.numba_helper import is_permutation
from timstofu.sort_and_pepper import is_lex_nondecreasing

from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort

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

# provide counts and indices if available.
tof_scan_urt = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
)
tof_scan_urt.is_sorted()
tof_scan_urt.sort()
assert tof_scan_urt.is_sorted()
# tof_scan_urt.argsort()


# scan_frame_events_per_tof = count1D(tof_scan_urt.counts.tof)
# plt.scatter(np.arange(len(scan_frame_events_per_tof)), scan_frame_events_per_tof)
# plt.xscale("log")
# plt.yscale("log")
# plt.show()


# good, now the other thing: the silly willy densification.
#
#
# this takes too long.
@numba.njit(parallel=True)
def nbmap(index, tensor, scans, urts, intensities, progress_proxy):
    for i in numba.prange(len(index) - 1):
        s = index[i]
        e = index[i + 1]
        local_tensor = tensor.copy()
        for j in range(s, e):
            local_tensor[scans[j], urts[j]] = intensities[j]
        if progress_proxy is not None:
            progress_proxy.update(1)


tensor = np.zeros(
    shape=(
        tof_scan_urt.col2max.scan,
        tof_scan_urt.col2max.urt,
    ),
    dtype=np.uint8,
)

# this does not work. Alternative is the direct frame-scan-tof-intensity.
# A small tensor woudld need filling up: so we are back to base 1.


with ProgressBar(
    total=len(tof_scan_urt.counts.tof) - 1,
    desc="Densification test",
) as progress_proxy:
    nbmap(tof_scan_urt.index("tof"), tensor, progress_proxy)

# can we do it differently??? Like how?
from timstofu.stats import _count1D


@numba.njit(parallel=True)
def nbmap(index, scans, progress_proxy, res):
    for i in numba.prange(len(index) - 1):
        s = index[i]
        e = index[i + 1]
        res[i] = np.sum(scans)
        if progress_proxy is not None:
            progress_proxy.update(1)


res = np.empty(shape=len(tof_scan_urt.counts.tof), dtype=np.uint64)
with ProgressBar(
    total=len(tof_scan_urt.counts.tof) - 1,
    desc="Counting scans per tof",
) as progress_proxy:
    nbmap(
        tof_scan_urt.index("tof"),
        tof_scan_urt.extract("scan"),
        progress_proxy,
        res,
    )

# Very inefficient to do calculations using tof dim. why????



# problem is that there are too many allocations compared to how many scans there are.
(frame, scan, tof)?
(frame, tof, scan)?
(scan, tof, frame)?

