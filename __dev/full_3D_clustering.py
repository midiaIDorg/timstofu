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
# urt_tof_scan = Pivot.new(
#     urt=urts,
#     tof=tofs,
#     scan=scans,
# )
# urt_tof_scan.is_sorted()
# urt_tof_scan.sort()
# assert urt_tof_scan.is_sorted()


tof_scan_urt = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
)
tof_scan_urt.sort()
assert tof_scan_urt.is_sorted()

# shape = np.append(2 * radii + 1, 2)
# N = 1_000
weights = intensities
data = tof_scan_urt.array

radii = dict(tof=1, scan=2, urt=2)
chunk_ends = divide_indices(len(tof_scan_urt))


diffs_dct = tof_scan_urt.get_stencil_diffs(tof=2, scan=2, urt=2)
diffs = np.array(list(diffs_dct.values()))
# diffs = diffs[diffs >= 0]


@numba.njit(parallel=True)
def nbmap(
    chunk_ends,
    diffs,
    data,
    # results_updater,
    # results_updater_args,
    progress_proxy,
):
    ONE = np.uintp(1)
    diffs = diffs.astype(np.intp)
    assert chunk_ends[-1, -1] == len(data)
    diff_starts = diffs[:, 0].copy()
    diff_ends = diffs[:, 1].copy()

    for i in numba.prange(len(chunk_ends)):
        chunk_s, chunk_e = chunk_ends[i]
        window_starts = np.full(len(diffs), chunk_s, data.dtype)  # INDEX
        window_ends = np.full(len(diffs), chunk_s, data.dtype)  # INDEX

        for c_idx in range(chunk_s, chunk_e):  # INDEX OF THE CURRENT WINDOW'S CENTER
            center_val = np.intp(data[c_idx])  # CENTER VALUE
            # UPDATE INDEX: REMEMBER DATA IS STRICTLY INCREASING
            for j in range(len(diffs)):
                t_s = center_val + diff_starts[j]  # TARGET START
                t_e = center_val + diff_ends[j]  # TARGET END

                # MOVE START
                while (
                    window_starts[j] < c_idx and np.intp(data[window_starts[j]]) < t_s
                ):
                    window_starts[j] += ONE

                # MOVE END
                window_ends[j] = max(window_starts[j], window_ends[j])
                while window_ends[j] < chunk_e and np.intp(data[window_ends[j]]) <= t_e:
                    window_ends[j] += ONE

            # UPDATE RESULTS
            # for stencil_s, stencil_e in diffs:
            #     pass
            #     # results_updater(c_idx, stencil_idx, _I, *results_updater_args)

        progress_proxy.update(chunk_e - chunk_s)


with ProgressBar(
    total=len(tof_scan_urt),
    desc="Counting scans per tof",
) as progress_proxy:
    nbmap(
        chunk_ends,
        diffs,
        data,
        # results_updater,
        # tuple(*results_updater_results),
        progress_proxy,
    )
