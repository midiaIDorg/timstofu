"""
%load_ext autoreload
%autoreload 2
"""
# %%time
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
from tqdm import tqdm

# from timstofu.math import moving_window
from timstofu.math import _minmax
from timstofu.math import bit_width
from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import merge_intervals
from timstofu.misc import filtering_str
from timstofu.misc import get_max_count
from timstofu.misc import iter_array_splits
from timstofu.misc import matrix_to_data_dict
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
from timstofu.stats import counts2index
from timstofu.stats import fill2Dcounts
from timstofu.stats import get_index
from timstofu.timstofmisc import deduce_shift_and_spacing

paranoid = False
discretize_intensity = False
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
radii = DotDict(tof=1, scan=20, urt=10)

# TODO: ask Michal to allow to get the order we want: will save only a bit of RAM.
data_handler = OpenTIMS(folder_dot_d)

D = data_handler[data_handler.ms1_frames]
assert is_lex_increasing(D, strictly=False), "Data is not urt-scan-tof sorted."

ms1_shift, ms1_spacing = deduce_shift_and_spacing(data_handler.ms1_frames)
D[:, 0] -= ms1_shift
D[:, 0] //= ms1_spacing
if discretize_intensity:
    discretize(D[:, 4], D[:, 4], transform=log2)

maxes = DotDict(zip(["urt", "scan", "tof", "intensity"], D.max(axis=0)))
tof_counts = count1D(D[:, 2])
# tof_index = get_index(tof_counts)

##### SORTING: data was frame-scan-tof ordered and our sort is stable, so ...
tof_frame_scan_order = argcountsort(D[:, 2], tof_counts)

D[:, 3] = D[tof_frame_scan_order, 3]  # satellite data: can stay where it is
D[:, 2] = D[tof_frame_scan_order, 1]  # moving scans down
D[:, 1] = D[tof_frame_scan_order, 0]  # moving urts down
repeat(counts=tof_counts, results=D[:, 0])  # re-constructing tofs
assert is_lex_increasing(D), "Data was not lex sorted by tof-frame-scan."
# plot_counts(counts.tof)

data = matrix_to_data_dict(D, ["tof", "urt", "scan", "intensity"])
tof_urt = np.zeros(
    (maxes.tof + 1, maxes.urt + 1 + 1),  # +1 to avoid X[len(X)], +1 to allow index.
    get_min_int_data_type(len(D)),
)
fill2Dcounts(data.tof, data.urt, tof_urt)

perc_of_urt_tof_occupied = round(
    np.count_nonzero(tof_urt) / np.prod(tof_urt.shape) * 100, 1
)
print(f"The tof-urt counts are occupied at {perc_of_urt_tof_occupied}%.")
tof_urt_size, tof_urt_size_cnt = np.unique(tof_urt, return_counts=True)

counts2index(tof_urt)
chunks = divide_indices(len(D))


# xy2idx = tof_urt
# X_RADIUS, Y_RADIUS, Z_RADIUS = radii.tof, radii.urt, radii.scan


# TODO: make xy2idx on the flight: save on RAM
@numba.njit(parallel=True, boundscheck=True)
def moving_widow(D, chunks, xy2idx, radii, foo, foo_args=(), progress=None):
    X_RADIUS, Y_RADIUS, Z_RADIUS = radii
    MIN_X = np.intp(0)
    MIN_Y = np.intp(0)
    MIN_Z = np.intp(0)
    MAX_X = np.intp(xy2idx.shape[0])
    MAX_Y = np.intp(xy2idx.shape[1] - 1)
    XX = D[:, 0]
    YY = D[:, 1]
    ZZ = D[:, 2]
    INTENSITIES = D[:, 3]

    for chunk_idx in numba.prange(len(chunks)):
        chunk_start, chunk_end = chunks[chunk_idx]

        for center_idx in range(chunk_start, chunk_end):
            X = np.intp(XX[center_idx])
            Y = np.intp(YY[center_idx])
            Z = np.intp(ZZ[center_idx])
            I = np.intp(INTENSITIES[center_idx])

            min_x = max(X - X_RADIUS, MIN_X)
            max_x = min(X + X_RADIUS + 1, MAX_X)
            min_y = max(Y - Y_RADIUS, MIN_Y)
            max_y = min(Y + Y_RADIUS + 1, MAX_Y)
            min_z = Z - Z_RADIUS
            max_z = Z + Z_RADIUS

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    s_idx = xy2idx[x, y]
                    e_idx = xy2idx[x, y + 1]
                    # TODO: if x,y the same as previously, simply reuse idx of last smallest z.
                    for stencil_idx in range(s_idx, e_idx):
                        # linear search: given low occupation of tof-urt cells,
                        # PROBABLY faster than doing binary search. DEFINITELY SIMPLER.
                        z = ZZ[stencil_idx]
                        if z > max_z:
                            break
                        if z >= min_z:  # call foo only on nonzero intensities
                            foo(center_idx, stencil_idx, *foo_args)

        if progress is not None:
            progress.update(chunk_end - chunk_start)


@numba.njit
def get_neighbor_stats(center_idx, stencil_idx, D, counts, maxes, sums):
    counts[center_idx] += 1
    stencil_intensity = D[stencil_idx, 3]
    maxes[center_idx] = max(maxes[center_idx], stencil_intensity)
    sums[center_idx] += stencil_intensity


neighbor_stats = DotDict(
    counts=np.zeros(len(D), get_min_int_data_type(maxes.intensity)),
    maxes=np.zeros(len(D), get_min_int_data_type(maxes.intensity)),
    sums=np.zeros(len(D), get_min_int_data_type(maxes.intensity)),
)
with ProgressBar(
    total=len(D),
    desc=f"Getting stats in window {dict(radii)}",
) as progress:
    moving_widow(
        D,
        chunks,
        tof_urt,
        np.array((radii.tof, radii.urt, radii.scan)),
        get_neighbor_stats,
        (D, *neighbor_stats.values()),
        progress,
    )


(neighbor_stats.counts == 0).nonzero()
counts_val, counts_freq = np.unique(neighbor_stats.counts, return_counts=True)

plt.plot(counts_val, counts_freq)
plt.show()

# Now, implement the test comparing direct and indirect calc.

K = 10000
indices_of_random_events = np.sort(np.random.choice(len(D), size=K))
random_calculated = DotDict(
    {c: arr[indices_of_random_events] for c, arr in neighbor_stats.items()}
)
tof_counts = count1D(data.tof)
tof_index = get_index(tof_counts)
dpd = pd.DataFrame(D, copy=False, columns=["tof", "urt", "scan", "intensity"])
expected = DotDict(
    maxes=np.zeros(len(indices_of_random_events), np.uint32),
    sums=np.zeros(len(indices_of_random_events), np.uint32),
    counts=np.zeros(len(indices_of_random_events), np.uint32),
)
for i, idx in enumerate(tqdm(indices_of_random_events)):
    TOF, URT, SCAN, INTENSITY = D[idx]
    MIN_IDX = tof_index[max(TOF - radii.tof, 0)]
    MAX_IDX = tof_index[TOF + radii.tof + 1]
    tof_local_df = dpd.iloc[MIN_IDX:MAX_IDX].query(
        f"abs({TOF}-tof) <= {radii.tof} and abs({URT}-urt) <= {radii.urt} and abs({SCAN}-scan) <= {radii.scan}"
    )
    assert random_calculated.counts[i] == len(tof_local_df)
    assert random_calculated.maxes[i] == tof_local_df.intensity.max()
    assert random_calculated.sums[i] == tof_local_df.intensity.sum()
