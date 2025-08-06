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
from kilograms import scatterplot_matrix
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
from timstofu.windowing import (
    assert_local_counts_maxes_sums_are_as_with_direct_calculation,
)
from timstofu.windowing import get_local_counts_maxes_sums
from timstofu.windowing import moving_window

show_plots = False
paranoid = False
discretize_intensity = False
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
# RADIUS = DotDict(tof=3, urt=10, scan=40)
RADIUS = DotDict(tof=1, urt=10, scan=20)
get_max_count(RADIUS)

# TODO: ask Michal to allow to get the order we want: will save only a bit of RAM.
data_handler = OpenTIMS(folder_dot_d)

D = data_handler[data_handler.ms1_frames]
assert is_lex_increasing(D, strictly=False), "Data is not urt-scan-tof sorted."

ms1_shift, ms1_spacing = deduce_shift_and_spacing(data_handler.ms1_frames)
D[:, 0] -= ms1_shift
D[:, 0] //= ms1_spacing
if discretize_intensity:
    discretize(D[:, 4], D[:, 4], transform=log2)

MAX = DotDict(zip(["urt", "scan", "tof", "intensity"], D.max(axis=0)))
tof_counts = count1D(D[:, 2])
# tof_index = get_index(tof_counts)

##### SORTING: data was frame-scan-tof ordered and our sort is stable, so ...
tof_frame_scan_order = argcountsort(D[:, 2], tof_counts)

D[:, 3] = D[tof_frame_scan_order, 3]  # satellite data: can stay where it is
D[:, 2] = D[tof_frame_scan_order, 1]  # moving scans down
D[:, 1] = D[tof_frame_scan_order, 0]  # moving urts down
repeat(counts=tof_counts, results=D[:, 0])  # re-constructing tofs
assert is_lex_increasing(D), "Data was not lex sorted by tof-frame-scan."
raw = matrix_to_data_dict(D, ["tof", "urt", "scan", "intensity"])
# plot_counts(counts.tof)

tof_urt = np.zeros((MAX.tof + 1, MAX.urt + 2), get_min_int_data_type(len(D)))
fill2Dcounts(raw.tof, raw.urt, tof_urt)
perc_of_urt_tof_occupied = round(
    np.count_nonzero(tof_urt) / np.prod(tof_urt.shape) * 100, 1
)
print(f"The tof-urt counts are occupied at {perc_of_urt_tof_occupied}%.")
tof_urt_size, tof_urt_size_cnt = np.unique(tof_urt, return_counts=True)
counts2index(tof_urt)


neighbor_stats = get_local_counts_maxes_sums(
    raw.tof,
    raw.urt,
    raw.scan,
    raw.intensity,
    RADIUS.tof,
    RADIUS.urt,
    RADIUS.scan,
    tof_urt,
)
assert np.all(neighbor_stats.counts)
counts_val, counts_freq = np.unique(neighbor_stats.counts, return_counts=True)
if show_plots:
    plt.plot(counts_val, counts_freq)
    plt.show()

assert_local_counts_maxes_sums_are_as_with_direct_calculation(
    neighbor_stats,
    raw.tof,
    raw.urt,
    raw.scan,
    raw.intensity,
    RADIUS.tof,
    RADIUS.urt,
    RADIUS.scan,
    number_of_random_sampled_events=10_000,
)


## NOW, how the pltos look like?
neighbor_stats_pd = pd.DataFrame(neighbor_stats, copy=False)
np.unique(raw.intensity == neighbor_stats.maxes, return_counts=True)


res = pd.DataFrame()
res["counts"] = neighbor_stats.counts
res["log_counts"] = np.log10(neighbor_stats.counts)
res["log_TIC"] = np.log10(neighbor_stats.sums)
res["log_max_intensity"] = np.log10(neighbor_stats.maxes)
scatterplot_matrix(res)


# GETTING LOCAL MAXIMA
def get_box_centers(*radius_buffer_max):
    yield from itertools.product(
        *(range(r + 1, m, 2 * (r + b)) for r, b, m in radius_buffer_max)
    )


@numba.njit(parallel=True, boundscheck=True)
def get_argmaxes(
    intensities, xy2index, x_centers, y_centers, top_intense_event_indices
):
    assert len(x_centers) == len(y_centers)
    for i in range(len(x_centers)):
        x = x_centers[i]
        y = y_centers[i]
        s = xy2index[x, y]
        e = xy2index[x, y + 1]
        if e < s:
            top_intense_event_indices[i] = -2
        elif e == s:
            top_intense_event_indices[i] = -1
        else:
            top_intense_event_indices[i] = s + np.argmax(intensities[s:e])
    return top_intense_event_indices


BUFFER = DotDict(tof=10, urt=10)
boxes = pd.DataFrame(
    list(
        get_box_centers(
            (RADIUS.tof, BUFFER.tof, MAX.tof), (RADIUS.urt, BUFFER.urt, MAX.urt)
        )
    ),
    columns=["tof", "urt"],
)
boxes["argmax"] = get_argmaxes(
    D[:, 3],
    tof_urt,
    boxes.tof.to_numpy(),
    boxes.urt.to_numpy(),
    np.empty(len(boxes), get_min_int_data_type(len(D))),
)
nonempty_boxes = boxes[boxes.argmax >= 0].copy()
nonempty_boxes["top_tof"] = D[nonempty_boxes.argmax, 0]
nonempty_boxes["top_urt"] = D[nonempty_boxes.argmax, 1]
nonempty_boxes["top_scan"] = D[nonempty_boxes.argmax, 2]
nonempty_boxes["top_intensity"] = D[nonempty_boxes.argmax, 3]
plt.scatter(
    nonempty_boxes.tof,
    nonempty_boxes.urt,
    s=nonempty_boxes.intensity / nonempty_boxes.intensity.max(),
)
plt.show()

#######


@numba.njit(boundscheck=True)
def fill_marginals(
    center_idx: int,
    stencil_idx: int,
    marginals,
    radii: list[int],
    D: NDArray,
):
    intensity = D[stencil_idx, 4]
    offset = 0
    for i, radius in enumerate(radii):
        j = radii[i] + np.intp(D[stencil_idx, i]) - np.intp(D[center_idx, i])
        marginal[j] += intensity
        offset += radius * 2 + 1


@numba.njit
def analyze_marginals(center_idx, marginals, radii):
    # what to do with the marginals? I don't know yet.
    marginals[:] = 0  # clean-up after each stencil


# chunks = divide_indices(len(D))
