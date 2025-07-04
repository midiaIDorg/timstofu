"""
%load_ext autoreload
%autoreload 2
"""
import itertools
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numba_progress import ProgressBar
from opentimspy import OpenTIMS

from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import moving_window
from timstofu.numba_helper import decount
from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import map_onto_lexsorted_indexed_data
from timstofu.numba_helper import melt
from timstofu.pivot import Pivot
from timstofu.plotting import plot_discrete_marginals
from timstofu.sort import argcountsort
from timstofu.sort import grouped_argsort
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.stats import count1D
from timstofu.stats import count2D_marginals
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

dintensity_counts = np.zeros(dtype=np.int32, shape=256)
dintensity_counts = discretize(intensities, transform=log2)

pivot = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
    dintensity=dintensity_counts,  # perhaps this should go out???
)
pivot.sort()
assert pivot.is_sorted()
radii = dict(
    tof=3,
    scan=20,
    urt=10,
)

# shape = np.append(2 * radii + 1, 2)
# N = 1_000
trabant = pivot["dintensity"]
# chunk_ends = divide_indices(len(pivot), k=16 * 100)
chunk_ends = divide_indices(len(pivot), k=16)


diffs_dct = pivot.get_stencil_diffs(**radii)
diffs = np.array(list(diffs_dct.values()))
# diffs = diffs[diffs >= 0]


# TODO: make it into a method of the Pivot.
@numba.njit(parallel=True, boundscheck=True)
def moving_window(
    chunk_ends,
    diffs,
    data,
    updater,
    updater_args=(),
    densify=False,
    progress_proxy=None,
):
    ONE = np.uintp(1)
    diffs = diffs.astype(np.intp)

    msg = "Chunk ends must be a 2D array of start to end values to iterate over."
    assert len(chunk_ends) > 0, msg
    assert len(chunk_ends.shape) == 2, msg
    if len(chunk_ends) > 1:
        prev_chunk = chunk_ends[0]
        for i in range(1, len(chunk_ends)):
            chunk = chunk_ends[i]
            assert prev_chunk[1] == chunk[0]
            prev_chunk = chunk
    assert chunk_ends[-1, -1] == len(data), "Chunks must divide all of the data."

    diff_starts = diffs[:, 0]
    diff_ends = diffs[:, 1]

    for i in numba.prange(len(chunk_ends)):
        # for i in range(len(chunk_ends)):
        chunk_s, chunk_e = chunk_ends[i]
        window_starts = np.full(len(diffs), chunk_s, data.dtype)  # INDEX
        window_ends = np.full(len(diffs), chunk_s, data.dtype)  # INDEX

        if densify:
            event_results = np.zeros(shape=(5, 5, 5), dtype=np.uintp)

        for c_idx in range(chunk_s, chunk_e):  # INDEX OF THE CURRENT WINDOW'S CENTER
            center_val = np.intp(data[c_idx])  # CENTER VALUE

            if densify:
                event_results[:] = 0  # ZERO OUT RESULTS

            # UPDATE INDEX: REMEMBER DATA IS STRICTLY INCREASING
            for j in range(len(diffs)):
                t_s = center_val + diff_starts[j]  # TARGET START
                t_e = center_val + diff_ends[j]  # TARGET END

                # MOVE START
                while (
                    window_starts[j] < chunk_e and np.intp(data[window_starts[j]]) < t_s
                ):
                    window_starts[j] += ONE

                # MOVE END
                window_ends[j] = max(window_starts[j], window_ends[j])
                while window_ends[j] < chunk_e and np.intp(data[window_ends[j]]) <= t_e:
                    window_ends[j] += ONE

            # UPDATE RESULTS
            for window_start, window_end in zip(window_starts, window_ends):
                if window_start < window_end:  # RUN UPDATER ONLY ON NON-EMPTY WINDOWS
                    updater(
                        c_idx,  # BASICALLY, WHERE TO WRITE RESULTS
                        window_start,
                        window_end,  # WINDOW TO ITERATE OVER
                        *updater_args,  # OTHER ARGUMENTS AND RESULT ARRAYS
                    )

        if progress_proxy is not None:
            progress_proxy.update(chunk_e - chunk_s)


@numba.njit(boundscheck=True)
def local_max_sum_count(
    current_idx,  # those are
    start_idx,  # essentially
    end_idx,  # pointers to data
    intensities,
    maxes,
    sums,
    counts,
):
    maxes[current_idx] = max(maxes[current_idx], intensities[start_idx:end_idx].max())
    counts[current_idx] += end_idx - start_idx
    sums[current_idx] += intensities[start_idx:end_idx].sum()


updater = local_max_sum_count
updater_results = DotDict(
    maxes=np.zeros(len(pivot), trabant.dintensity.dtype),
    sums=np.zeros(len(pivot), np.uint32),
    counts=np.zeros(
        len(pivot),
        get_min_int_data_type(np.prod(np.array(list(radii.values())) * 2 + 1)),
    ),
)

# chunk_ends2 = np.array([[ 0, len(pivot) ]])
with ProgressBar(
    total=len(pivot),
    desc=f"Getting stats in window {radii}",
) as progress_proxy:
    moving_window(
        chunk_ends,
        diffs,
        pivot.array,
        updater,
        (trabant.dintensity, *tuple(updater_results.values())),
        False,
        progress_proxy,
    )


updater_results.counts.max()
w = count1D(updater_results.counts)

plt.scatter(np.arange(len(w)), w)
plt.yscale("log")
plt.xlabel(
    f"# Events in box with radii {radii}. Max events = {math.prod(r*2+1 for r in radii.values())}"
)
plt.ylabel("COUNT")
plt.show()


from matplotlib.colors import LogNorm

updater_results_df = pd.DataFrame(updater_results, copy=False)
marginals = count2D_marginals(updater_results_df)
plot_discrete_marginals(
    marginals,
    imshow_kwargs=dict(norm=LogNorm()),
)


np.unique(updater_results.maxes == trabant.dintensity, return_counts=True)


updater_results.counts
updater_results.sums - trabant.dintensity


def within_box(event1, event2, radii):
    event1 = np.asarray(event1, dtype=int)
    event2 = np.asarray(event2, dtype=int)
    radii = np.asarray(radii, dtype=int)

    return np.all(np.abs(event2 - event1) <= radii)


def test_moving_window():
    full_data = np.array(
        list(itertools.product(range(3), range(2, 6), range(4))), dtype=np.uintp
    )
    x = full_data[:, 0]
    y = full_data[:, 1]
    z = full_data[:, 2]

    intensities = np.arange(len(z))

    maxes = np.array((10, 10, 10))
    pivot = Pivot.new(x=x, y=y, z=z, _maxes=maxes)
    np.testing.assert_equal(
        pivot.array,
        (full_data @ ((np.cumprod(maxes) / 10)[::-1]).astype(int)).astype(int),
    )

    # chunk_ends = divide_indices(len(pivot), k=16)
    chunk_ends = np.array([[0, len(z)]])

    radii = dict(x=1, y=1, z=1)
    diffs_dct = pivot.get_stencil_diffs(**radii)
    diffs = np.array(list(diffs_dct.values()))
    results = DotDict(
        maxes=np.zeros(len(pivot), intensities.dtype),
        sums=np.zeros(len(pivot), np.uint32),
        counts=np.zeros(
            len(pivot),
            get_min_int_data_type(np.prod(np.array(list(radii.values())) * 2 + 1)),
        ),
    )

    data = pivot.array
    updater = local_max_sum_count
    with ProgressBar(
        total=len(pivot),
        desc=f"Getting stats in window {radii}",
    ) as progress_proxy:
        moving_window(
            chunk_ends,
            diffs,
            data,
            updater,
            (intensities, *tuple(results.values())),
            False,  # density
            progress_proxy,
        )

    # direct inefficient window computations
    RADII = np.array(list(radii.values()))
    expected = DotDict(maxes=[], sums=[], counts=[])
    for row_A in full_data:
        events_in_box_A = np.array(
            [within_box(row_A, row_B, RADII) for row_B in full_data]
        )
        local_intensities = intensities[events_in_box_A]
        expected.maxes.append(local_intensities.max())
        expected.sums.append(local_intensities.sum())
        expected.counts.append(len(local_intensities))
    expected = DotDict({c: np.array(v) for c, v in expected.items()})

    assert set(expected) == set(results)

    for c in expected:
        assert np.all(expected[c] == results[c]), f"`{c}` not the same."
