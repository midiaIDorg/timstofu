import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numba_progress import ProgressBar
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Any
from typing import Callable


from timstofu.numba_helper import divide_indices
from timstofu.numba_helper import get_min_int_data_type
from timstofu.stats import count1D
from timstofu.stats import get_index


@numba.njit
def do_nothing(center_idx, *args):
    return None


# TODO: make xy2idx on the flight: save on RAM
@numba.njit(parallel=True, boundscheck=True)
def moving_window(
    xx: NDArray,
    yy: NDArray,
    zz: NDArray,
    x_tol: int,
    y_tol: int,
    z_tol: int,
    chunks: NDArray,
    xy2idx: NDArray,
    progress: ProgressBar,
    stencil_foo: Callable = do_nothing,
    stencil_foo_args: tuple[Any, ...] = (),
    center_foo: do_nothing = do_nothing,
    center_foo_args: tuple[Any, ...] = (),
    scratchpad: NDArray = np.array([], int),
):
    MIN_X = MIN_Y = MIN_Z = np.intp(0)
    MAX_X = np.intp(xy2idx.shape[0])
    MAX_Y = np.intp(xy2idx.shape[1] - 1)

    for chunk_idx in numba.prange(len(chunks)):
        chunk_start, chunk_end = chunks[chunk_idx]
        stencil_scratch = scratchpad.copy()

        for center_idx in range(chunk_start, chunk_end):
            X = np.intp(xx[center_idx])
            Y = np.intp(yy[center_idx])
            Z = np.intp(zz[center_idx])

            min_x = max(X - x_tol, MIN_X)
            max_x = min(X + x_tol + 1, MAX_X)
            min_y = max(Y - y_tol, MIN_Y)
            max_y = min(Y + y_tol + 1, MAX_Y)
            min_z = Z - z_tol
            max_z = Z + z_tol

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    s_idx = xy2idx[x, y]
                    e_idx = xy2idx[x, y + 1]
                    # TODO: if x,y the same as previously, simply reuse idx of last smallest z.
                    # but that would be non-trivial to implement.
                    for stencil_idx in range(s_idx, e_idx):
                        # linear search: given low occupation of tof-urt cells,
                        # PROBABLY faster than doing binary search. DEFINITELY SIMPLER.
                        z = zz[stencil_idx]
                        if z > max_z:
                            break
                        if z >= min_z:  # call foo only on nonzero intensities
                            stencil_foo(
                                center_idx,
                                stencil_idx,
                                stencil_scratch,
                                *stencil_foo_args,
                            )
            center_foo(center_idx, stencil_scratch, *center_foo_args)

        if progress is not None:
            progress.update(chunk_end - chunk_start)


@numba.njit
def _get_local_counts_maxes_sums(
    center_idx,
    stencil_idx,
    stencil_scratch,
    intensities,
    counts,
    maxes,
    sums,
):
    counts[center_idx] += 1
    stencil_intensity = intensities[stencil_idx]
    maxes[center_idx] = max(maxes[center_idx], stencil_intensity)
    sums[center_idx] += stencil_intensity


def get_local_counts_maxes_sums(
    xx, yy, zz, intensities, radius_x, radius_y, radius_z, xy2idx, **kwargs
):
    assert len(xx) == len(yy)
    assert len(xx) == len(zz)
    assert len(xx) == len(intensities)
    assert radius_x > 0
    assert radius_y > 0
    assert radius_z > 0

    events_cnt = len(xx)
    chunks = divide_indices(events_cnt)
    max_intensity = intensities.max()

    neighbor_stats = DotDict(
        counts=np.zeros(events_cnt, get_min_int_data_type(max_intensity)),
        maxes=np.zeros(events_cnt, get_min_int_data_type(max_intensity)),
        sums=np.zeros(events_cnt, get_min_int_data_type(max_intensity)),
    )
    with ProgressBar(
        total=events_cnt,
        desc=f"Getting stats in window with radii: x={radius_x} y={radius_y} z ={radius_z}",
        **kwargs,
    ) as progress:
        moving_window(
            xx,
            yy,
            zz,
            radius_x,
            radius_y,
            radius_z,
            chunks,
            xy2idx,
            progress,
            _get_local_counts_maxes_sums,
            (intensities, *neighbor_stats.values()),
        )

    return neighbor_stats


def assert_local_counts_maxes_sums_are_as_with_direct_calculation(
    neighbor_stats: DotDict[str, NDArray],
    tofs: NDArray,
    urts: NDArray,
    scans: NDArray,
    intensities: NDArray,
    radius_tof: int,
    radius_urt: int,
    radius_scan: int,
    number_of_random_sampled_events: int = 10_000,
):
    """Used as test."""
    assert len(tofs) == len(urts)
    assert len(tofs) == len(scans)
    assert len(tofs) == len(intensities)
    assert radius_tof > 0
    assert radius_urt > 0
    assert radius_scan > 0

    events_cnt = len(tofs)
    indices_of_random_events = np.sort(
        np.random.choice(events_cnt, size=number_of_random_sampled_events)
    )
    random_calculated = DotDict(
        {c: arr[indices_of_random_events] for c, arr in neighbor_stats.items()}
    )
    tof_counts = count1D(tofs)
    tof_index = get_index(tof_counts)
    dpd = pd.DataFrame(
        dict(tof=tofs, urt=urts, scan=scans, intensity=intensities), copy=False
    )
    expected = DotDict(
        maxes=np.zeros(len(indices_of_random_events), np.uint64),
        sums=np.zeros(len(indices_of_random_events), np.uint64),
        counts=np.zeros(len(indices_of_random_events), np.uint64),
    )
    for i, idx in enumerate(
        tqdm(indices_of_random_events, desc="Comparing naive to fast calculation")
    ):
        TOF = tofs[idx]
        URT = urts[idx]
        SCAN = scans[idx]
        INTENSITY = intensities[idx]
        MIN_IDX = tof_index[max(TOF - radius_tof, 0)]
        MAX_IDX = tof_index[TOF + radius_tof + 1]
        tof_local_df = dpd.iloc[MIN_IDX:MAX_IDX].query(
            f"abs({TOF}-tof) <= {radius_tof} and abs({URT}-urt) <= {radius_urt} and abs({SCAN}-scan) <= {radius_scan}"
        )
        assert random_calculated.counts[i] == len(tof_local_df)
        assert random_calculated.maxes[i] == tof_local_df.intensity.max()
        assert random_calculated.sums[i] == tof_local_df.intensity.sum()
