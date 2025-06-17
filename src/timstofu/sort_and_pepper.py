import functools
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from numba_progress import ProgressBar

from timstofu.numba_helper import inputs_series_to_numpy
from timstofu.stats import _count_unique
from timstofu.stats import count2D
from timstofu.stats import count_unique_for_indexed_data
from timstofu.stats import cumsum
from timstofu.stats import get_precumsums

from math import inf


@numba.njit(cache=True)
def _is_nondecreasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev > x:
            return False
        x_prev = x
    return True


is_nondecreasing = inputs_series_to_numpy(_is_nondecreasing)


@numba.njit
def _is_lex_nondecreasing(*arrays: npt.NDArray) -> bool:
    N = len(arrays[0])
    for arr in arrays:
        assert len(arr) == N
    for i in range(1, N):
        for arr in arrays:
            prev = arr[i - 1]
            curr = arr[i]
            if prev < curr:
                break  # go to next array
            elif prev > curr:
                return False  # out of order
            # else: equal, keep comparing next key
    return True


is_lex_nondecreasing = inputs_series_to_numpy(_is_lex_nondecreasing)


def test_is_lex_nondecreasing():
    assert _is_lex_nondecreasing([1, 1, 1, 2], [1, 2, 3, 1])
    assert _is_lex_nondecreasing([1, 1, 1, 2], [1, 2, 2, 1])
    assert not _is_lex_nondecreasing([1, 1, 1, 2], [1, 2, 1, 1])


@numba.njit
def _is_lex_strictly_increasing(*arrays: npt.NDArray) -> bool:
    N = len(arrays[0])
    for arr in arrays:
        assert len(arr) == N
    for i in range(1, N):
        strictly_greater_found = False
        for arr in arrays:
            prev = arr[i - 1]
            curr = arr[i]
            if prev < curr:
                strictly_greater_found = True
                break
            elif prev > curr:
                return False  # strictly decreasing: invalid
        if not strictly_greater_found:
            return False  # tuples are equal: not strictly increasing
    return True


def test_is_lex_strictly_increasing():
    assert _is_lex_strictly_increasing([1, 1, 1, 2], [1, 2, 3, 1])
    assert not _is_lex_strictly_increasing([1, 1, 1, 2], [1, 2, 2, 1])


is_lex_strictly_increasing = inputs_series_to_numpy(_is_lex_strictly_increasing)


@numba.njit(boundscheck=True, cache=True)
def deduplicate(
    sorted_tofs: npt.NDArray,
    sorted_intensities: npt.NDArray,
    nondeduplicated_counts: npt.NDArray,
    deduplicated_event_count: int,
    progress_proxy: ProgressBar | None = None,
):
    dedup_tofs = np.empty(
        dtype=sorted_tofs.dtype,
        shape=deduplicated_event_count,
    )
    dedup_intensities = np.zeros(
        dtype=sorted_intensities.dtype,
        shape=deduplicated_event_count,
    )

    counts_idx = 0
    current_group_count = 0
    dedup_idx = -1
    prev_tof = -inf

    for i, (tof, intensity) in enumerate(zip(sorted_tofs, sorted_intensities)):
        if current_group_count == nondeduplicated_counts[counts_idx]:
            counts_idx += 1
            current_group_count = 0
            prev_tof = -inf  # force top > prev_tof

        if tof > prev_tof:
            dedup_idx += 1
            dedup_tofs[dedup_idx] = tof

        dedup_intensities[dedup_idx] += intensity
        prev_tof = tof
        current_group_count += 1
        if progress_proxy is not None:
            progress_proxy.update(1)

    assert dedup_idx == deduplicated_event_count - 1
    assert counts_idx == len(nondeduplicated_counts) - 1

    return dedup_tofs, dedup_intensities


# this should be wrapped.


@numba.njit(boundscheck=True)
def _lexargcountsort2D(
    xx: npt.NDArray,
    yy: npt.NDArray,
    x_y_to_cumsum: npt.NDArray,
    copy: bool = True,
):
    """Get the order sorting xx and yy lexicographically.

    Arguments:
        xx  (np.array): Array of any subptype of np.integer.
        yy  (np.array): Array of any subptype of np.integer.
        x_y_to_cumsum (np.array): 2D array mapping x,y to cumulated sum of occurrences of x,y in xx and yy.
    """
    assert len(xx.shape) == 1
    assert xx.shape == yy.shape
    assert len(x_y_to_cumsum.shape) == 2
    if copy:
        x_y_to_cumsum = x_y_to_cumsum.copy()
    output = np.empty(
        dtype=x_y_to_cumsum.dtype,
        shape=xx.shape,
    )
    # going backwards preserves input order in (frame,scan) groups
    for i in range(len(xx) - 1, -1, -1):
        x = xx[i]
        y = yy[i]
        # do not remove this cast: uint-int->float in numpy.
        x_y_to_cumsum[x, y] -= np.uint32(1)
        idx = x_y_to_cumsum[x, y]
        assert idx >= 0, "Circular boom!!!"
        assert idx < len(xx), "idx beyond scope"
        output[idx] = i
    return output


@functools.wraps(_lexargcountsort2D)
@inputs_series_to_numpy
def lexargcountsort2D(xx: npt.NDArray, yy: npt.NDArray, *args):
    assert np.issubdtype(xx.dtype, np.integer)
    assert np.issubdtype(yy.dtype, np.integer)
    return _lexargcountsort2D(xx, yy, *args)


def test_lexargcountsort2D():
    frames = np.array([1, 1, 2, 2, 3])
    scans = np.array([1, 1, 3, 4, 2])

    rev_frames = frames[::-1]
    rev_scans = scans[::-1]

    frame_scan_to_count, *frame_scan_ranges = count2D(rev_frames, rev_scans)
    our_order = lexargcountsort2D(
        rev_frames,
        rev_scans,
        cumsum(frame_scan_to_count),
    )
    expected_order = np.lexsort((rev_scans, rev_frames))
    # np.testing.assert_array_equal(our_order, expected_order)
    # this here won't work: np.lexsort is not stable.
    # need to evaluate on test data.
    for xx in (frames, scans):
        np.testing.assert_array_equal(xx[our_order], xx[expected_order])


@numba.njit(boundscheck=True)
def lexargcountsort2D_to_3D(
    xy_to_first_idx: npt.NDArray,
    xy_to_count: npt.NDArray,
    xy_presorted_zz: npt.NDArray,
    xy_order: npt.NDArray,
    copy: bool = True,
) -> npt.NDArray:
    """Complete the sort."""
    xy_to_xyz_order = (
        np.empty(
            dtype=xy_to_first_idx.dtype,
            shape=xy_presorted_zz.shape,
        )
        if copy
        else xy_order
    )
    for x, y in zip(*xy_to_count.nonzero()):
        s = xy_to_first_idx[x, y]
        e = s + xy_to_count[x, y]
        order_z_between_s_and_e = np.argsort(xy_presorted_zz[s:e])
        xy_to_xyz_order[s:e] = xy_order[s:e][order_z_between_s_and_e]
    return xy_to_xyz_order


@inputs_series_to_numpy
def argcountsort3D(
    xx: npt.NDArray | pd.Series,
    yy: npt.NDArray | pd.Series,
    zz: npt.NDArray | pd.Series,
    return_counts: bool = False,
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Get the order sorting any array as long as the provided integer arrays lexicographically by xx, yy, and zz.

    Parameters
    ----------
    xx (np.array): A 1D array.
    yy (np.array): A 1D array.
    zz (np.array): A 1D array.
    return_counts (bool): Return also xy2count.

    Returns
    ```````
    np.array|tuple[np.array,np.array,np.array]: the reindexing needed to establish the lexicogrpahical order or that and counts and index arrays.

    Notes
    -----
    If order is the results, then tuples zip(xx[order], yy[order], zz[order]) will be lexicographically sorted (nondecreasing).
    """
    xy2count, *_ = count2D(xx, yy)
    xy2first_idx = get_precumsums(xy2count)
    xy_order = lexargcountsort2D(
        xx,
        yy,
        xy2first_idx + xy2count,
        False,
    )
    xyz_order = lexargcountsort2D_to_3D(
        xy2first_idx,
        xy2count,
        zz[xy_order],
        xy_order,
        False,
    )
    if return_counts:
        return xyz_order, xy2count, xy2first_idx
    return xyz_order


def test_argcountsort3D():
    xx = np.array([2, 2, 2, 1, 1, 2, 2, 3, 3])
    yy = np.array([2, 2, 2, 5, 4, 3, 4, 2, 1])
    zz = np.array([3, 2, 12, 355, 424, 23, 4, 2, 1])

    order = argcountsort3D(xx, yy, zz)
    expected_order = np.lexsort((zz, yy, xx))

    for tt in (xx, yy, zz):
        np.testing.assert_array_equal(
            tt[order],
            tt[expected_order],
        )


def test_count_unique_for_indexed_data():
    xx = np.array([1, 1, 1, 1, 2, 2, 2])
    yy = np.array([2, 1, 2, 1, 1, 2, 1])
    zz = np.array([2, 1, 2, 2, 1, 2, 1])

    order, xy2count, xy2first_idx = argcountsort3D(xx, yy, zz, return_counts=True)
    assert is_lex_nondecreasing(xx[order], yy[order], zz[order])

    mega_cast = lambda type: lambda xx: tuple(map(type, xx))
    res = set(
        map(
            mega_cast(int),
            zip(
                *count_unique_for_indexed_data(
                    zz[order], xy2count, xy2first_idx
                ).nonzero()
            ),
        )
    )
    expected_res = set(map(mega_cast(int), zip(xx, yy)))
    assert res == expected_res
