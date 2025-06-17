import functools
import math

import numba
import numpy as np
import numpy.typing as npt

from numba_progress import ProgressBar

from timstofu.numba_helper import inputs_series_to_numpy
from timstofu.numba_helper import zeros_copy


# optimization: tabulate erf differences.
@numba.njit
def trivial_normal_cdf(x, mu=0.0, sigma=1.0):
    """
    Compute the CDF of the normal distribution for given values.

    This is chapgpt generated. I would never write that much of useless documentation...

    Parameters:
    x : array_like
        Points at which to evaluate the CDF.
    mu : float, optional
        Mean of the normal distribution (default is 0.0).
    sigma : float, optional
        Standard deviation of the normal distribution (default is 1.0).

    Returns:
    array_like
        CDF values corresponding to x.
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def discrete_hist(arr) -> tuple[npt.NDArray]:
    """Make a discrete histogram data.

    Returns:
        tuple: sorted values and their counts."""
    unique, counts = np.unique(arr, return_counts=True)
    return unique, counts


@numba.njit
def _count_sorted(xx: npt.NDArray) -> npt.NDArray:
    """Count elements of a sorted array.

    Arguments:
        xx: Array of ints.

    Returns:
        npt.NDArray: Array of counts.
    """
    counts = np.zeros(shape=xx[-1] + 1, dtype=np.uint32)
    for x in xx:
        counts[x] += 1
    return counts


count_sorted = functools.wraps(_count_sorted)(inputs_series_to_numpy(_count_sorted))


@numba.njit
def _minmax(xx: npt.NDArray, *args):
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


minmax = functools.wraps(_minmax)(inputs_series_to_numpy(_minmax))


@numba.njit
def _count1D(xx: npt.NDArray) -> npt.NDArray:
    """Count elements in an array of ints.

    Arguments:
        xx: Array of ints.

    Returns:
        tuple: two arrays: range of elements and their counts
    """
    min_x, max_x = _minmax(xx)
    min_x = int(min_x)
    max_x = int(max_x)
    _counts = np.zeros(shape=max_x + 1 - min_x, dtype=np.uint32)
    for x in xx:
        _counts[x - min_x] += 1
    _range = np.arange(min_x, max_x + 1)
    return _range, _counts


count1D = functools.wraps(_count1D)(inputs_series_to_numpy(_count1D))


@numba.njit(boundscheck=True)
def _count2D(
    xx: npt.NDArray,
    yy: npt.NDArray,
    dtype: type = np.uint64,
) -> tuple[npt.NDArray, float | int, float | int, float | int, float | int]:
    """
    Count 2D stats of occurrences of tuples (x,y).
    """
    assert len(xx) == len(yy)
    min_x, max_x = _minmax(xx)
    min_y, max_y = _minmax(yy)
    cnts = np.zeros(dtype=np.uint64, shape=(max_x + 1, max_y + 1))

    for i in range(len(xx)):
        cnts[xx[i], yy[i]] += 1

    return cnts, min_x, max_x, min_y, max_y


count2D = functools.wraps(_count1D)(inputs_series_to_numpy(_count2D))


@numba.njit
def cumsum(xx):
    return np.cumsum(xx).reshape(xx.shape)


@numba.njit
def get_precumsums(counts: npt.NDArray) -> npt.NDArray:
    """
    Compute the cumulative sum of entries in a 2D array up to (but not including) each element,
    using lexicographic (row-major) order.

    Parameters
    ----------
    counts : np.ndarray
        A 2D NumPy array of counts.

    Returns
    -------
    np.ndarray
        A 2D array of the same shape as `counts`, where each element (i, j) contains
        the sum of all elements that come before (i, j) in row-major order.
    """
    cumsum = np.cumsum(counts.flatten())
    precumsum = np.roll(cumsum, 1)
    precumsum[0] = 0
    return precumsum.reshape(counts.shape)


@functools.wraps(get_precumsums)
@numba.njit
def alt_get_precumsums(counts: npt.NDArray) -> npt.NDArray:
    return np.cumsum(counts).reshape(counts.shape) - counts


@numba.njit(boundscheck=True)
def _count_unique(sorted_xx: npt.NDArray) -> int:
    """Count unique occurrences of a sorted array.

    Arguments:
        sorted_xx (np.array): Sorted numbers.

    Returns:
        int: Number of unique elements in the sorted input.
    """
    cnt = 0
    x_prev = math.inf
    for x in sorted_xx:
        cnt += x != x_prev
        x_prev = x
    return cnt


count_unique = functools.wraps(_count_unique)(inputs_series_to_numpy(_count_unique))


@numba.njit(boundscheck=True)
def count_unique_for_indexed_data(
    zz_lexsorted: npt.NDArray,
    counts: npt.NDArray,
    index: npt.NDArray,
    progress_proxy: ProgressBar | None = None,
) -> npt.NDArray:
    """Count the number of unique (x,y,z) tuples in a sorted order with counts and index."""
    unique_counts = zeros_copy(counts)
    for x, y in zip(*counts.nonzero()):
        first_idx = index[x, y]
        last_idx = first_idx + counts[x, y]
        unique_counts[x, y] = _count_unique(zz_lexsorted[first_idx:last_idx])
        if progress_proxy is not None:
            progress_proxy.update(1)
    return unique_counts
