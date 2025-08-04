import functools
import itertools
import math

import numba
import numpy as np

from numba_progress import ProgressBar
from numpy.typing import NDArray

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


def discrete_hist(arr) -> tuple[NDArray]:
    """Make a discrete histogram data.

    Returns:
        tuple: sorted values and their counts."""
    unique, counts = np.unique(arr, return_counts=True)
    return unique, counts


@numba.njit
def _count_sorted(xx: NDArray) -> NDArray:
    """Count elements of a sorted array.

    Arguments:
        xx: Array of ints.

    Returns:
        NDArray: Array of counts.
    """
    counts = np.zeros(shape=xx[-1] + 1, dtype=np.uint32)
    for x in xx:
        counts[x] += 1
    return counts


count_sorted = functools.wraps(_count_sorted)(inputs_series_to_numpy(_count_sorted))


@numba.njit
def _minmax(xx: NDArray, *args):
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


minmax = functools.wraps(_minmax)(inputs_series_to_numpy(_minmax))


@numba.njit
def _count1Dsparse(xx: NDArray) -> NDArray:
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


count1Dsparse = functools.wraps(_count1Dsparse)(inputs_series_to_numpy(_count1Dsparse))


@numba.njit(boundscheck=True)
def _count1D(xx: NDArray, counts: NDArray) -> NDArray:
    """Count elements in an array of ints.

    Arguments:
        xx: Array of ints.

    Returns:
        np.array: Counts.
    """
    one_but_not_the_same = np.uintp(1)
    for x in xx:
        counts[x] += one_but_not_the_same
    return counts


@functools.wraps(_count1D)
@inputs_series_to_numpy
def count1D(xx: NDArray, counts: NDArray | None = None):
    if counts is None:
        counts = np.zeros(shape=int(xx.max()) + 1, dtype=np.uint32)
    return _count1D(xx, counts)


@numba.njit(boundscheck=True)
def fill2Dcounts(
    xx: NDArray,
    yy: NDArray,
    counts: NDArray,
) -> None:
    """
    Count 2D stats of occurrences of tuples (x,y).
    """
    assert len(xx) == len(yy)
    for i in range(len(xx)):
        counts[xx[i], yy[i]] += 1


@numba.njit(boundscheck=True)
def _count2D(
    xx: NDArray,
    yy: NDArray,
    dtype: type = np.uint64,
) -> tuple[NDArray, float | int, float | int, float | int, float | int]:
    """
    Count 2D stats of occurrences of tuples (x,y).
    """
    assert len(xx) == len(yy)
    min_x, max_x = _minmax(xx)
    min_y, max_y = _minmax(yy)
    cnts = np.zeros(dtype=dtype, shape=(max_x + 1, max_y + 1))

    for i in range(len(xx)):
        cnts[xx[i], yy[i]] += 1

    return cnts, min_x, max_x, min_y, max_y


count2D = functools.wraps(_count1D)(inputs_series_to_numpy(_count2D))


def count2D_marginals(df):
    """Compute 2D count tables (co-occurrence matrices) for all pairs of columns in a DataFrame.

    For each unique pair of columns in the input DataFrame, this function computes a 2D
    contingency table using the `count2D` function, which should return a 2D array or
    similar structure representing the frequency of joint occurrences of values.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing categorical or discrete data. All column pairs will be used
        to compute joint counts.

    Returns
    -------
    dict[tuple[str, str], Any]
        A dictionary mapping each pair of column names `(col_A, col_B)` to the result of
        `count2D(df[col_A], df[col_B])`. The output format of the values depends on the
        implementation of `count2D`.
    """
    return {
        (col_A, col_B): count2D(df[col_A], df[col_B])
        for col_A, col_B in itertools.combinations(df, 2)
    }


@numba.njit
def cumsum(xx):
    return np.cumsum(xx).reshape(xx.shape)


# TODO: There should be one function. We should not use counts, but enlarge the index by 1 in every dim to make better use of RAM.
@numba.njit
def get_precumsums(counts: NDArray) -> NDArray:
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
def alt_get_precumsums(counts: NDArray) -> NDArray:
    return np.cumsum(counts).reshape(counts.shape) - counts


@numba.njit(boundscheck=True)
def _count_unique(sorted_xx: NDArray) -> int:
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
    zz_lexsorted: NDArray,
    counts: NDArray,
    index: NDArray,
    unique_counts: NDArray | None = None,
    progress_proxy: ProgressBar | None = None,
) -> NDArray:
    """Count the number of unique (x,y,z) tuples in a sorted order with counts and index."""
    if unique_counts is None:
        unique_counts = zeros_copy(counts)
    for x, y in zip(*counts.nonzero()):
        first_idx = index[x, y]
        last_idx = first_idx + counts[x, y]
        unique_counts[x, y] = _count_unique(zz_lexsorted[first_idx:last_idx])
        if progress_proxy is not None:
            progress_proxy.update(1)
    return unique_counts


@numba.njit(boundscheck=True)
def get_index(counts: NDArray, index: NDArray | None = None) -> NDArray:
    """Turn counts into cumulated sums offset by one 0 at the beginning.

    This function should be used to create an indexed view of elements in another table where elements are in groups and counts summarizes how many times they occur in those groups.
    See test_get_index.

    Arguments:
        counts (NDArray): An array of counts.
        index (NDArray|None): Optional place to store index.

    Returns:
        np.array: A table with 0 and then cumulated counts.
    """
    if index is None:
        index = np.empty(shape=len(counts) + 1, dtype=np.uint32)
    assert len(index) == len(counts) + 1
    index[0] = 0
    i = 1
    for cnt in counts:
        index[i] = index[i - 1] + cnt
        i += 1
    return index


def test_get_index():
    counts = np.array([1, 2, 5, 60])
    cumsum_idx = get_index(counts)
    np.testing.assert_equal(cumsum_idx[1:], np.cumsum(counts))


# @numba.njit(boundscheck=True)
# def get_window_borders(i, xx, radius, left=0, right=0):
#     """
#     Parameters
#     ----------
#     i (int): Current index in xx.
#     xx (np.array): A sorted array (non-decrasing).
#     left (int): Currently explored left end.
#     right (int): Currently explored right end.
#     radius (int): A positive integer.

#     Returns
#     -------
#     tuple: updated values of left and right.
#     """
#     N = len(xx)
#     x = np.intp(xx[i])
#     # Move left pointer to ensure xx[i] - xx[left] <= radius
#     while left < N and x - xx[left] > radius:
#         left += 1
#     right = max(i, right)
#     # Move right pointer to ensure xx[right] - xx[i] <= radius
#     while right < N and xx[right] - x <= radius:
#         right += 1
#     return left, right


@numba.njit(boundscheck=True)
def get_window_borders(
    i: int,
    i_max: int,
    xx: NDArray,
    radius: int,
    left: int = 0,
    right: int = 0,
):
    """
    Parameters
    ----------
    i (int): Current index in xx.
    i_max (int): The max value of i+j can take (necessary not to go into another group in xx).
    xx (np.array): A sorted array (non-decrasing).
    left (int): Currently explored left end.
    right (int): Currently explored right end.
    radius (int): A positive integer.

    Returns
    -------
    tuple: updated values of left and right.
    """
    x = xx[i]
    # Move left pointer to ensure xx[i] - xx[left] <= radius
    while left < i and xx[left] + radius < x:
        left += 1
    right = max(i, right)
    max_right = min(i_max, i + radius)
    # Move right pointer to ensure xx[right] - xx[i] <= radius
    while right < max_right and xx[right] <= x + radius:
        right += 1
    return left, right


def test_get_window_borders():
    N = 1000
    K = 100
    xx = np.arange(N)
    for i in range(K, len(xx) - K):
        s, e = get_window_borders(i, xx, K)
        assert e - s == 2 * K + 1


@numba.njit(boundscheck=True)
def max_around(ww, i, left, right):
    assert left < right
    _max = -math.inf
    for j in range(left, right):
        if j != i:
            _max = max(_max, ww[j])
    return _max


@numba.njit(boundscheck=True)
def max_intensity_in_window(results, xx, weights, radius, left=0, right=0):
    for i in range(len(xx)):
        left, right = get_window_borders(i, xx, radius, left, right)
        neighborhood_max = max_around(weights, i, left, right)
        results[i] = neighborhood_max


@numba.njit(boundscheck=True, parallel=True)
def get_unique_cnts_in_groups(xx_index: NDArray, yy: NDArray):
    unique_y_per_x = np.zeros(shape=len(xx_index) - 1, dtype=np.uint32)
    for i in numba.prange(len(xx_index) - 1):
        x_s = xx_index[i]
        x_e = xx_index[i + 1]
        y_prev = yy[x_s]
        for j in range(x_s + 1, x_e):
            y = yy[j]
            if y != y_prev:
                unique_y_per_x[i] += 1
            y_prev = y
    return unique_y_per_x


@numba.njit
def roll(X, k):
    assert k > 0
    for i in range(len(X), k - 1, -1):
        X[i] = X[i - k]
    for i in range(k):
        X[i] = 0


@numba.njit
def inplace_cumsum(X):
    for i in range(1, len(X)):
        X[i] += X[i - 1]


@numba.njit
def counts2index(X) -> None:
    X = X.ravel()
    inplace_cumsum(X)
    roll(X, 1)


def test_counts2index():
    test = np.arange(1, 11).reshape((2, 5))
    w = np.cumsum(test)[:-1]
    counts2index(test)
    np.testing.assert_equal(test.ravel()[1:], w)
    assert test[0, 0] == 0
