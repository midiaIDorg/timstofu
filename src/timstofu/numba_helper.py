# -*- coding: utf-8 -*-
import functools
import itertools
import numba

import numpy as np
import pandas as pd

from numba_progress import ProgressBar
from numpy.typing import NDArray


@numba.njit
def overwrite(xx, what_with=0):
    for i in range(len(xx)):
        xx[i] = what_with
    return xx


@numba.njit
def _get_min_int_data_type(x, signed=True):
    if signed:
        if x <= 2**7 - 1:
            return 8
        elif x <= 2**15 - 1:
            return 16
        elif x <= 2**31 - 1:
            return 32
        else:
            return 64
    else:
        if x <= 2**8 - 1:
            return 8
        elif x <= 2**16 - 1:
            return 16
        elif x <= 2**32 - 1:
            return 32
        else:
            return 64


np_uints = (np.uint8, np.uint16, np.uint32, np.uint64)


def get_min_int_data_type(x, signed: bool = True):
    """
    Determine the minimal integer type code required to represent a given value.

    Parameters
    ----------
    x : int
        The input integer value for which to determine the minimal data type.
    signed : bool, optional
        Whether to use signed integer types. Defaults to False (unsigned).

    Returns
    -------
    int
        An integer code representing the minimal data type required:

        - If `signed` is False:
            - 0 : np.uint8
            - 1 : np.uint16
            - 2 : np.uint32
            - 3 : np.uint64
        - If `signed` is True:
            - 0 : np.int8
            - 1 : np.int16
            - 2 : np.int32
            - 3 : np.int64

    Notes
    -----
    For numba compatibility, use `_get_min_int_data_type` that returns an integer.
    """
    bits = _get_min_int_data_type(x, signed)
    return np.dtype(f"{'int' if signed else 'uint'}{bits}")


import numpy as np


def minimal_uint_type_from_list(xs):
    """
    Determine the minimal NumPy unsigned integer type needed to store a bit sum
    derived from a list of non-negative integers.

    For each element `x` in `xs`, computes log2(x + 1) to estimate the number of
    bits needed to uniquely represent the value. Sums these bit estimates across
    the list, then returns the smallest NumPy unsigned integer type (e.g., uint8,
    uint16, etc.) that can store the total bit count.

    Generated automatically by ChatGPT.

    Parameters
    ----------
    xs : array-like of int
        A list or array of non-negative integers.

    Returns
    -------
    dtype : numpy.dtype
        The smallest NumPy unsigned integer type capable of storing the total bit count.

    Raises
    ------
    ValueError
        If any value in `xs` is negative or the total bit requirement exceeds 64 bits.
    """
    xs = np.asarray(xs)
    if np.any(xs < 0):
        raise ValueError("All values must be non-negative")

    uints = [np.uint8, np.uint16, np.uint32, np.uint64]
    limits = [np.iinfo(dt).bits for dt in uints]

    if xs.size == 0:
        total_bits = 1
    else:
        xs_copy = xs.copy()
        xs_copy[np.argmin(xs_copy)] += 1
        total_bits = int(np.ceil(np.sum(np.log2(xs_copy))))

    for dtype, limit in zip(uints, limits):
        if total_bits <= limit:
            return dtype

    raise ValueError("Sum requires more than 64 bits")


def inputs_series_to_numpy(foo):
    @functools.wraps(foo)
    def wrapper(*args, **kwargs):
        args = [arg.to_numpy() if isinstance(arg, pd.Series) else arg for arg in args]
        kwargs = {
            k: v.to_numpy() if isinstance(v, pd.Series) else v
            for k, v in kwargs.items()
        }
        return foo(*args, **kwargs)

    return wrapper


@numba.njit
def split_args_into_K(K, *args):
    assert (
        len(args) % K == 0
    ), "Need to pass in the same number of columns for both groups, e.g. A1 A2 A3 B1 B2 B3"
    N = int(len(args) // K)
    res = []
    I = 0
    for _ in range(K):
        args_group = []
        for i in range(I, I + N):
            args_group.append(args[i])
        res.append(args_group)
        I += N
    return res


def test_split_args_into_K():
    A, B, C, D, E, F = range(6)
    for a, b in zip(split_args_into_K(2, A, B, C, D, E, F), [[A, B, C], [D, E, F]]):
        assert a == b

    for a, b in zip(split_args_into_K(3, A, B, C, D, E, F), [[A, B], [C, D], [E, F]]):
        assert a == b

    for a, b in zip(
        split_args_into_K(6, A, B, C, D, E, F), [[A], [B], [C], [D], [E], [F]]
    ):
        assert a == b

    A = np.array([1, 2, 3])
    B = np.array([5, 6, 7])
    C = np.array([8, 9, 10])
    D = np.array([11, 12, 13])
    E = np.array([14, 15, 16])
    F = np.array([17, 18, 19])
    for a, b in zip(split_args_into_K(2, A, B, C, D, E, F), [[A, B, C], [D, E, F]]):
        assert a == b


@numba.njit
def add_matrices_with_potentially_different_shapes(*arrays: NDArray):
    """Add a variable number of 2D arrays with potentially different shapes.

    Result will simply be the maximal shaped array.

    Arguments:
        *arrays: 2D arrays to be added.
    """
    S_rows = 0
    S_cols = 0
    for arr in arrays:
        assert arr.dtype == arrays[0].dtype
        assert len(arr.shape) == 2
        rows, cols = arr.shape
        S_rows = max(S_rows, rows)
        S_cols = max(S_cols, cols)

    S = np.zeros(dtype=arrays[0].dtype, shape=(S_rows, S_cols))
    for arr in arrays:
        rows, cols = arr.shape
        S[:rows, :cols] += arr

    return S


def test_add_matrices_with_potentially_different_shapes():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6, 7]])
    c = np.array([[8], [9], [10]])
    # Expected result:
    # Shape: 3x3 (maximum shape)
    # a padded: [[1, 2, 0], [3, 4, 0], [0, 0, 0]]
    # b padded: [[5, 6, 7], [0, 0, 0], [0, 0, 0]]
    # c padded: [[8, 0, 0], [9, 0, 0], [10, 0, 0]]
    # sum:     [[14, 8, 7], [12, 4, 0], [10, 0, 0]]
    result = add_matrices_with_potentially_different_shapes(a, b, c)
    expected = np.array([[14, 8, 7], [12, 4, 0], [10, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@numba.njit
def write_orderly(
    in_arr: NDArray,
    out_arr: NDArray,
    order: NDArray,
) -> NDArray:
    """Write fron in_arr to out_arr using order."""
    assert len(in_arr) == len(out_arr)
    assert len(in_arr) == len(order)
    for i in range(len(in_arr)):
        out_arr[i] = in_arr[order[i]]
    return out_arr


@numba.njit
def copy(
    in_arr: NDArray,
    out_arr: NDArray,
) -> NDArray:
    """Write fron in_arr to out_arr using order."""
    assert len(in_arr) == len(out_arr)
    for i in range(len(in_arr)):
        out_arr[i] = in_arr[i]
    return out_arr


@numba.njit(cache=True)
def empty_copy(xx):
    return np.empty(dtype=xx.dtype, shape=xx.shape)


@numba.njit(cache=True)
def zeros_copy(xx):
    return np.zeros(dtype=xx.dtype, shape=xx.shape)


@numba.njit
def melt(arr):
    """
    Converts a N-dimensional NumPy array (aka dense data) into coordinate and value vectors for non-zero elements (aka sparse data).

    Parameters
    ----------
    arr : ndarray
        An N-dimensional NumPy array.

    Returns
    -------
    coords : tuple of 1D ndarrays
        A tuple containing N arrays, each corresponding to the indices of non-zero elements
        along one dimension of `arr`.

    values : 1D ndarray
        A 1D array of the non-zero values from `arr`, ordered according to their indices
        in `coords`.

    Notes
    -----
    This function mimics a "melt" operation, returning the coordinates and values
    of all non-zero entries in the input array. It is JIT-compiled using Numba for performance.
    """
    nonzero = arr.nonzero()
    values = np.empty(dtype=arr.dtype, shape=nonzero[0].shape)
    for i, idx in enumerate(zip(*nonzero)):
        values[i] = arr[idx]
    return (*nonzero, values)


@numba.njit
def decount(xx: NDArray, counts: NDArray, _results: NDArray | None = None):
    """Opposite to melt for 1D data.

    Equivalent of np.reapeat(xx, counts) that preserves dtype and allows use of a preallocated array.

    Parameters:
        xx (np.array): Array with entries that will get repeated.
        counts (np.array): Array with numbers of repeats to perform for each element of xx.
        _results (np.array): Optional preallocated place for keeping the resulting long format version of xx.

    Returns:
        np.array:
    """
    xx_final_size = counts.sum()
    if _results is None:
        _results = np.empty(shape=xx_final_size, dtype=xx.dtype)
    assert xx_final_size == len(_results)
    i = 0
    for x, cnt in zip(xx, counts):
        for _ in range(cnt):
            _results[i] = x
            i += 1
    return _results


def to_numpy(xx: NDArray | pd.Series):
    if isinstance(xx, pd.Series):
        xx = xx.to_numpy()
    return xx


def numba_wrap(foo):
    """Wrap function `foo` in all types of numba modes."""
    return {
        "python": foo,
        **{
            (
                "safe" if boundscheck else "fast",
                "multi_threaded" if parallel else "single_threaded",
            ): functools.wraps(foo)(
                numba.njit(boundscheck=boundscheck, parallel=parallel)(foo)
            )
            for boundscheck, parallel in itertools.product((True, False), repeat=2)
        },
    }


# TODO: roll back to previous function and write a python wrapper simply.
# Also: this definitely can be done withing groups.
# So we can still use index.
@numba.njit
def permute_inplace(
    array: tuple[NDArray, ...],
    permutation: NDArray,
    visited: NDArray | None = None,
) -> NDArray:
    """Apply order on orbits (cycles) of the permutation `permutation` in-place on `array`.

    Likely best to do it for tables already in RAM to assure random access.

    Parameters
    ----------
    array (np.array): An array to permutate in-place.
    permutation (np.array): Permutation of indices to apply.

    Returns
    -------
    np.array: a boolear array of visited places for reuse in another calls of this function or similar.
    """
    N = len(array)
    if visited is None:
        visited = np.empty(N, dtype=np.bool_)
    else:
        assert len(visited) == N
    visited[:] = False
    for i in range(N):
        if visited[i]:
            continue
        if permutation[i] == i:
            visited[i] = True
            continue
        j = i
        tmp = array[i]
        while True:
            visited[j] = True
            next_j = permutation[j]
            if next_j == i:  # cycle finished
                array[j] = tmp
                break
            array[j] = array[next_j]
            j = next_j
    return visited


def test_permute_inplace():
    xx = np.random.permutation(1000)
    yy = xx.copy()
    permutation = np.argsort(xx)
    _visited = permute_inplace(permutation, (yy,))
    assert np.all(_visited)
    np.testing.assert_equal(yy, xx[permutation])


@numba.njit(parallel=True)
def permute_into(xx: NDArray, permutation: NDArray, yy: NDArray | None = None):
    if yy is None:
        yy = np.empty(shape=xx.shape, dtype=xx.dtype)
    for i in numba.prange(len(xx)):
        yy[i] = xx[permutation[i]]
    return yy


@numba.njit(boundscheck=True, parallel=True)
def map_onto_lexsorted_indexed_data(
    xx_index: NDArray,
    yy_by_xx: NDArray,
    foo,
    foo_args,
    progress_proxy: ProgressBar | None = None,
) -> NDArray:
    """Here we iterate over indexed xx but in groups by yy.

    Notes
    -----
    Data (xx,yy) must be lexicographically sorted.
    `foo` must be
    """
    unique_y_per_x = np.zeros(shape=len(xx_index) - 1, dtype=np.uint32)
    for i in numba.prange(len(xx_index) - 1):
        j_s = xx_index[i]
        j_e = xx_index[i + 1]
        j_prev = j_s
        y_prev = yy_by_xx[j_prev]
        for j in range(j_s + 1, j_e):
            y = yy_by_xx[j]
            if y != y_prev:
                unique_y_per_x[i] += 1
                foo(j_prev, j, *foo_args)
                y_prev = y
                j_prev = j
        foo(j_prev, j_e, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(1)
    return unique_y_per_x


@numba.njit(boundscheck=True)
def test_foo_for_map_onto_lexsorted_indexed_data(left, right, *args):
    pass


def is_permutation(xx):
    visited = np.zeros(len(xx), dtype=np.bool_)
    visited[xx] = True
    return np.sum(visited) == len(xx)


@numba.njit(boundscheck=True)
def get_index_2D(xx_index: NDArray, yy: NDArray, paranoid: bool = False):
    unique_y_per_x = np.zeros(shape=len(xx_index) - 1, dtype=np.uint32)
    res = []
    x_s = xx_index[0]
    for i, x_e in enumerate(xx_index[1:]):
        if paranoid:
            assert is_lex_nondecreasing(yy[x_s:x_e])
        y_s = yy[x_s]
        for j in range(x_s + 1, x_e):
            y_e = yy[j]
            if y_e != y_s:
                res.append(j)
                unique_y_per_x[i] += 1
            y_s = y_e
        x_s = x_e
    res.append(x_e)
    return res, unique_y_per_x


@numba.njit
def is_arange(res, _min=0, _max=None):
    if _max is None:
        _max = len(res)
    for i, r in zip(range(_min, _max), res):
        if i != r:
            return False, i, r
    return True, i, r


@numba.njit
def divide_indices(N: int, k: int = numba.get_num_threads()):
    # Compute chunk sizes (some chunks may be 1 element longer to handle remainders)
    chunk_sizes = [(N + i) // k for i in range(k)]

    # Convert sizes to start and end indices
    indices = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        indices.append((start, end))
        start = end
    return indices
