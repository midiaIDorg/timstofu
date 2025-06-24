# -*- coding: utf-8 -*-
import functools
import itertools
import numba

import numpy as np
import numpy.typing as npt
import pandas as pd

from numba_progress import ProgressBar


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
def add_matrices_with_potentially_different_shapes(*arrays: npt.NDArray):
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
    in_arr: npt.NDArray,
    out_arr: npt.NDArray,
    order: npt.NDArray,
) -> npt.NDArray:
    """Write fron in_arr to out_arr using order."""
    assert len(in_arr) == len(out_arr)
    assert len(in_arr) == len(order)
    for i in range(len(in_arr)):
        out_arr[i] = in_arr[order[i]]
    return out_arr


@numba.njit
def copy(
    in_arr: npt.NDArray,
    out_arr: npt.NDArray,
) -> npt.NDArray:
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


def to_numpy(xx: npt.NDArray | pd.Series):
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


@numba.njit
def permute_inplace(xx, permutation, visited=None):
    """Apply order on orbits (cycles) of the permutation `permutation` in-place in `xx`.

    Likely best to do it for tables already in RAM to assure random access.

    Parameters
    ----------
    xx (np.array): Array to permutate.
    permutation (np.array): Permutation of indices to apply.

    Returns
    -------

    np.array: Reference to the `xx` input (piping-friendliness).
    """
    assert len(xx) == len(
        permutation
    ), "Cannot apply a `permutation` if `xx` has different length."
    if visited is None:
        visited = np.zeros(len(xx), dtype=np.bool_)
    else:
        visited[:] = False
    for i in range(len(xx)):
        if visited[i]:
            continue
        if permutation[i] == i:
            visited[i] = True
            continue
        j = i
        tmp = xx[i]
        while True:
            visited[j] = True
            next_j = permutation[j]
            if next_j == i:  # cycle finished
                xx[j] = tmp
                break
            xx[j] = xx[next_j]
            j = next_j
    return visited


def test_permute_inplace():
    xx = np.random.permutation(1000)
    yy = xx.copy()
    permutation = np.argsort(xx)
    _visited = permute_inplace(yy, permutation)
    assert np.all(_visited)
    np.testing.assert_equal(yy, xx[permutation])


@numba.njit(boundscheck=True, parallel=True)
def map_onto_lexsorted_indexed_data(
    xx_index: npt.NDArray,
    yy_by_xx: npt.NDArray,
    foo,
    foo_args,
    progress_proxy: ProgressBar | None = None,
) -> npt.NDArray:
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
def get_index_2D(xx_index: npt.NDArray, yy: npt.NDArray, paranoid: bool = False):
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
