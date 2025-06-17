import functools
import numba

import numpy as np
import numpy.typing as npt
import pandas as pd


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
            - 0 → np.uint8
            - 1 → np.uint16
            - 2 → np.uint32
            - 3 → np.uint64
        - If `signed` is True:
            - 0 → np.int8
            - 1 → np.int16
            - 2 → np.int32
            - 3 → np.int64

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
