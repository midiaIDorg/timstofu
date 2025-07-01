import functools
import numba
import numpy as np

from numpy.typing import NDArray

from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import inputs_series_to_numpy
from timstofu.stats import count1D
from timstofu.stats import get_index


@numba.njit
def _argcountsort(xx: NDArray, counts: NDArray, order: NDArray) -> None:
    """Get the order sorting xx by values in group_index."""
    cumsums = np.cumsum(counts)
    # going backwards for a stable sort.
    for i in range(len(xx) - 1, -1, -1):
        x = xx[i]
        cumsums[x] -= np.uint32(1)  # !!! uint-int->float in numpy !!!
        j = cumsums[x]
        assert j >= 0, "Circular boom!!!"
        assert j < len(xx), "idx beyond scope"
        order[j] = i


def argcountsort(
    xx: NDArray,
    counts: NDArray | None = None,
    order: NDArray | None = None,
    return_counts: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    assert len(xx.shape) == 1, "Input `xx` must be 1D."
    counts = count1D(xx) if counts is None else counts
    if order is None:
        order = np.empty(
            xx.shape,
            get_min_int_data_type(
                len(xx),
                signed=False,
            ),
        )
    assert xx.shape == order.shape
    _argcountsort(xx, counts, order)
    if return_counts:
        return order, counts
    return order


@numba.njit(parallel=True)
def _grouped_argsort(xx: NDArray, group_index: NDArray, order: NDArray) -> None:
    """Sort arrays.

    Parameters
    ----------
    xx (np.array): Array to be argsorted, grouped by index.
    grouped_index (np.array): 1D array with counts, one field larger than the number of different values of the grouper.
    order (np.array): Place to store results.

    Notes
    -----
    `group_index[i]:group_index[i+1]` returns a view into all members of the i-th group.
    """
    assert len(xx) == len(order)
    assert len(xx) == group_index[-1]
    for i in numba.prange(len(group_index) - 1):
        s = group_index[i]
        e = group_index[i + 1]
        order[s:e] = s + np.argsort(xx[s:e])


@functools.wraps(_grouped_argsort)
@inputs_series_to_numpy
def grouped_argsort(
    xx, group_index: NDArray | None, order: NDArray | None = None
) -> NDArray:
    if group_index is None:
        group_index = get_index(xx)
    assert len(xx) == group_index[-1]
    if order is None:
        order = np.empty(xx.shape, get_min_int_data_type(len(xx)))
    _grouped_argsort(xx, group_index, order)
    return order


@numba.njit(parallel=True)
def _grouped_sort(unsorted: NDArray, group_index: NDArray, output: NDArray) -> None:
    assert len(unsorted) == len(output)
    assert len(unsorted) == group_index[-1]
    for i in numba.prange(len(group_index) - 1):
        s = group_index[i]
        e = group_index[i + 1]
        output[s:e] = np.sort(unsorted[s:e])


@numba.njit(cache=True)
def _is_nondecreasing(xx):
    x_prev = -math.inf
    for x in xx:
        if x_prev > x:
            return False
        x_prev = x
    return True


is_nondecreasing = inputs_series_to_numpy(_is_nondecreasing)


@numba.njit
def _is_lex_nondecreasing(*arrays: NDArray) -> bool:
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
def _is_lex_strictly_increasing(*arrays: NDArray) -> bool:
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
