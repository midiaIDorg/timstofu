import math
import numba
import numpy as np
import operator

from numpy.typing import NDArray
from timstofu.stats import _minmax


@numba.njit(parallel=True)
def mod(xx: NDArray, divider: int, out: NDArray):
    assert len(xx) == len(out)
    for i in numba.prange(len(xx)):
        out[i] = xx[i] % divider
    return out


@numba.njit(parallel=True)
def div(xx: NDArray, divider: int, out: NDArray):
    assert len(xx) == len(out)
    for i in numba.prange(len(xx)):
        out[i] = xx[i] // divider
    return out


@numba.njit(parallel=True)
def mod_then_div(
    xx: NDArray,
    divider0: int,
    divider1: int,
    out: NDArray,
):
    assert len(xx) == len(out)
    assert divider0 >= divider1
    for i in numba.prange(len(xx)):
        out[i] = (xx[i] % divider0) // divider1
    return out


@numba.njit(parallel=True)
def inplace_mult(res: NDArray, xx: NDArray) -> None:
    for i in numba.prange(len(res)):
        res[i] *= xx[i]


@numba.njit(parallel=True)
def inplace_mult_by_constant(res: NDArray, mult: NDArray) -> None:
    for i in numba.prange(len(res)):
        res[i] *= mult


@numba.njit(parallel=True)
def inplace_add(res: NDArray, xx: NDArray) -> None:
    for i in numba.prange(len(res)):
        res[i] += xx[i]


def horner(arrays: tuple[NDArray], maxes: tuple[int], res: NDArray | None) -> NDArray:
    assert len(arrays) == len(maxes)
    N = None
    for arr in arrays:
        N = len(arr) if N is None else N
        assert len(arr) == N
    assert N == len(res)
    res[:] = arrays[0]
    for i in range(1, len(arrays)):
        inplace_mult_by_constant(res, maxes[i])
        inplace_add(res, arrays[i])
    return res


@numba.njit
def pack(values, maxes):
    """A hidden Horner crawling around the corner."""
    r = 0
    for i in range(len(maxes)):
        r *= maxes[i]
        r += values[i]
    return r


@numba.njit
def unpack3D(x, max1, max2):
    """Given maxes, get components of x in reverse Horner scheme.

    Specialized when x is know to consist of 3 entries. (3x faster to `unpack`).
    """
    c = x % max2
    x //= max2
    b = x % max1
    x //= max1
    return c, b, x


@numba.njit
def unpack(x, maxes):
    """Given maxes, get components of x in reverse Horner scheme."""
    for m in maxes[::-1]:
        yield x % m
        x //= m


@numba.njit
def unpack_np(x, maxes):
    """Given maxes, get components of x in reverse Horner scheme."""
    N = len(maxes)
    res = np.empty(shape=N, dtype=np.uint64)
    for i in range(len(maxes) - 1, -1, -1):
        m = maxes[i]
        res[i] = x % m
        x //= m
    return res


@numba.njit(boundscheck=True)
def max_nonzero_up(i, xx, radius):
    i = np.intp(i)
    N = len(xx)
    x_prev = xx[i]
    for j in range(radius):
        p = i + j + 1
        if p == N:
            return j
        x = xx[p]
        if x != x_prev + 1:
            return j
        x_prev = x
    return j


@numba.njit(boundscheck=True)
def max_nonzero_down(i, xx, radius):
    i = np.intp(i)
    N = len(xx)
    x_prev = xx[i]
    for j in range(radius):
        p = i - j - 1
        if p < 0:
            return j
        x = xx[p]
        if x + 1 < x_prev:
            return j
        x_prev = x
    return j


@numba.njit
def sum_weights(weights, left, right):
    r = 0
    for i in range(left, right):
        r += weights[i]
    return r


def test_max_nonzero():
    xx = np.array([1, 2, 5, 6, 7, 8, 10, 12, 123], dtype=np.uint32)
    i = 3
    assert xx[i + max_nonzero_up(i, xx, 3)] == 8
    assert xx[i - max_nonzero_down(i, xx, 3)] == 5

    xx = np.array([1, 2, 5, 6, 7, 8, 10, 12, 123], dtype=np.uint32)
    i = 0
    assert xx[i + max_nonzero_up(i, xx, 3)] == 2
    assert xx[max_nonzero_down(i, xx, 3)] == 1

    i = len(xx) - 1
    assert xx[i + max_nonzero_up(i, xx, 3)] == xx[-1]
    assert xx[i - max_nonzero_down(i, xx, 3)] == 123


@numba.njit(fastmath=True)
def log2(xx):
    return np.log2(xx)


@numba.njit(parallel=True)
def discretize(xx, res=None, transform=log2):
    if res is None:
        res = np.empty(shape=xx.shape, dtype=np.uint8)
    assert len(res) == len(xx)
    _min, _max = _minmax(xx)
    _min = transform(_min)
    _max = transform(_max)
    mult = 255 / (_max - _min)
    for i in numba.prange(len(res)):
        res[i] = (transform(xx[i]) - _min) * mult
    return res


@numba.njit(parallel=True)
def _reduce_resolution(xx: NDArray, factor: float, output: NDArray) -> None:
    for i in numba.prange(len(xx)):
        output[i] = math.ceil(xx[i] / factor)


def reduce_resolution(
    xx: NDArray, factor: float, output: NDArray | None = None
) -> NDArray:
    if output is None:
        output = np.empty(dtype=xx.dtype, shape=xx.shape)
    assert len(output) == len(xx)
    _reduce_resolution(xx, factor, output)
    return output


@numba.njit(parallel=True, boundscheck=True)
def moving_window(
    chunk_ends,
    diffs,
    data,
    updater,
    updater_args,
    progress_proxy,
):
    ONE = np.uintp(1)
    diffs = diffs.astype(np.intp)
    assert chunk_ends[-1, -1] == len(data)
    diff_starts = diffs[:, 0]
    diff_ends = diffs[:, 1]

    for i in numba.prange(len(chunk_ends)):
        # for i in range(len(chunk_ends)):
        chunk_s, chunk_e = chunk_ends[i]
        window_starts = np.full(len(diffs), chunk_s, data.dtype)  # INDEX
        window_ends = np.full(len(diffs), chunk_s, data.dtype)  # INDEX

        for c_idx in range(chunk_s, chunk_e):  # INDEX OF THE CURRENT WINDOW'S CENTER
            center_val = np.intp(data[c_idx])  # CENTER VALUE
            # UPDATE INDEX: REMEMBER DATA IS STRICTLY INCREASING
            for j in range(len(diffs)):
                t_s = center_val + diff_starts[j]  # TARGET START
                t_e = center_val + diff_ends[j]  # TARGET END

                # MOVE START
                while (
                    window_starts[j] < c_idx and np.intp(data[window_starts[j]]) < t_s
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

        progress_proxy.update(chunk_e - chunk_s)


## Funny: operator.mod worked, but own function did not work.
# def make_divmod(foo):
#     def real_foo(
#         xx: NDArray,
#         out: NDArray,
#         *args,
#     ):
#         for i in numba.prange(len(xx)):
#             out[i] = foo(xx[i], *args)
#         return out

#     return real_foo


# mod = numba.njit(parallel=True)(make_divmod(operator.mod))
# div = numba.njit(parallel=True)(make_divmod(operator.floordiv))


# @numba.njit
# def _div_then_mod(number, div0, div1):
#     return (number % div0) // div1

# div_then_mod = numba.njit(parallel=True)(make_divmod(_div_then_mod))
