import numba
import numpy as np
import operator

from numpy.typing import NDArray


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
