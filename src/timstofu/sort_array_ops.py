import numba

from numpy.typing import NDArray


@numba.njit
def is_lex_greater(xx: NDArray, yy: NDArray, or_equal: bool = False) -> bool:
    """xx < yy or xx <= yy"""
    for x, y in zip(xx, yy):
        if x > y:
            return False
        elif x < y:
            return True
    return or_equal


@numba.njit
def apply_on_consecutive_pairs(XX: NDArray, foo, *foo_args):
    if len(XX) <= 1:
        return True, 0
    x_prev = XX[0]
    for i in range(1, len(XX)):
        x = XX[i]
        if not foo(x_prev, x, *foo_args):
            return False, i
        x_prev = x
    return True, i


def is_lex_increasing(
    data: NDArray, strictly: bool = True, return_index: bool = False
) -> tuple[bool, int] | bool:
    decision, index = apply_on_consecutive_pairs(data, is_lex_greater, not strictly)
    if return_index:
        return decision, index
    return decision
