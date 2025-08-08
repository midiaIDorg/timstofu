import numba
import numpy as np

from collections.abc import Sized
from dictodot import DotDict
from numpy.typing import NDArray
from typing import Iterable


def is_data_dict(dct: dict[str, NDArray]) -> bool:
    if not isinstance(dct, dict):
        return False
    N = None
    for c, v in df.items():
        N = len(v) if N is None else N
        if len(v) != N:
            return False
    return True


def filtering_str(radii: dict[str, int], event_to_check: dict[str, int]) -> str:
    assert set(radii) == set(event_to_check)
    return " and ".join(
        f"abs({dim_name} - {event_to_check[dim_name]}) <= {radius}"
        for dim_name, radius in radii.items()
    )


def get_max_count(radii: dict[str, int]) -> int:
    return int(np.prod((np.array(list(radii.values())) + 1) * 2))


@numba.njit
def iter_array_splits(N: NDArray, k: int) -> Iterable[tuple[NDArray, int]]:
    """
    Split a number N into `k` approximately equal chunks.
    The first chunks will be slightly larger if N is not divisible by k.
    """
    q, r = divmod(N, k)  # q = base size, r = remainder
    start = 0
    for i in range(k):
        end = start + q + (1 if i < r else 0)
        yield start, end
        start = end


def matrix_to_data_dict(
    matrix: NDArray,
    columns: list[str] | tuple[str, ...],
) -> DotDict[str, NDArray]:
    assert len(matrix.shape) == 2
    assert matrix.shape[1] == len(columns)
    dd = DotDict()
    for i, col in enumerate(columns):
        dd[col] = matrix[:, i]
    return dd


@numba.njit
def approximate_block_sum(arr: NDArray, k: int, m: int) -> NDArray:
    rows, cols = arr.shape
    out = np.zeros((k, m), dtype=arr.dtype)

    # Precompute integer edges
    row_edges = np.empty(k + 1, dtype=np.int64)
    col_edges = np.empty(m + 1, dtype=np.int64)
    for i in range(k + 1):
        row_edges[i] = (i * rows) // k
    for j in range(m + 1):
        col_edges[j] = (j * cols) // m

    # Sum over blocks
    for bi in range(k):
        r0 = row_edges[bi]
        r1 = row_edges[bi + 1]
        for bj in range(m):
            c0 = col_edges[bj]
            c1 = col_edges[bj + 1]
            s = 0
            for r in range(r0, r1):
                for c in range(c0, c1):
                    s += arr[r, c]
            out[bi, bj] = s

    return out
