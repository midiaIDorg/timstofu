from numpy.typing import NDArray


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


def split_array(arr: NDArray, k: int, right_buffer: int = 0) -> list[NDArray]:
    """
    Split a 1D NumPy array into `k` approximately equal chunks.
    The first chunks will be slightly larger if N is not divisible by k.
    """
    N = len(arr)
    q, r = divmod(N, k)  # q = base size, r = remainder
    splits = []
    start = 0
    for i in range(k):
        end = start + q + (1 if i < r else 0)
        splits.append(arr[start : end + right_buffer])
        start = end
    return splits
