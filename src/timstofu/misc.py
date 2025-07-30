import numpy.typing as npt


def is_data_dict(dct: dict[str, npt.NDArray]) -> bool:
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
