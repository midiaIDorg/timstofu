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
