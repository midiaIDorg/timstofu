import json
import numpy as np
import shutil

from dataclasses import dataclass
from dictodot import DotDict
from pathlib import Path


def empty(name: str = "", *args, **kwargs):
    return np.empty(*args, **kwargs)


def to_int(shape):
    try:
        return int(shape)
    except TypeError:
        assert isinstance(shape, tuple)
        return tuple(map(int, shape))


def get_empty_mmapped_array(
    path: str | Path,
    shape: int | tuple[int, ...],
    mode: str = "w+",
    dtype: str = "float32",
    *args,
    **kwargs,
):
    assert isinstance(
        dtype, str
    ), "Use string dtypes, please. Like `uint32`, not `np.uint32`."
    path = Path(path)
    with open(path.parent / f"{path.name}.meta.json", "w") as f:
        json.dump({"shape": to_int(shape), "dtype": dtype, "mode": mode}, f)
    return np.memmap(
        path,
        dtype=dtype,
        mode=mode,
        shape=shape,
        *args,
        **kwargs,
    )


def load_mmapped_array(cls, path: Path, mode: str = "r+", *args, **kwargs):
    path = Path(path)
    with open(path.parent / f"{path.name}.meta.json", "r") as f:
        m = json.load(f)
    return np.memmap(
        path,
        dtype=m["dtype"],
        mode=mode,
        shape=tuple(m["shape"]),
        *args,
        **kwargs,
    )


class MmapedArrayValuedDict:
    def __init__(self, folder: str | Path, mode: str = "w+", *arg, **kwargs):
        self.folder = Path(folder)
        self.data = DotDict(
            {
                path.stem: load_mmapped_array(path, mode=mode, *arg, **kwargs)
                for path in self.folder.glob("*.dat")
            }
        )

    def get_empty(self):
        def wrapper(name: str, *args, **kwargs):
            path = self.folder / f"{name}.dat"
            assert name not in self.data, f"Path `{path}` already allocated."
            arr = get_empty_mmapped_array(path=path, *args, **kwargs)
            self.data[name] = arr
            return arr

        return wrapper
