import json
import numpy as np
import numpy.typing as npt

import functools

from pathlib import Path
from time import time

from dictodot import DotDict


def get_memmapped_dotdict(
    folder: str | Path,
    column_to_type_and_shape: dict[str, tuple[type, int | tuple[int, ...]]],
    mode: str = "w+",
    **other_np_memmap_kwargs,
) -> DotDict[str, npt.NDArray]:
    """Get a DotDict mapping names to memory mapped arrays.

    Arguments:
        folder (str | Path): where to store the data. Best not on RAMDISK (haha, low level joke).
        column_to_type_and_shape (dict): Mapping of valid column names (such that can be use as file names in your file system) to their type and shape. Those can vary.
        mode (str): mode of the memmap.
        **other_np_memmap_kwargs: see ?np.memmap

    Returns:
        DotDict: A mapping of array names to arrays.
    """
    folder = Path(folder)
    dtypes = []
    arrays = {}
    shapes = {}
    for name, (dtype, shape) in column_to_type_and_shape.items():
        dtypes.append((name, dtype))
        arrays[name] = np.memmap(
            folder / f"{name}.npy",
            mode=mode,
            shape=shape,
            dtype=dtype,
            **other_np_memmap_kwargs,
        )
        shapes[name] = (
            tuple(map(int, shape)) if isinstance(shape, tuple) else int(shape)
        )
    scheme = {col: (type_str, shapes[col]) for col, type_str in np.dtype(dtypes).descr}
    with open(folder / "scheme.json", "w") as f:
        json.dump(scheme, f)
    return DotDict(arrays)


def open_memmapped_data(
    folder: str | Path,
    mode="r",
    **other_np_memmap_kwargs,
) -> DotDict[str, npt.NDArray]:
    """Read existing memmapped arrays in a folder."""
    folder = Path(folder)
    with open(folder / "scheme.json", "r") as f:
        scheme = json.load(f)
    return DotDict(
        {
            name: np.memmap(
                folder / f"{name}.npy",
                mode=mode,
                dtype=np.dtype(type_str),
                shape=shape,
                **other_np_memmap_kwargs,
            )
            for name, (type_str, shape) in scheme.items()
        }
    )


def flush_results(results: DotDict[str, npt.NDArray], verbose: bool = True) -> None:
    for col, arr in results.items():
        if verbose:
            print(f"Flushing {col}.")
        arr.flush()


class MemmappedArrays:
    @functools.wraps(get_memmapped_dotdict)
    def __init__(self, *args, verbose=True, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.arrays = get_memmapped_dotdict(*self.args, **self.kwargs)
        self.verbose = verbose

    def __enter__(self):
        return self.arrays

    def __exit__(self, exc_type, exc_value, traceback):
        start_time = time()
        flush_results(self.arrays, verbose=self.verbose)
        end_time = time()
        if self.verbose:
            print(f"Flushing took {round(end_time-start_time, 2)} seconds")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return False  # return True to suppress exceptions
