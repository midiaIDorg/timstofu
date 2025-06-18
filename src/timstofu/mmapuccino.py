"""
TODO: Make this into a separate module.
TODO: Add tests.
"""


import json
import numpy as np
import shutil
import time

from dataclasses import dataclass
from dictodot import DotDict
from pathlib import Path
from typing import Callable


def empty(name: str = "", *args, **kwargs):
    """
    Returns an uninitialized NumPy array with the specified shape and dtype.

    Parameters:
        name (str): Unused parameter for compatibility or labeling (optional).
        *args: Positional arguments to pass to np.empty.
        **kwargs: Keyword arguments to pass to np.empty.

    Returns:
        np.ndarray: An uninitialized NumPy array.
    """
    return np.empty(*args, **kwargs)


def zeros(name: str = "", *args, **kwargs):
    """
    Returns a NumPy array filled with zeros with the specified shape and dtype.

    Parameters:
        name (str): Unused parameter for compatibility or labeling (optional).
        *args: Positional arguments to pass to np.zeros.
        **kwargs: Keyword arguments to pass to np.zeros.

    Returns:
        np.ndarray: A NumPy array filled with zeros.
    """
    return np.zeros(*args, **kwargs)


def to_int(shape):
    """
    Converts a shape to an integer or a tuple of integers.

    Parameters:
        shape (int or tuple): The shape to convert.

    Returns:
        int or tuple: The converted shape.

    Raises:
        AssertionError: If shape is not int or tuple of ints.
    """
    if isinstance(shape, np.integer):
        return int(shape)
    if isinstance(shape, tuple)
        return tuple(map(int, shape))


def get_empty_mmapped_array(
    path: str | Path,
    shape: int | tuple[int, ...],
    mode: str = "w+",
    dtype: str = "float32",
    *args,
    **kwargs,
):
    """
    Create an empty memory-mapped array and save its metadata as JSON.

    Parameters:
        path (str | Path): Path to the memory-mapped file.
        shape (int | tuple): Shape of the array.
        mode (str): File mode ('w+', 'r+', etc.).
        dtype (str): Data type of the array (must be a string, e.g., 'float32').
        *args: Additional positional arguments to np.memmap.
        **kwargs: Additional keyword arguments to np.memmap.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
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


def load_mmapped_array(path: Path, mode: str = "r+", *args, **kwargs):
    """
    Load a memory-mapped array using metadata from a JSON file.

    Parameters:
        path (Path): Path to the memory-mapped file (without .meta.json).
        mode (str): File mode to open the memory-mapped array.
        *args: Additional positional arguments to np.memmap.
        **kwargs: Additional keyword arguments to np.memmap.

    Returns:
        np.memmap: The loaded memory-mapped array.
    """
    path = Path(path)
    with open(path.parent / f"{path.name}.meta.json", "r") as f:
        m = json.load(f)
    return np.memmap(
        path,
        dtype=m["dtype"],
        mode=mode,
        shape=m["shape"],
        *args,
        **kwargs,
    )


class MmapedArrayValuedDict:
    """
    A dictionary-like wrapper where each value is a memory-mapped NumPy array.
    Arrays are automatically loaded from a directory and metadata files.

    Attributes:
        folder (Path): The path to the folder containing .dat and .meta.json files.
        data (DotDict): A dictionary of loaded memory-mapped arrays.
    """
    def __init__(self, folder: str | Path, mode: str = "w+", *arg, **kwargs):
        """Initialize the MmapedArrayValuedDict and load all existing .dat arrays.

        Parameters:
            folder (str | Path): Directory containing the .dat and .meta.json files.
            mode (str): File mode for loading the memory-mapped arrays.
            *arg: Additional positional arguments to pass to load_mmapped_array.
            **kwargs: Additional keyword arguments to pass to load_mmapped_array.
        """
        self.folder = Path(folder)
        self.data = DotDict(
            {
                path.stem: load_mmapped_array(path, mode=mode, *arg, **kwargs)
                for path in self.folder.glob("*.dat")
            }
        )

    def _get_empty(self, zero_out: bool = False):
        """
        Internal method to create an empty or zeroed memory-mapped array.

        Parameters:
            zero_out (bool): If True, zero out the array after creation.

        Returns:
            Callable: A function that creates and registers a new memory-mapped array.
        """
        def wrapper(name: str, *args, **kwargs):
            """
            Creates a new memory-mapped array and adds it to the internal dictionary.

            Parameters:
                name (str): Key name for the new array.
                *args: Positional arguments to pass to get_empty_mmapped_array.
                **kwargs: Keyword arguments to pass to get_empty_mmapped_array.

            Returns:
                np.memmap: The newly created memory-mapped array.
            """
            path = self.folder / f"{name}.dat"
            assert name not in self.data, f"Path `{path}` already allocated."
            arr = get_empty_mmapped_array(path=path, *args, **kwargs)
            if zero_out:
                arr[:] = 0
            self.data[name] = arr
            return arr

        return wrapper

    @property
    def empty(self) -> Callable:
        """Return a function to create and register an uninitialized memory-mapped array.

        Returns:
            Callable: Function that creates an uninitialized array.
        """
        return self._get_empty(zero_out=False)

    @property
    def zeros(self) -> Callable:
        """Return a function to create and register a zero-initialized memory-mapped array.

        Returns:
            Callable: Function that creates a zero-initialized array.
        """
        return self._get_empty(zero_out=True)

    def sync(self, verbose: bool = False) -> dict[str, int]:
        """Force OS to write to disk immediately and not when it wants.

        This might be important only for runtime evaluation.
        DON'T MESS WITH YOUR OS.

        Parameters:
        -----------
        verbose (bool): be verbose

        Returns
        -------
        dict: Mapping of array name to flush runtime.
        """
        flush_runtime = {}
        for col, val in self.data.items():
            start = time.time()
            val.flush()
            flush_runtime[col] = time.time() - start
            if verbose:
                print(f"Synced array `{col}` in `{flush_runtime[col]}` seconds.")
        return flush_runtime
