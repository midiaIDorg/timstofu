"""
%load_ext autoreload
%autoreload 2
"""
from dictodot import DotDict
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from pathlib import Path
from timstofu.numba_helper import copy
from timstofu.tofu import CompactDataset
from timstofu.tofu import LexSortedDataset
from timstofu.tofu import empty

from mmapuccino import MmapedArrayValuedDict
from opentimspy import OpenTIMS

import numpy as np
import shutil


path = Path("/home/matteo/tmp/test_tdf_map.tofu")
shutil.rmtree(path)
path.mkdir(parents=True)
md = MmapedArrayValuedDict(path)

rawdata = OpenTIMS("/home/matteo/data_for_midiaID/F9477.d")
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=rawdata,
    level="precursor",
    satellite_data=["tof", "intensity", "mz"],
    _empty=md.empty,
)
md.data


tdf_column_to_dtype["tof"]

from opentimspy import column_to_dtype as tdf_column_to_dtype

np.uint32.__name__

# fragment_dataset = LexSortedDataset.from_tdf(
#     folder_dot_d=folder_dot_d,
#     level="fragment",
# )

import numpy as np

_in = np.array([1, 2, 3, -4], dtype=np.int32)
_out = np.empty(shape=_in.shape, dtype=np.uint32)
copy(_in, _out)
# TODO: look into addition again.
