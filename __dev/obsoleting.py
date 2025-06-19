"""
%load_ext autoreload
%autoreload 2
"""
from dictodot import DotDict
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from timstofu.numba_helper import copy
from timstofu.tofu import CompactDataset
from timstofu.tofu import LexSortedDataset
from timstofu.tofu import empty


import numpy as np

cd = CompactDataset(counts=np.array([[0, 0, 1], [1, 2, 0]]), columns=DotDict())


dataset_dd = open_dataset_dct(
    "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/datasets/memmapped/5/raw.d.cache"
)
df = dataset_df = open_dataset(
    "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/datasets/memmapped/5/raw.d.cache"
)

_get_columns({c: dd[c] for c in dd if c not in {"frame", "scan"}})


dd = DotDict(a=2)
isinstance(dd, DotDict)
isinstance({1: 3}, DotDict)


from opentimspy import OpenTIMS
from tqdm import tqdm

level = "precursor"
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"  # small data
tqdm_kwargs = {}
# precursor_dataset = LexSortedDataset.from_tdf(
#     folder_dot_d=folder_dot_d,
#     level="precursor",
# )
# fragment_dataset = LexSortedDataset.from_tdf(
#     folder_dot_d=folder_dot_d,
#     level="fragment",
# )

import numpy as np

_in = np.array([1, 2, 3, -4], dtype=np.int32)
_out = np.empty(shape=_in.shape, dtype=np.uint32)
copy(_in, _out)


from opentimspy import column_to_dtype


# TODO: clean all of timstofu.memmapped shit.
