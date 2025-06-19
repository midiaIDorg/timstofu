"""
%load_ext autoreload
%autoreload 2
"""
from dictodot import DotDict
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from timstofu.tofu import CompactDataset
from timstofu.tofu import LexSortedDataset


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


folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"  # small data
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
)
fragment_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="fragment",
)
