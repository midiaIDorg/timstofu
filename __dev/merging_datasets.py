"""
%load_ext autoreload
%autoreload 2
"""
import numba
import numpy as np
import numpy.typing as npt
import shutil

from math import inf
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from numba_progress import ProgressBar
from pathlib import Path
from timstofu.sort_and_pepper import argcountsort3D
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import lexargcountsort2D
from timstofu.sort_and_pepper import lexargcountsort2D_to_3D
from timstofu.sort_and_pepper import test_count_unique_for_indexed_data
from timstofu.stats import _count_unique
from timstofu.stats import count_unique_for_indexed_data
from timstofu.stats import zeros_copy
from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset


precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
)
fragment_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="fragment",
)

precursor_dataset + fragment_dataset

# TODO: it would be nice to have a mechanism to make the decision about the memmapped serializer outside the function, to support mine and Michals when he does it.
# Have it!

from memmapped_tofu import MemmappedArrays
from memmapped_tofu import RamArrays


# no, we need a context that will make those instead! So exactly like MemmappedContext.
Context = IdentityContext(
    dedup_tofs=np.empty(
        dtype=sorted_tofs.dtype,
        shape=deduplicated_event_count,
    ),
    dedup_intensities=np.zeros(
        dtype=sorted_intensities.dtype,
        shape=deduplicated_event_count,
    ),
)
