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
from opentimspy import OpenTIMS
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


folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
)
fragment_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="fragment",
)

precursor_dataset + fragment_dataset
