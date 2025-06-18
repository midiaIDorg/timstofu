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

from mmapuccino import MmapedArrayValuedDict
from mmapuccino import empty


clusters_path = "/home/matteo/tmp/test1.mmappet"  # real fragment clusters
output_folder: str | Path | None = "/tmp/test_blah.tofu"
shutil.rmtree(output_folder)
sorted_clusters_on_drive, lex_order = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path), output_folder=output_folder
)

sorted_clusters = LexSortedClusters.from_tofu("/tmp/test.tofu")  # this is deduplicated.
deduplicated_clusters_in_ram = (
    sorted_clusters.deduplicate()
)  # TODO: missing memmapped equivalent.

memmaped_folder = Path("/tmp/test.tofu")
shutil.rmtree(memmaped_folder)
memmaped_folder.mkdir()

md = MmapedArrayValuedDict(memmaped_folder)
deduplicated_clusters_on_disk = sorted_clusters.deduplicate(
    _empty=md.empty, _zeros=md.zeros
)

assert deduplicated_clusters_on_disk == deduplicated_clusters_in_ram
