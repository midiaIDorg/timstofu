"""
%load_ext autoreload
%autoreload 2
"""
import numba
import numpy as np
import pandas as pd

from pathlib import Path
from shutil import rmtree

from mmapped_df import open_dataset_dct
from mmapuccino import MmapedArrayValuedDict

from timstofu.numba_helper import decount
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import melt
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset


simulated_precursors_path = Path("/home/matteo/tmp/simulated_precursors.mmappet")
simulated_sorted_clusters_path = Path(
    "/home/matteo/tmp/simulated_sorted_clusters.mmappet"
)

rmtree(simulated_sorted_clusters_path, ignore_errors=True)
simulated_sorted_clusters_path.mkdir(parents=True)
rmtree(simulated_precursors_path, ignore_errors=True)
simulated_precursors_path.mkdir(parents=True)

mmap_sorted_clusters = MmapedArrayValuedDict(simulated_sorted_clusters_path)
mmap_simulated_precursors = MmapedArrayValuedDict(simulated_precursors_path)

df = open_dataset_dct("/home/matteo/tmp/test1.mmappet")
sorted_clusters = LexSortedClusters.from_df(
    df,
    _empty=mmap_sorted_clusters.empty,
)

sorted_clusters.counts.nonzero()
melt(sorted_clusters.counts)

sorted_clusters.melt_index(very_long=False)
sorted_clusters.melt_index(very_long=True)
# make it into a tofu.CompactDataset method.









is_lex_nondecreasing
sorted_clusters

sorted_clusters

simulated_precursors = sorted_clusters.deduplicate(
    _empty=mmap_simulated_precursors.empty,
    _zeros=mmap_simulated_precursors.zeros,
)

if ms_level == PRECURSOR_LEVEL:
    # drt = discrete retention time
    drt2frame = np.unique(simulated_precursors.counts.nonzero()[0])


    drt_max = ??? # TODO
    
    # TODO: cast drts to uint16 if applicable.

    frame2drt_dct = {int(frame): drt for drt, frame in enumerate(drt2frame)}
    drt_scan_to_count = simulated_precursors.counts[drt2frame, :]
    drts, scans, drt_scan_to_count = melt(drt_scan_to_count)
else:
    # TODO: we need to still do something here, or not?
    drts, scans, frame_scan_to_count = melt(simulated_precursors.counts)
