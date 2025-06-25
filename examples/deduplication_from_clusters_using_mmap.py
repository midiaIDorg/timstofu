"""
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path
from shutil import rmtree

from mmapped_df import open_dataset_dct
from mmapuccino import MmapedArrayValuedDict

from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.tofu import LexSortedClusters


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
assert sorted_clusters.is_sorted()

simulated_precursors = sorted_clusters.deduplicate(
    _empty=mmap_simulated_precursors.empty,
    _zeros=mmap_simulated_precursors.zeros,
)
assert simulated_precursors.is_sorted()


s = simulated_precursors.index[1, 450]
e = simulated_precursors.index[1, 451]

assert is_lex_nondecreasing(simulated_precursors.columns.tof[s:e])
