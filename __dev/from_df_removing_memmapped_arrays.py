"""
%load_ext autoreload
%autoreload 2
"""
import shutil

from mmapped_df import open_dataset_dct
from pathlib import Path


from timstofu.tofu import LexSortedClusters

from mmapuccino import MmapedArrayValuedDict


clusters_path = "/home/matteo/tmp/test1.mmappet"  # real fragment clusters
df = open_dataset_dct(clusters_path)


output_folder = "/tmp/test_blah.tofu"
shutil.rmtree(output_folder)

%%time
sorted_clusters_in_ram, lex_order = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path), output_folder=output_folder
)
# 5.36"

memmaped_folder = Path("/home/matteo/tmp/test_sorted_clusters.tofu")
shutil.rmtree(memmaped_folder)

%%time
memmaped_folder.mkdir()
md = MmapedArrayValuedDict(memmaped_folder)
sorted_clusters_on_disk, lex_order = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path),
    output_folder=output_folder,
    _empty=md.empty,
)
# 6.32" in /tmp, 7.18" on SSD.

assert sorted_clusters_on_disk == sorted_clusters_in_ram
