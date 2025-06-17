"""
%load_ext autoreload
%autoreload 2
"""
from mmapped_df import open_dataset
from numba_progress import ProgressBar

from timstofu.sort_and_pepper import argcountsort3D
from timstofu.sort_and_pepper import lexargcountsort2D
from timstofu.sort_and_pepper import lexargcountsort2D_to_3D
from timstofu.tofu import LexSortedClusters


clusters_path = "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/clusters/tims/reformated/441/clusters.startrek"  # real fragment clusters
clusters_path = "/home/matteo/tmp/test1.mmappet"  # real fragment clusters

from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.stats import count2D
from timstofu.stats import get_precumsums


df = open_dataset(clusters_path)
paranoid: bool = True

for col in ("frame", "scan", "tof", "intensity", "ClusterID"):
    assert col in df.columns

frame_scan_to_count, min_frame, max_frame, min_scan, max_scan = count2D(
    df["frame"], df["scan"]
)
frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)
lex_order = argcountsort3D(df.frame, df.scan, df.tof)

if paranoid:
    assert is_lex_nondecreasing(
        df.frame.iloc[lex_order],
        df.scan.iloc[lex_order],
        df.tof.iloc[lex_order],
    )

LexSortedClusters(
    index=frame_scan_to_first_idx,
    counts=frame_scan_to_count,
    columns=DotDict(
        tof=df.tof.to_numpy()[lex_order],
        intensity=df.intensity.to_numpy()[lex_order],
        ClusterID=df.ClusterID.to_numpy()[lex_order],
    ),
    frame_scan_to_unique_tofs_count=frame_scan_to_unique_tofs_count,
)
