"""
%load_ext autoreload
%autoreload 2
"""
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from numba_progress import ProgressBar
from pathlib import Path

import shutil

from timstofu.sort_and_pepper import lexargcountsort2D
from timstofu.sort_and_pepper import lexargcountsort2D_to_3D
from timstofu.tofu import LexSortedClusters


clusters_path = "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/clusters/tims/reformated/441/clusters.startrek"  # real fragment clusters
clusters_path = "/home/matteo/tmp/test1.mmappet"  # real fragment clusters

output_folder: str | Path | None = "/tmp/test_blah.tofu"
shutil.rmtree(output_folder)
sorted_clusters_on_drive = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path),
    output_folder=output_folder
)
sorted_clusters_in_ram = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path)
)

sorted_clusters_in_ram == sorted_clusters_on_drive

import numpy as np




from dictodot import DotDict
from timstofu.memmapped_tofu import IdentityContext
from timstofu.memmapped_tofu import MemmappedArrays
from timstofu.stats import count2D
from timstofu.stats import get_precumsums

df = open_dataset_dct(clusters_path)
paranoid: bool = True
force: bool = False


output_folder: str | Path | None = None
# @classmethod
# def from_df(
#     cls,
#     df: pd.DataFrame | dict[str, npt.NDArray],
#     presort: bool = True,
#     paranoid: bool = False,
# ) -> CompactDataset:

dct_df = df if isinstance(df, dict) else {c: df[c].to_numpy(copy=False) for c in df}
for col in ("frame", "scan", "tof", "intensity", "ClusterID"):
    assert col in dct_df

satellite_columns = set(dct_df) - set(["frame", "scan"])

frame_scan_to_count, min_frame, max_frame, min_scan, max_scan = count2D(
    dct_df["frame"], dct_df["scan"]
)
frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)
lex_order = argcountsort3D(dct_df["frame"], dct_df["scan"], dct_df["tof"])

if paranoid:
    assert is_lex_nondecreasing(
        dct_df["frame"][lex_order],
        dct_df["scan"][lex_order],
        dct_df["tof"][lex_order],
    )

if output_folder is None:
    Context = IdentityContext(
        counts=frame_scan_to_count, **{c: dct_df[c].to_numpy() for c in satellite_columns}
    )
    return sorted_clusters = LexSortedClusters(
        index=frame_scan_to_first_idx,
        counts=frame_scan_to_count,
        columns=DotDict({
            dct_df[c].to_numpy()[lex_order]
            for c in satellite_columns
        }),
        frame_scan_to_unique_tofs_count=None,
    )
else:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=force)
    with MemmappedArrays(
        folder=output_folder,
        column_to_type_and_shape={
            "counts": (frame_scan_to_count.dtype, frame_scan_to_count.shape),
            **{c: (dct_df[c].dtype, dct_df[c].shape) for c in satellite_columns},
        },
        mode="w+",
    ) as context:
        context["counts"][:] = frame_scan_to_count
        columns = DotDict()
        for c in satellite_columns:
            context[c][:] = dct_df[c].to_numpy()
            context[c] = context[c][lex_order]
            columns[c] = context[c]
        sorted_clusters = LexSortedClusters(
            index=frame_scan_to_first_idx,
            counts=context["counts"],
            columns=columns,
            frame_scan_to_unique_tofs_count=None,
        )
