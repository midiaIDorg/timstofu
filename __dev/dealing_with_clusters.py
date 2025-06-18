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


clusters_path = "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/clusters/tims/reformated/441/clusters.startrek"  # real fragment clusters
clusters_path = "/home/matteo/tmp/test1.mmappet"  # real fragment clusters

output_folder: str | Path | None = "/tmp/test_blah.tofu"
shutil.rmtree(output_folder)
sorted_clusters_on_drive, lex_order = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path),
    output_folder=output_folder
)
sorted_clusters_in_ram, lex_order = LexSortedClusters.from_df(
    df=open_dataset_dct(clusters_path)
)
# write some test for it all.


# OK, now what? The addiition of CompactDatasets
folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d" # small data
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
)
fragment_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="fragment",
)

precursor_dataset + fragment_dataset
# combine_datasets: does not allow for memmapped arrays at all.





sorted_clusters_in_ram == sorted_clusters_on_drive

######################################################################


sorted_clusters = LexSortedClusters.from_tofu("/tmp/test_blah.tofu")
deduplicated_clusters = sorted_clusters.deduplicate() # TODO: missing memmapped equivalent.

sorted_clusters.count_unique_frame_scan_tof_tuples()




output_path = Path(output_path)
output_path.mkdir(parents=True, exist_ok=force)
size = np.sum(raw_data.frames["NumPeaks"][frames - 1])
scheme = {col: (dtype, size) for col, dtype in satelite_data_dtypes.items()}
scheme["counts"] = (
    np.uint32,
    (raw_data.max_frame + 1, raw_data.max_scan + 1),
)
with MemmappedArrays(
    folder=output_path,
    column_to_type_and_shape=scheme,
    mode="w+",
) as context:
    raw_data.query(
        frames,
        columns={
            col: arr for col, arr in context.items() if col != "counts"
        },
    )
    raw_data.count_frame_scan_occurrences(
        frames=frames_it, counts=context.counts
    )
return cls.from_tofu(output_path)






unique_counts = sorted_clusters.count_unique_frame_scan_tof_tuples()
deduplicated_event_count = unique_counts.sum()
sorted_tofs = sorted_clusters.columns.tof
sorted_intensities = sorted_clusters.columns.intensity
nondeduplicated_counts = sorted_clusters.counts[sorted_clusters.counts>0]



xx = np.array([1, 1, 1, 1, 2, 2, 2])
yy = np.array([2, 1, 2, 1, 1, 2, 1])
zz = np.array([2, 1, 2, 2, 1, 2, 1])

order, xy2count, xy2first_idx = argcountsort3D(xx, yy, zz, return_counts=True)
assert is_lex_nondecreasing(xx[order], yy[order], zz[order])

count_unique_for_indexed_data(
    zz[order], xy2count, xy2first_idx
).nonzero()

test_count_unique_for_indexed_data()




@numba.njit(boundscheck=True)
def deduplicate(
    sorted_tofs: npt.NDArray,
    sorted_intensities: npt.NDArray,
    nondeduplicated_counts: npt.NDArray,
    deduplicated_event_count: int,
    progress_proxy: ProgressBar | None = None,
):
    dedup_tofs = np.empty(
        dtype=sorted_tofs.dtype,
        shape=deduplicated_event_count,
    )
    dedup_intensities = np.zeros(
        dtype=sorted_intensities.dtype,
        shape=deduplicated_event_count,
    )

    counts_idx = 0
    current_group_count = 0
    dedup_idx = -1
    prev_tof = -inf


    for i, (tof, intensity) in enumerate(zip(sorted_tofs, sorted_intensities)):
        # if dedup_idx == 83185278:
        #     return dedup_idx, i, tof, prev_tof, intensity, counts_idx, current_group_count, dedup_tofs, dedup_intensities

        if current_group_count == nondeduplicated_counts[counts_idx]:
            counts_idx += 1
            current_group_count = 0
            prev_tof = -inf  # force top > prev_tof

        if tof > prev_tof:
            dedup_idx += 1
            # dedup_tofs[dedup_idx] = tof

        # dedup_intensities[dedup_idx] += intensity
        prev_tof = tof
        current_group_count += 1
        if progress_proxy is not None:
            progress_proxy.update(1)

    # assert dedup_idx == deduplicated_event_count - 1
    # assert counts_idx == len(nondeduplicated_counts) - 1
    return dedup_idx, i, tof, prev_tof, intensity, counts_idx, current_group_count, dedup_tofs, dedup_intensities
    # return dedup_tofs, dedup_intensities

with ProgressBar(total=sorted_clusters.counts.sum()) as progress_proxy:
    dedup_idx, i, tof, prev_tof, intensity, counts_idx, current_group_count, dedup_tofs, dedup_intensities = deduplicate(
        sorted_tofs = sorted_clusters.columns.tof,
        sorted_intensities = sorted_clusters.columns.intensity,
        nondeduplicated_counts = sorted_clusters.counts[sorted_clusters.counts>0],
        deduplicated_event_count = unique_counts.sum(),
        progress_proxy=progress_proxy
    )
dedup_idx
len(dedup_intensities)
len(sorted_tofs) == sorted_clusters.counts.sum()

unique_counts.sum()


from dictodot import DotDict
from timstofu.sort_and_pepper import is_lex_nondecreasing

dd = DotDict(open_dataset_dct(clusters_path))
assert is_lex_nondecreasing(
    dd.frame[lex_order], dd.scan[lex_order], dd.tof[lex_order]
), "We did not get a lexicographically sorted data."






dedup_tofs = np.empty(
    dtype=sorted_tofs.dtype,
    shape=deduplicated_event_count,
)
dedup_intensities = np.zeros(
    dtype=sorted_intensities.dtype,
    shape=deduplicated_event_count,
)

counts_idx = 0
current_group_count = 0
dedup_idx = -1
prev_tof = -inf

# i, (tof,intensity) = next(enumerate(zip(sorted_tofs, sorted_intensities)))
for i, (tof, intensity) in enumerate(zip(sorted_tofs, sorted_intensities)):
    if current_group_count == nondeduplicated_counts[counts_idx]:
        counts_idx += 1
        current_group_count = 0
        prev_tof = -inf  # force top > prev_tof

    if tof > prev_tof:
        dedup_idx += 1
        dedup_tofs[dedup_idx] = tof

    dedup_intensities[dedup_idx] += intensity
    prev_tof = tof
    current_group_count += 1
    if progress_proxy is not None:
        progress_proxy.update(1)

assert dedup_idx == deduplicated_event_count - 1
assert counts_idx == len(nondeduplicated_counts) - 1



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
