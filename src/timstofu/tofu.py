from __future__ import annotations

from dataclasses import dataclass
from numba_progress import ProgressBar
from opentimspy import OpenTIMS
from pathlib import Path
from typing import Any

from dictodot import DotDict
from tqdm import tqdm

from timstofu.numba_helper import add_matrices_with_potentially_different_shapes
from timstofu.numba_helper import split_args_into_K

from timstofu.sort_and_pepper import deduplicate
from timstofu.sort_and_pepper import presort_tofs_and_intensities_per_frame_scan
from timstofu.sort_and_pepper import sort_events_lexicographically_per_frame_scan_tof
from timstofu.sort_and_pepper import sorted_in_frame_scan_groups

from timstofu.memmapped_tofu import MemmappedArrays
from timstofu.memmapped_tofu import open_memmapped_data

from timstofu.stats import count2D
from timstofu.stats import get_precumsums

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


# TODO: generalize to collections of arrays: will be simpler.
@numba.njit
def combine_datasets(
    self_counts: npt.NDArray,
    self_index: npt.NDArray,
    other_counts: npt.NDArray,
    other_index: npt.NDArray,
    progress_proxy: ProgressBar | None = None,
    *self_and_other_columns,
):
    # TODO: likely pass in a function to create memory mapped tables instead.
    # or simply paths.
    assert self_index.dtype == other_index.dtype
    assert self_counts.dtype == other_counts.dtype

    counts = add_matrices_with_potentially_different_shapes(self_counts, other_counts)
    index = get_precumsums(counts)

    s_cols, o_cols = split_args_into_K(2, *self_and_other_columns)

    s_N = len(s_cols[0])
    for s_col in s_cols:
        assert len(s_col) == s_N

    o_N = len(o_cols[0])
    for o_col in o_cols:
        assert len(o_col) == o_N

    N = s_N + o_N

    r_cols = []
    for s_col, o_col in zip(s_cols, o_cols):
        assert s_col.dtype == o_col.dtype
        r_cols.append(np.empty(shape=N, dtype=s_col.dtype))

    for frame, scan in zip(*counts.nonzero()):
        idx = index[frame, scan]
        cnt = counts[frame, scan]

        s_idx = self_index[frame, scan]
        o_idx = other_index[frame, scan]
        s_cnt = self_counts[frame, scan]
        o_cnt = other_counts[frame, scan]
        assert cnt == s_cnt + o_cnt

        for s_col, o_col, r_col in zip(s_cols, o_cols, r_cols):
            r_col[idx : idx + s_cnt] = s_col[s_idx : s_idx + s_cnt]
            r_col[idx + s_cnt : idx + s_cnt + o_cnt] = o_col[o_idx : o_idx + o_cnt]

        if progress_proxy is not None:
            progress_proxy.update(1)

    return index, counts, r_cols


@dataclass
class CompactDataset:
    index: npt.NDArray
    counts: npt.NDArray
    columns: DotDict

    def __post_init__(self):
        assert len(self.columns) > 0
        assert len(self.index.shape) == 2
        assert self.index.shape == self.counts.shape
        first_arr = None
        for arr in self.columns.values():
            first_arr = arr if first_arr is None else first_arr
            assert len(first_arr) == len(arr)

    def __len__(self):
        return len(next(iter(self.columns.values())))

    def __add__(self, other) -> CompactDataset:
        assert set(self.columns) == set(other.columns)

        counts = add_matrices_with_potentially_different_shapes(
            self.counts, other.counts
        )

        with ProgressBar(
            desc="Merging datasets",
            total=np.count_nonzero(counts),
        ) as progress_proxy:
            index, counts, columns = combine_datasets(
                self.counts,
                self.index,
                other.counts,
                other.index,
                progress_proxy,
                *self.columns.values(),
                *other.columns.values(),
            )

        return self.__class__(
            index=index,
            counts=counts,
            columns=DotDict(zip(self.columns, columns)),
        )

    def to_npz(self, output_path: str, compress: bool = True) -> None:
        """Save to npz format."""
        (np.savez_compressed if compress else np.savez)(
            file=output_path,
            index=self.index,
            counts=self.counts,
            **self.columns,
        )

    @classmethod
    def from_npz(cls, path: str):
        loaded = np.load(path)
        return cls(
            index=loaded["index"],
            counts=loaded["counts"],
            columns=DotDict(
                **{col: loaded[col] for col in loaded if col not in ("index", "counts")}
            ),
        )

    @classmethod
    def from_tofu(cls, folder: str | Path, mode="r", **kwargs):
        columns = open_memmapped_data(folder)
        counts = columns.pop("counts")
        return cls(index=get_precumsums(counts), counts=counts, columns=columns)


# I just direclty need that function.

# TODO: need to reimplement
# * presort_tofs_and_intensities_per_frame_scan
# * sort_events_lexicographically_per_frame_scan_tof
# to allow for variable number of columns.
# THEN: I can sort both clusters and datasets.
# The latter I need for adding noise on top of a dataset.


@dataclass
class LexSortedClusters(CompactDataset):
    frame_scan_to_unique_tofs_count: npt.NDArray | None = (
        None  # this not at all elegant. more like an elephant.
    )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        presort: bool = True,
        paranoid: bool = False,
    ) -> CompactDataset:
        """
        Arguments:
            df (pd.DataFrame): Opened data frame with ClusterID column.
            presort (bool): Presort clusters in (frame,scan) but not tof.
            paranoid (bool): Do checks that inidicate a need to visit a psychiastrist and fast.
        """
        for col in ("frame", "scan", "tof", "intensity", "ClusterID"):
            assert col in df.columns

        frame_scan_to_count, *frame_scan_ranges = count2D(df["frame"], df["scan"])
        frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)

        if presort:
            with ProgressBar(
                total=len(df),
                desc="Binning (tof,intensity) inside (frame,scan) groups",
            ) as progress_proxy:
                (
                    sorted_tofs,  # presorted by frame and scan
                    sorted_intensities,  # presorted by frame and scan
                    sorted_cluster_ids,  # presorted by frame and scan
                ) = presort_tofs_and_intensities_per_frame_scan(
                    frame_scan_to_first_idx=frame_scan_to_first_idx,
                    frame_scan_to_count=frame_scan_to_count,
                    frames=df.frame.to_numpy(),
                    scans=df.scan.to_numpy(),
                    tofs=df.tof.to_numpy(),
                    intensities=df.intensity.to_numpy(),
                    cluster_ids=df.ClusterID.to_numpy(),
                    progress_proxy=progress_proxy,
                )
        total_frame_scan_pairs = np.count_nonzero(frame_scan_to_count)

        with ProgressBar(
            total=total_frame_scan_pairs,
            desc="Sorting (tof,intensity) within (frame,scan) groups",
        ) as progress_proxy:
            frame_scan_to_unique_tofs_count = sort_events_lexicographically_per_frame_scan_tof(
                frame_scan_to_first_idx=frame_scan_to_first_idx,
                frame_scan_to_count=frame_scan_to_count,
                frame_scan_sorted_tofs=sorted_tofs,  # fully sorted after this finishes
                frame_scan_sorted_intensities=sorted_intensities,  # fully sorted after this finishes
                frame_scan_sorted_cluster_ids=sorted_cluster_ids,
                progress_proxy=progress_proxy,
            )

        assert np.all(frame_scan_to_unique_tofs_count <= frame_scan_to_count)
        assert np.all(frame_scan_to_unique_tofs_count[frame_scan_to_count > 0] > 0)

        if paranoid:
            with ProgressBar(
                total=total_frame_scan_pairs,
                desc="Checking if TOFs are sorted in groups of (frame,scan)",
            ) as progress_proxy:
                assert sorted_in_frame_scan_groups(
                    sorted_tofs,
                    frame_scan_to_first_idx,
                    frame_scan_to_count,
                    progress_proxy,
                )

        counts = frame_scan_to_count[frame_scan_to_count > 0]

        if paranoid:
            assert (
                len(sorted_tofs) == frame_scan_to_count[frame_scan_to_count > 0].sum()
            )
            assert len(sorted_intensities) == len(sorted_tofs)

        return cls(
            index=frame_scan_to_first_idx,
            counts=frame_scan_to_count,
            columns=DotDict(
                tof=sorted_tofs,
                intensity=sorted_intensities,
                ClusterID=sorted_cluster_ids,
            ),
            frame_scan_to_unique_tofs_count=frame_scan_to_unique_tofs_count,
        )

    def deduplicate(self) -> LexSortedDataset:
        assert self.frame_scan_to_unique_tofs_count is not None
        with ProgressBar(
            total=len(self),
            desc="Deduplicating (tof,intensity) pairs per (frame,scan)",
        ) as progress_proxy:
            dedup_tofs, dedup_intensities = deduplicate(
                self.columns.tof,
                self.columns.intensity,
                self.counts[self.counts > 0],
                self.frame_scan_to_unique_tofs_count.sum(),
                progress_proxy,
            )

        return self.__class__(
            index=get_precumsums(self.frame_scan_to_unique_tofs_count),
            counts=self.frame_scan_to_unique_tofs_count,
            columns=DotDict(
                tof=dedup_tofs,
                intensity=dedup_intensities,
            ),
        )


@dataclass
class LexSortedDataset(CompactDataset):
    """A tdf equivalent."""

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> CompactDataset:
        """
        Arguments:
            df (pd.DataFrame): A data frame with TDF content (likely from .startrek mmapped format).
        """
        for col in ("frame", "scan", "tof", "intensity"):
            assert col in df.columns

        assert (
            "ClusterID" not in df.columns
        ), "For representing Clusters, please use `SortedClusters.from_df`."

        frame_scan_to_count, *frame_scan_ranges = count2D(df["frame"], df["scan"])
        frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)
        return cls(
            index=frame_scan_to_first_idx,
            counts=frame_scan_to_count,
            columns=DotDict(
                tof=df.tof.to_numpy(),
                intensity=df.intensity.to_numpy(),
            ),
        )

    @classmethod
    def from_tdf(
        cls,
        folder_dot_d: str | Path | OpenTIMS,
        level: str = "both",
        output_path: str | Path | None = None,
        satelite_data_dtypes: dict[str, type] = dict(
            tof=np.uint32,
            intensity=np.uint32,
        ),
        tqdm_kwargs: dict[str, Any] = {},
        force: bool = False,
        mode: str = "r",
    ):
        """Create an instance of LexSortedDataset from a timsTOF .d folder.

        Arguments:
            folder_dot_d (str|OpenTIMS): Path to the .d folder.
            level (stre): What data to write: precursor, fragment, or both?
            output_path (str): Optional folder to store memory-mapped arrays.
            satelite_data_dtypes (dict): when provided output_path, what will the types be of each of the columns stored on disk?

        """
        assert level in ("precursor", "fragment", "both")
        raw_data = (
            folder_dot_d
            if isinstance(folder_dot_d, OpenTIMS)
            else OpenTIMS(folder_dot_d)
        )
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=force)
        match level:
            case "precursor":
                frames = raw_data.ms1_frames
            case "fragment":
                frames = raw_data.ms2_frames
            case "both":
                frames = raw_data.frames["Id"]

        if "desc" not in tqdm_kwargs:
            tqdm_kwargs["desc"] = "Counting (frame,scan) pairs among events"
        if "total" not in tqdm_kwargs:
            tqdm_kwargs["total"] = len(frames)

        if output_path is None:
            frame_scan_to_count = np.zeros(
                dtype=np.uint32,
                shape=(raw_data.max_frame + 1, raw_data.max_scan + 1),
            )
            columns = DotDict(
                raw_data.query(
                    frames,
                    columns=list(satelite_data_dtypes),
                )
            )
            for frame in tqdm(frames, **tqdm_kwargs):
                frame_data = raw_data.query(frame, columns="scan")
                unique_scans, counts = np.unique(frame_data["scan"], return_counts=True)
                frame_scan_to_count[frame, unique_scans] = counts

            return cls(
                index=get_precumsums(frame_scan_to_count),
                counts=frame_scan_to_count,
                columns=columns,
            )

        size = np.sum(raw_data.frames["NumPeaks"][frames - 1])

        scheme = {col: (dtype, size) for col, dtype in satelite_data_dtypes.items()}
        scheme["counts"] = (np.uint32, (raw_data.max_frame + 1, raw_data.max_scan + 1))

        with MemmappedArrays(
            folder=output_path,
            column_to_type_and_shape=scheme,
            mode="w+",
        ) as mm:
            raw_data.query(
                frames,
                columns={col: arr for col, arr in mm.items() if col != "counts"},
            )
            for frame in tqdm(frames, **tqdm_kwargs):
                frame_data = raw_data.query(frame, columns="scan")
                unique_scans, counts = np.unique(frame_data["scan"], return_counts=True)
                mm.counts[frame, unique_scans] = counts

        return cls.from_tofu(output_path)

    def to_tdf(self, path: str):
        raise NotImplementedError


# TODO: make obsolete
def sort_and_deduplicate_clusters(
    clusters_df: pd.DataFrame,
    paranoid: bool = False,
) -> tuple[CompactDataset, CompactDataset]:
    """Sort clusters by frame, scan, and tof, and deduplicate it to retain only unique pairs.

    Intensities of the same events, (frame, scan, tof)-wise, from different clusters will be added.

    Arguments:
        clusters_df (pd.DataFrame): A data frame with clustered events. Clusters are recognized by column ClusterID.
        paranoid (bool): Do checks that inidicate a need to visit a psychiastrist and fast.

    Returns:
        tuple: Deduplicated dataset and sorted dataset.
    """
    for col in ("ClusterID", "frame", "scan", "tof", "intensity"):
        assert col in clusters_df.columns

    clusters = DotDict.FromFrame(clusters_df)

    simulated_events_count = len(clusters.frame)
    frame_scan_to_count, *frame_scan_ranges = count2D(
        clusters["frame"], clusters["scan"]
    )
    frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)

    with ProgressBar(
        total=simulated_events_count,
        desc="Binning (tof,intensity) inside (frame,scan) groups",
    ) as progress_proxy:
        (
            sorted_tofs,  # presorted by frame and scan
            sorted_intensities,  # presorted by frame and scan
            sorted_cluster_ids,  # presorted by frame and scan
        ) = presort_tofs_and_intensities_per_frame_scan(
            frame_scan_to_first_idx=frame_scan_to_first_idx,
            frame_scan_to_count=frame_scan_to_count,
            frames=clusters.frame,
            scans=clusters.scan,
            tofs=clusters.tof,
            intensities=clusters.intensity,
            cluster_ids=clusters.ClusterID,
            progress_proxy=progress_proxy,
        )

    total_frame_scan_pairs = np.count_nonzero(frame_scan_to_count)

    with ProgressBar(
        total=total_frame_scan_pairs,
        desc="Sorting (tof,intensity) within (frame,scan) groups",
    ) as progress_proxy:
        frame_scan_to_unique_tofs_count = sort_events_lexicographically_per_frame_scan_tof(
            frame_scan_to_first_idx=frame_scan_to_first_idx,
            frame_scan_to_count=frame_scan_to_count,
            frame_scan_sorted_tofs=sorted_tofs,  # fully sorted after this finishes
            frame_scan_sorted_intensities=sorted_intensities,  # fully sorted after this finishes
            frame_scan_sorted_cluster_ids=sorted_cluster_ids,
            progress_proxy=progress_proxy,
        )

    assert np.all(frame_scan_to_unique_tofs_count <= frame_scan_to_count)
    assert np.all(frame_scan_to_unique_tofs_count[frame_scan_to_count > 0] > 0)

    if paranoid:
        with ProgressBar(
            total=total_frame_scan_pairs,
            desc="Checking if TOFs are sorted in groups of (frame,scan)",
        ) as progress_proxy:
            assert sorted_in_frame_scan_groups(
                sorted_tofs,
                frame_scan_to_first_idx,
                frame_scan_to_count,
                progress_proxy,
            )

    deduplicated_event_count = frame_scan_to_unique_tofs_count.sum()
    counts = frame_scan_to_count[frame_scan_to_count > 0]

    if paranoid:
        assert len(sorted_tofs) == frame_scan_to_count[frame_scan_to_count > 0].sum()
        assert len(sorted_intensities) == len(sorted_tofs)

    sorted_dataset = CompactDataset(
        frame_scan_to_first_idx,
        frame_scan_to_count,
        DotDict(
            tof=sorted_tofs,
            intensity=sorted_intensities,
            ClusterID=sorted_cluster_ids,
        ),
    )

    with ProgressBar(
        total=simulated_events_count,
        desc="Deduplicating (tof,intensity) pairs per (frame,scan)",
    ) as progress_proxy:
        dedup_tofs, dedup_intensities = deduplicate(
            sorted_tofs,
            sorted_intensities,
            frame_scan_to_count[frame_scan_to_count > 0],
            deduplicated_event_count,
            progress_proxy,
        )

    frame_scan_to_first_unique_idx = get_precumsums(frame_scan_to_unique_tofs_count)

    unique_dataset = CompactDataset(
        frame_scan_to_first_unique_idx,
        frame_scan_to_unique_tofs_count,
        DotDict(
            tof=dedup_tofs,
            intensity=dedup_intensities,
        ),
    )

    return unique_dataset, sorted_dataset
