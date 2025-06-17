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
from timstofu.numba_helper import write_orderly

from timstofu.sort_and_pepper import argcountsort3D
from timstofu.sort_and_pepper import deduplicate
from timstofu.sort_and_pepper import is_lex_nondecreasing

from timstofu.memmapped_tofu import MemmappedArrays
from timstofu.memmapped_tofu import open_memmapped_data

from timstofu.stats import count2D
from timstofu.stats import count_unique_for_indexed_data
from timstofu.stats import get_precumsums

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


# TODO: generalize to collections of arrays: will be simpler.
# This should be simpler if we have the order already.
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

    def __eq__(self, other):
        try:
            np.testing.assert_equal(self.counts, other.counts)
        except AssertionError:
            return False
        try:
            np.testing.assert_equal(self.index, other.index)
        except AssertionError:
            return False
        if set(self.columns) != set(other.columns):
            return False
        for col in self.columns:
            try:
                np.testing.assert_equal(self.columns[col], other.columns[col])
            except AssertionError:
                return False
        return True


# I just direclty need that function.
# THEN: I can sort both clusters and datasets.
# The latter I need for adding noise on top of a dataset.


@dataclass(eq=False)
class LexSortedClusters(CompactDataset):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame | dict[str, npt.NDArray],
        output_folder: str | Path | None = None,
        paranoid: bool = False,
        force: bool = False,
    ) -> tuple[CompactDataset, npt.NDArray]:
        """
        Arguments:
            df (pd.DataFrame): Opened data frame with ClusterID column.
            presort (bool): Presort clusters in (frame,scan) but not tof.
            paranoid (bool): Do checks that inidicate a need to visit a psychiastrist and fast.
        """
        for col in ("frame", "scan", "tof", "intensity", "ClusterID"):
            assert col in df

        dd = DotDict(
            df if isinstance(df, dict) else {c: df[c].to_numpy(copy=False) for c in df}
        )
        satellite_columns = set(dd) - set(["frame", "scan"])

        frame_scan_to_count, *_ = count2D(dd.frame, dd.scan)
        frame_scan_to_first_idx = get_precumsums(frame_scan_to_count)
        lex_order = argcountsort3D(dd.frame, dd.scan, dd.tof)

        if paranoid:
            assert is_lex_nondecreasing(
                dd.frame[lex_order], dd.scan[lex_order], dd.tof[lex_order]
            ), "We did not get a lexicographically sorted data."

        if output_folder is None:
            return (
                LexSortedClusters(
                    index=frame_scan_to_first_idx,
                    counts=frame_scan_to_count,
                    columns=DotDict({c: dd[c][lex_order] for c in satellite_columns}),
                ),
                lex_order,
            )

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=force)
        with MemmappedArrays(
            folder=output_folder,
            column_to_type_and_shape={
                "counts": (frame_scan_to_count.dtype, frame_scan_to_count.shape),
                **{c: (dd[c].dtype, dd[c].shape) for c in satellite_columns},
            },
            mode="w+",
        ) as context:
            context["counts"][:] = frame_scan_to_count
            columns = DotDict()
            for c in satellite_columns:
                write_orderly(in_arr=dd[c], out_arr=context[c], order=lex_order)
                columns[c] = context[c]
            return (
                LexSortedClusters(
                    index=frame_scan_to_first_idx,
                    counts=context["counts"],
                    columns=columns,
                ),
                lex_order,
            )

    def count_unique_frame_scan_tof_tuples(self):
        unique_counts = count_unique_for_indexed_data(
            self.columns.tof, self.counts, self.index
        )
        assert np.all(
            unique_counts <= self.counts
        ), "The number of unique counts is sometimes higher than non-unique counts of (frame,scan,tof) tuples. ABOMINATION!"
        return unique_counts

    def deduplicate(self) -> LexSortedDataset:
        unique_counts = self.count_unique_frame_scan_tof_tuples()
        with ProgressBar(
            total=len(self),
            desc="Deduplicating (tof,intensity) pairs per (frame,scan)",
        ) as progress_proxy:
            dedup_tofs, dedup_intensities = deduplicate(
                self.columns.tof,
                self.columns.intensity,
                self.counts[self.counts > 0],
                unique_counts.sum(),
                progress_proxy,
            )
        return self.__class__(
            index=get_precumsums(unique_counts),
            counts=unique_counts,
            columns=DotDict(
                tof=dedup_tofs,
                intensity=dedup_intensities,
            ),
        )


@dataclass(eq=False)  # stupid dataclass bullshit.
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
        frames_it = tqdm(frames, **tqdm_kwargs)

        if output_path is None:
            columns = DotDict(
                raw_data.query(
                    frames,
                    columns=list(satelite_data_dtypes),
                )
            )
            frame_scan_to_count = raw_data.count_frame_scan_occurrences(
                frames=frames_it, counts=mm.counts
            )
            return cls(
                index=get_precumsums(frame_scan_to_count),
                counts=frame_scan_to_count,
                columns=columns,
            )
        else:
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

    def to_tdf(self, path: str):
        raise NotImplementedError
