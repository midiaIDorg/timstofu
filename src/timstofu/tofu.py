from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dictodot import DotDict
from mmapuccino import MmapedArrayValuedDict
from mmapuccino import empty
from mmapuccino import zeros
from numba_progress import ProgressBar
from opentimspy import OpenTIMS
from opentimspy import column_to_dtype as tdf_column_to_dtype
from tqdm import tqdm

from timstofu.numba_helper import add_matrices_with_potentially_different_shapes
from timstofu.numba_helper import copy
from timstofu.numba_helper import split_args_into_K
from timstofu.numba_helper import write_orderly

from timstofu.sort_and_pepper import argcountsort3D
from timstofu.sort_and_pepper import deduplicate
from timstofu.sort_and_pepper import is_lex_nondecreasing

from timstofu.stats import count2D
from timstofu.stats import count_unique_for_indexed_data
from timstofu.stats import get_precumsums

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

# END OBSOLETE


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

    counts = add_matrices_with_potentially_different_shapes(
        self_counts, other_counts
    )  # this supports multiple counts.
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
    counts: npt.NDArray
    columns: DotDict
    index: npt.NDArray | None = None

    def __post_init__(self):
        if self.index is None:
            self.index = get_precumsums(self.counts)
        assert len(self.index.shape) == 2
        assert self.index.shape == self.counts.shape
        if not isinstance(self.columns, DotDict):
            assert isinstance(self.columns, dict)
            self.columns = DotDict(self.columns)
        first_arr = None
        for arr in self.columns.values():
            first_arr = arr if first_arr is None else first_arr
            assert len(first_arr) == len(arr)

    def __len__(self):
        if len(self.columns) == 0:
            return 0
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

        # TODO: missing deduplication here, specific to certain classes.
        # self.postprocess()?

        return self.__class__(
            index=index,
            counts=counts,
            columns=DotDict(zip(self.columns, columns)),
        )

    def to_npz(self, output_path: str, compress: bool = True) -> None:
        """Save to npz format (including columns)."""
        (np.savez_compressed if compress else np.savez)(
            file=output_path,
            index=self.index,
            counts=self.counts,
            **self.columns,
        )

    @classmethod
    def from_npz(cls, path: str):
        """Read from npz format."""
        loaded = np.load(path)
        return cls(
            index=loaded["index"],
            counts=loaded["counts"],
            columns={
                col: loaded[col] for col in loaded if col not in ("index", "counts")
            },
        )

    @classmethod
    def from_tofu(cls, folder: str | Path, mode="r", *args, **kwargs):
        """Read from the .tofu memory mapped format."""
        dct = MmapedArrayValuedDict(folder=folder, mode=mode, *args, **kwargs)
        return cls(counts=dct.data.pop("counts"), columns=dct.data)

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


def _count_frame_scans(
    frames: npt.NDArray,
    scans: npt.NDArray,
    _empty: Callable = empty,
) -> npt.NDArray:
    _frame_scan_to_count, *_ = count2D(frames, scans)
    frame_scan_to_count = _empty(
        name="counts",
        dtype=_frame_scan_to_count.dtype.str,
        shape=_frame_scan_to_count.shape,
    )
    frame_scan_to_count[:] = _frame_scan_to_count
    return frame_scan_to_count


@dataclass(eq=False)
class LexSortedClusters(CompactDataset):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame | dict[str, npt.NDArray],
        return_order: bool = False,
        _do_paranoid_checks: bool = False,
        _empty: Callable = empty,
    ) -> CompactDataset | tuple[CompactDataset, npt.NDArray]:
        """
        Arguments:
            df (pd.DataFrame): Opened data frame with ClusterID column.
            presort (bool): Presort clusters in (frame,scan) but not tof.
            return_order: bool = False
            _do_paranoid_checks (bool): Do checks that inidicate a need to visit a psychiastrist and fast.
            _empty (Callable): Allocator of empty space. Defaults to a wrapper around np.empty.
        """
        for col in ("frame", "scan", "tof", "intensity", "ClusterID"):
            assert col in df

        dd = DotDict(
            df if isinstance(df, dict) else {c: df[c].to_numpy(copy=False) for c in df}
        )

        frame_scan_to_count = _count_frame_scans(dd.frame, dd.scan, _empty)
        lex_order, _, frame_scan_to_first_idx = argcountsort3D(
            dd.frame, dd.scan, dd.tof, return_counts=True
        )

        if _do_paranoid_checks:
            assert is_lex_nondecreasing(
                dd.frame[lex_order], dd.scan[lex_order], dd.tof[lex_order]
            ), "We did not get a lexicographically sorted data."

        sorted_clusters = LexSortedClusters(
            counts=frame_scan_to_count,
            index=frame_scan_to_first_idx,
            columns={
                c: write_orderly(
                    in_arr=v,
                    out_arr=_empty(name=c, dtype=v.dtype.str, shape=v.shape),
                    order=lex_order,
                )
                for c, v in dd.items()
                if c not in {"frame", "scan"}  # non satellite data
            },
        )
        return (sorted_clusters, lex_order) if return_order else sorted_clusters

    def count_unique_frame_scan_tof_tuples(
        self, unique_counts: npt.NDArray | None = None
    ):
        """Count unique (frame,scan,tof) occurrences among all events per each (frame,scan) pair.

        Parameters
        ----------
        unique_counts (np.array): Optional preallocated array.

        Returns
        -------
        np.array: Counts of unique occurrences of (frame, scan, tof) per (frame,scan)
        """
        unique_counts = count_unique_for_indexed_data(
            self.columns.tof,
            self.counts,
            self.index,
            unique_counts,
        )
        assert np.all(
            unique_counts <= self.counts
        ), "The number of unique counts is sometimes higher than non-unique counts of (frame,scan,tof) tuples. ABOMINATION!"
        return unique_counts

    def deduplicate(
        self,
        ProgressBarKwargs: dict[str, Any] = {},
        _empty: Callable = empty,
        _zeros: Callable = zeros,
    ) -> LexSortedDataset:
        """
        Deduplicate sorted clusters.

        Parameters
        ----------
        ProgressBarKwargs (dict): All different kwargs for numba_progress.ProgressBar.
        _empty (Callable): Allocator of empty space. Defaults to a wrapper around np.empty.
        _zeros (Callable): Allocator of zero array. Defaults to a wrapper around np.zeros.

        Returns
        -------
        LexSortedDataset: An equivalent of a TDF.
        """
        unique_counts = self.count_unique_frame_scan_tof_tuples(
            unique_counts=_zeros(
                name="counts",
                shape=self.counts.shape,
                dtype=self.counts.dtype.str,
            )
        )
        total_unique_cnt = unique_counts.sum()  # the size of things.
        dedup_tofs = _empty(
            name="tof",
            shape=total_unique_cnt,
            dtype=self.columns.tof.dtype.str,
        )
        dedup_intensities = _zeros(
            name="intensity",
            shape=total_unique_cnt,
            dtype=self.columns.intensity.dtype.str,
        )
        consecutive_frame_scan_groups_cnts = self.counts[self.counts > 0]

        ProgressBarKwargs["total"] = len(self)  # I know better...
        ProgressBarKwargs["desc"] = ProgressBarKwargs.get(
            "desc", "Deduplicating (tof,intensity) pairs per (frame,scan)"
        )

        with ProgressBar(**ProgressBarKwargs) as progress_proxy:
            dedup_tofs, dedup_intensities = deduplicate(
                self.columns.tof,
                self.columns.intensity,
                consecutive_frame_scan_groups_cnts,
                total_unique_cnt,
                progress_proxy,
                dedup_tofs,
                dedup_intensities,
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

    # TODO: missing memmap interface.
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame | dict[str, npt.NDArray],
        _empty=empty,
        _zeros=zeros,  # for now not used...
    ) -> CompactDataset:
        """
        Arguments:
            df (pd.DataFrame | dict): A data frame with TDF content (likely from .startrek mmapped format).
        """
        dd = DotDict(
            df if isinstance(df, dict) else {c: df[c].to_numpy(copy=False) for c in df}
        )
        for col in ("frame", "scan", "tof", "intensity"):
            assert col in dd
        assert (
            "ClusterID" not in dd
        ), "For representing Clusters, please use `SortedClusters.from_df`."

        return cls(
            counts=_count_frame_scans(dd.frame, dd.scan, _empty),
            columns={
                c: copy(v, _empty(name=c, dtype=v.dtype.str, shape=v.shape))
                for c, v in dd.items()
                if c not in {"frame", "scan"}
            },
        )

    @classmethod
    def from_tdf(
        cls,
        folder_dot_d: str | Path | OpenTIMS,
        level: str = "both",
        satellite_data: list[str] = ["tof", "intensity"],
        tqdm_kwargs: dict[str, Any] = {},
        _empty: Callable = empty,
    ):
        """Create an instance of LexSortedDataset from a timsTOF .d folder.

        Arguments:
            folder_dot_d (str|OpenTIMS): Path to the .d folder.
            level (stre): What data to write: precursor, fragment, or both?
            satellite_data (list): What columns to get from .d folder?
        """
        assert level in ("precursor", "fragment", "both")
        raw_data = (
            folder_dot_d
            if isinstance(folder_dot_d, OpenTIMS)
            else OpenTIMS(folder_dot_d)
        )
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

        return cls(
            counts=raw_data.count_frame_scan_occurrences(
                frames=tqdm(frames, **tqdm_kwargs),
                counts=_empty(
                    shape=(
                        raw_data.max_frame + 1,
                        raw_data.max_scan + 1,
                    )
                ),
            ),
            columns=raw_data.query(
                frames,
                columns={
                    c: _empty(name=c, dtype=tdf_column_to_dtype[c], shape=len(raw_data))
                    for c in satellite_data
                },
            ),
        )

    def to_tdf(self, path: str):
        raise NotImplementedError
