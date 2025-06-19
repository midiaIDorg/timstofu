"""
%load_ext autoreload
%autoreload 2
"""
from dictodot import DotDict
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from timstofu.tofu import CompactDataset


import numpy as np

cd = CompactDataset(counts=np.array([[0, 0, 1], [1, 2, 0]]), columns=DotDict())


dataset_dd = open_dataset_dct(
    "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/datasets/memmapped/5/raw.d.cache"
)
df = dataset_df = open_dataset(
    "/home/matteo/Projects/midia/midia_experiments/pipelines/devel/midia_pipe/tmp/datasets/memmapped/5/raw.d.cache"
)

_get_columns({c: dd[c] for c in dd if c not in {"frame", "scan"}})


dd = DotDict(a=2)
isinstance(dd, DotDict)
isinstance({1: 3}, DotDict)


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

        satelite_data_names = set(dd) - {"frame", "scan"}
        sorted_clusters = LexSortedClusters(
            counts=frame_scan_to_count,
            index=frame_scan_to_first_idx,
            columns=DotDict(
                {
                    c: write_orderly(
                        in_arr=dd[c],
                        out_arr=_empty(
                            name=c, dtype=dd[c].dtype.str, shape=dd[c].shape
                        ),
                        order=lex_order,
                    )
                    for c in satelite_data_names
                }
            ),
        )
        return (sorted_clusters, lex_order) if return_order else sorted_clusters
