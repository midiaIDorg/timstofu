from __future__ import annotations

import functools
import math
import numba
import numpy as np
import pandas as pd

from dataclasses import dataclass
from dataclasses import field
from numpy.typing import NDArray
from warnings import warn

from dictodot import DotDict

from timstofu.itertools import iter_stencil_indices
from timstofu.math import div
from timstofu.math import horner
from timstofu.math import merge_intervals
from timstofu.math import mod
from timstofu.math import mod_then_div
from timstofu.math import pack
from timstofu.math import unpack_np
from timstofu.misc import filtering_str
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import minimal_uint_type_from_list
from timstofu.numba_helper import permute_into
from timstofu.sort import _grouped_sort
from timstofu.sort import argcountsort
from timstofu.sort import is_lex_nondecreasing
from timstofu.stats import count1D
from timstofu.stats import get_index


from timstofu.sort import grouped_argsort


@numba.njit(parallel=True)
def repivot(
    array: NDArray,
    maxes: NDArray,
    permutation: NDArray,
    output: NDArray,
) -> None:
    assert len(array) == len(output)
    assert len(permutation) == len(maxes)
    permuted_maxes = maxes[permutation]
    for i in numba.prange(len(array)):
        coefs = unpack_np(array[i], maxes)
        output[i] = pack(coefs[permutation], permuted_maxes)
    return output


# TODO: we should revise the code so that maxes are really maxes not maxes + 1, which would be shape.


@dataclass
class Pivot:
    """This class could also be used for argsorts."""

    array: NDArray
    maxes: tuple[int]
    columns: tuple[str]
    counts: DotDict = field(default_factory=DotDict)

    def __post_init__(self):
        assert set(self.columns) == set(
            self.counts
        ), "We work under assumptions that for each column there shall be a count. Use `.new` or adapt it."
        self._indices = {}

    @classmethod
    def new(
        cls,
        _maxes: tuple[int, ...] | None = None,
        _array: NDArray | None = None,
        counts: DotDict | dict[str, NDArray] = DotDict(),
        **data: NDArray,
    ):
        N = None
        for arr in data.values():
            N = len(arr) if N is None else N
            assert len(arr) == N
        _maxes = (
            tuple(int(np.max(arr)) + 1 for arr in data.values())
            if _maxes is None
            else _maxes
        )
        _array = (
            np.empty(shape=N, dtype=minimal_uint_type_from_list(_maxes))
            if _array is None
            else _array
        )
        assert len(_array) == N
        counts = DotDict(counts)
        for c in counts:
            assert c in data
        for c, arr in data.items():
            counts[c] = counts.get(c, count1D(arr))
        return cls(
            array=horner(tuple(data.values()), _maxes, _array),
            maxes=_maxes,
            columns=tuple(data),
            counts=counts,
        )

    def __repr__(self):
        return f"Pivot[{':'.join(self.columns)}]"

    def permute(self, permutation, _array: NDArray | None = None):
        self.array = permute_into(
            xx=self.array,
            permutation=permutation,
            yy=_array,
        )

    # we can also reconstruct from arrays: that will be faster still.
    def repivot(self, new_columns_order, new_array: NDArray | None = None) -> Pivot:
        assert set(new_columns_order) == set(self.columns)
        if new_columns_order == self.columns:
            return self
        permutation = np.array([self.columns.index(c) for c in new_columns_order])
        new_array = repivot(
            array=self.array,
            maxes=np.array(self.maxes),
            permutation=permutation,
            output=self.array.copy() if new_array is None else new_array,
        )
        return self.__class__(
            array=new_array,
            maxes=tuple(np.array(self.maxes)[permutation]),
            columns=tuple(new_columns_order),
            counts=self.counts,
        )

    # we could force getting all counts and indices at new? simplifies code...
    def sort(self) -> None:
        if self.is_sorted():
            return None
        first_col_name = self.columns[0]
        first_col = self.extract(first_col_name)
        counts = self.counts[first_col_name]
        index = get_index(counts)
        if not is_lex_nondecreasing(first_col):  # need to presort on that dim
            first_col_order = argcountsort(first_col, counts)
            self.permute(first_col_order)
        _grouped_sort(self.array, index, self.array)

    def get_index(self, column):
        assert column in self.columns, f"Accepted column names are `{self.column}`."
        if column not in self._indices:
            self._indices[column] = get_index(self.counts[column])
        return self._indices[column]

    def is_sorted(self):
        return is_lex_nondecreasing(self.array)

    def argsort(
        self,
        order: NDArray | None = None,
        return_trivial: bool = False,
    ):
        if self.is_sorted():
            if return_trivial:
                return np.arange(len(self), dtype=get_min_int_data_type(len(self)))

            raise ValueError(
                "The data is sorted and you don't need an argsort. We can give you one if you choose `return_trivial=True`."
            )

        index = get_index(self.counts[self.columns[0]])
        if order is None:
            order = np.empty(
                shape=len(self), dtype=get_min_int_data_type(len(self), signed=False)
            )
        assert len(order) == len(self)
        assert index[-1] == len(self)
        grouped_argsort(self.array, index, order)
        return order

    def __len__(self):
        return len(self.array)

    # def index(self, column):
    #     assert column in self.columns
    #     return get_index(self.counts[column])

    @property
    def col2max(self):
        return DotDict(zip(self.columns, self.maxes))

    def extract(
        self,
        column: str,
        out: NDArray | None = None,
        indices: NDArray | slice | None = None,
    ) -> NDArray:
        """Extract the data from a given dimension.


        Data is kept merged. So simply unmerge that part you want.

        Parameters:
            indices (NDArray|slice|None): A subset of array to consider.
        """
        for k, col in enumerate(self.columns):
            if col == column:
                break
        else:
            raise ValueError(f"Column `{column}` not among [{', '.join(self.columns)}]")

        array = self.array if indices is None else self.array[indices]
        if out is None:
            out = np.empty(
                shape=len(array),
                dtype=get_min_int_data_type(self.maxes[k] - 1, signed=False),
            )
        assert len(out) == len(array)
        if k == 0:
            return div(array, math.prod(self.maxes[1:]), out)
        if k == len(self.columns) - 1:
            return mod(array, int(self.maxes[-1]), out)
        return mod_then_div(
            array,
            math.prod(self.maxes[k:]),
            math.prod(self.maxes[k + 1 :]),
            out,
        )

    def decode(self, indices: NDArray):
        if isinstance(indices, np.ndarray):
            assert len(indices) > 0
        return DotDict({c: self.extract(c, indices=indices) for c in self.columns})

    def get_events_in_box(
        self, center: dict[str, int], radii: dict[str, int]
    ) -> pd.DataFrame:
        leading_column = self.columns[0]
        leading_column_idx = self.get_index(leading_column)
        leading_event_value = center[leading_column]
        filtering_criterion = filtering_str(radii, center)
        events = pd.DataFrame(
            self.decode(
                indices=slice(
                    leading_column_idx[leading_event_value - radii[leading_column]],
                    leading_column_idx[leading_event_value + radii[leading_column] + 1],
                )
            ),
            copy=False,
        ).query(filtering_criterion)
        return events, filtering_criterion

    def __getitem__(self, columns: str | list[str]) -> NDArray:
        if isinstance(columns, str):
            columns = [columns]
        return DotDict({c: self.extract(c) for c in columns})

    def get_stencil_diffs(self, **radii):
        assert len(
            radii
        ), "Pass in radii mapping, e.g. tof=2, scan=2, frame=2, in order of the index."
        for (c, radius), col in zip(radii.items(), self.columns[: len(radii)]):
            assert (
                c == col
            ), f"radii must map the column name to radius in the same order as considered in this pivot: `({self.columns})`"
        return {
            ii: (
                pack((*ii, -last_radius), self.maxes[: len(radii)]),
                pack((*ii, last_radius), self.maxes[: len(radii)]),
            )
            for ii, last_radius in iter_stencil_indices(*radii.values())
        }

    # TODO: don't use that.
    def divide_chunks_to_avoid_race_conditions(
        self,
        chunk_ends: NDArray,
        radii: dict[str, int],
    ) -> tuple[NDArray, NDArray]:
        """Divide chunks of data to avoid race conditions.

        Some operations might compete for filling value tables.
        If that can happen, as with coloring, different threads might get into race conditions.
        To avoid threads to work on the same data, it can make sense to split the chunks they are supposed to work on into pre-processing ones and the remained.
        The fist group of indices are ascertained not to compete for results and prefill the results tables.
        The remaining sets of indices are also ascertained not to compete for result tables and also will have well defined starting conditions.

        Parameters:

        """
        raise NotImplementedError("This makes little sense to use.")
        assert (
            len(chunk_ends.shape) == 2
        ), "Chunks must be a 2D data: each row corresponds to start-end interval."
        assert len(chunk_ends) > 1, "Submit chunks with more than one interval.)"
        right_chunk_ends = chunk_ends[:-1, 1]
        leading_dim_name = self.columns[0]
        leading_dim_idx = get_index(self.counts[leading_dim_name])
        leading_dim_values_at_right_chunk_ends = self.extract(
            "tof", indices=right_chunk_ends
        )

        left_buffered_values = np.maximum(
            leading_dim_values_at_right_chunk_ends - radii[leading_dim_name], 0
        )
        right_buffered_values = np.minimum(
            leading_dim_values_at_right_chunk_ends + radii[leading_dim_name], len(self)
        )

        lefts = leading_dim_idx[left_buffered_values]
        rights = leading_dim_idx[right_buffered_values]
        lefts, rights = merge_intervals(
            lefts, rights
        )  # no need to optimize as long as chukns are separate.
        preclustering_chunk_ends = np.stack((lefts, rights), axis=1)

        remaining_ends = preclustering_chunk_ends.copy()
        if chunk_ends[0, 0] < remaining_ends[0, 0]:
            remaining_ends = np.insert(
                remaining_ends, 0, (chunk_ends[0, 0], chunk_ends[0, 0]), axis=0
            )
        if remaining_ends[-1, -1] < chunk_ends[-1, -1]:
            remaining_ends = np.append(
                remaining_ends,
                np.array([[chunk_ends[-1, -1], chunk_ends[-1, -1]]]),
                axis=0,
            )
        remaining_chunk_ends = np.stack(
            (remaining_ends[:-1, 1], remaining_ends[1:, 0]), axis=1
        )

        return preclustering_chunk_ends, remaining_chunk_ends


def test_Pivot():
    N = 100
    max_A = 40
    max_B = 20
    max_C = 10
    maxes = dict(A=max_A, B=max_B, C=max_C)
    inputs = {c: np.random.choice(_max, size=N) for c, _max in maxes.items()}
    pivot = Pivot.new(**inputs)
    np.testing.assert_equal(
        pivot.array, (inputs["A"] * max_B + inputs["B"]) * max_C + inputs["C"]
    )
    for c in maxes:
        np.testing.assert_equal(pivot.extract(c), inputs[c])
    assert pivot.col2max == maxes
    repivot = pivot.repivot(("B", "C", "A"))
    np.testing.assert_equal(
        repivot.array, (inputs["B"] * max_C + inputs["C"]) * max_A + inputs["A"]
    )
    for c in maxes:
        np.testing.assert_equal(repivot.extract(c), inputs[c])
    assert not repivot.is_sorted()
    repivot.sort()
    assert repivot.is_sorted()
