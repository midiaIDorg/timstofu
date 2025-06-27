from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray

from dictodot import DotDict

from timstofu.math import div
from timstofu.math import horner
from timstofu.math import mod
from timstofu.math import mod_then_div
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import permute_into


@dataclass
class Pivot:
    """This class could also be used for argsorts."""

    array: NDArray
    maxes: DotDict
    columns: tuple[str]

    @classmethod
    def new(
        cls,
        _maxes: dict[str, int] | None = None,
        _array: NDArray | None = None,
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
        max_size = math.prod(_maxes)
        _array = (
            np.empty(shape=N, dtype=get_min_int_data_type(max_size))
            if _array is None
            else _array
        )
        assert len(_array) == N
        return cls(
            array=horner(tuple(data.values()), _maxes, _array),
            maxes=_maxes,
            columns=tuple(data),
        )

    def __repr__(self):
        return f"Pivot[{':'.join(self.columns)}]"

    def permute(self, permutation, _array: NDArray | None = None):
        self.array = permute_into(
            xx=self.array,
            permutation=permutation,
            yy=_array,
        )

    def reorder(self, *new_columns) -> Pivot:
        assert set(new_columns) == set(self.columns)
        if new_columns == self.columns:
            return self

    def __len__(self):
        return len(self.array)

    @property
    def col2max(self):
        return DotDict(zip(self.columns, self.maxes))

    def extract(self, column: str, out: NDArray | None = None) -> NDArray:
        for k, col in enumerate(self.columns):
            if col == column:
                break
        else:
            raise ValueError(f"Column `{column}` not among [{', '.join(self.columns)}]")
        if out is None:
            out = np.empty(
                shape=len(self.array), dtype=get_min_int_data_type(self.maxes[k])
            )
        assert len(out) == len(self)
        if k == 0:
            return div(self.array, math.prod(self.maxes[1:]), out)
        if k == len(self.columns) - 1:
            return mod(self.array, int(self.maxes[-1]), out)
        return mod_then_div(
            self.array,
            math.prod(self.maxes[k:]),
            math.prod(self.maxes[k + 1 :]),
            out,
        )


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
