"""
%load_ext autoreload
%autoreload 2
"""
import numpy as np

from opentimspy import OpenTIMS

from timstofu.numba_helper import decount
from timstofu.pivot import Pivot
from timstofu.stats import get_index

from timstofu.numba_helper import is_permutation
from timstofu.sort_and_pepper import is_lex_nondecreasing

from timstofu.sort_and_pepper import grouped_argsort


folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)
# LexSortedDataset.from_tdf(folder_dot_d)

urt2frame = raw_data.ms1_frames
cols = raw_data.query(urt2frame, columns=("scan", "tof", "intensity"))
scans, tofs, intensities = cols.values()
urt_counts = raw_data.frames["NumPeaks"][urt2frame - 1]
urt_max = len(urt_counts)
urts = decount(
    np.arange(urt_max, dtype=scans.dtype),
    urt_counts,
)

urt_tof_scan = Pivot.new(
    urt=urts,
    tof=tofs,
    scan=scans,
)
urt_index = get_index(urt_counts)
urt_tof_scan_order = urt_tof_scan.argsort(urt=urt_index)
urt_tof_scan.permute(urt_tof_scan_order)


if paranoid:
    assert is_permutation(urt_tof_scan_order)
    assert is_lex_nondecreasing(urt_tof_scan.extract("tof"))  # boom


# test this function
from timstofu.numba_helper import get_min_int_data_type
from timstofu.stats import count1D

import numba


def grouped_argsort2(xx, group_index, order):
    """Sort arrays.

    Parameters
    ----------
    xx (np.array): Array to be argsorted, grouped by index.
    grouped_index (np.array): 1D array with counts, one field larger than the number of different values of the grouper.
    order (np.array): Place to store results.

    Notes
    -----
    `group_index[i]:group_index[i+1]` returns a view into all members of the i-th group.
    """
    assert len(xx) == len(order)
    assert len(xx) == group_index[-1]
    for i in numba.prange(len(group_index) - 1):
        s = group_index[i]
        e = group_index[i + 1]
        order[s:e] = s + np.argsort(xx[s:e])


def test_grouped_argsort():
    # xx = np.array([1, 1, 1, 1, 2, 2, 2])
    # yy = np.array([2, 1, 2, 1, 1, 2, 1])

    xx = np.random.randint(0, 10, size=100)
    xx.sort()
    yy = np.random.randint(0, 10, size=100)

    # pe
    xx_counts = count1D(xx)
    xx_index = get_index(xx_counts)
    order = np.empty(shape=yy.shape, dtype=get_min_int_data_type(len(yy), signed=False))
    grouped_argsort2(yy, xx_index, order)
    order2 = order.copy()
    grouped_argsort(yy, xx_index, order2)

    order3 = np.lexsort((yy, xx))

    np.testing.assert_equal(xx[order2], xx[order3])
    np.testing.assert_equal(yy[order2], yy[order3])
    # result was not sorted in one dim.
    # so we must start with sorted data as such???


urt_scan_tofs = Pivot.new(
    urt=urts,
    scan=scans,
    tof=tofs,
)
urt_tof_scans = urt_scan_tofs.repivot(("urt", "tof", "scan"))

urt2 = urt_tof_scans.extract("urt").astype(np.uint32)
tof2 = urt_tof_scans.extract("tof")
scan2 = urt_tof_scans.extract("scan").astype(np.uint32)
is_lex_nondecreasing(urt2, tof2, scan2)
# shit. Still need to debug.


# to start with a fresh pivot, we need to sort first dim.
# to do: modify the new?
