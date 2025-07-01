"""
%load_ext autoreload
%autoreload 2
"""
from opentimspy import OpenTIMS

import math
import matplotlib.pyplot as plt
import numba
import numpy as np

from numba_progress import ProgressBar
from numpy.typing import NDArray

from timstofu.math import discretize
from timstofu.math import log2
from timstofu.math import reduce_resolution
from timstofu.numba_helper import decount
from timstofu.numba_helper import get_min_int_data_type
from timstofu.numba_helper import is_arange
from timstofu.numba_helper import is_permutation
from timstofu.numba_helper import minimal_uint_type_from_list
from timstofu.pivot import Pivot
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.stats import count1D
from timstofu.stats import get_index


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
# perhaps Pivot could store that?
tof_counts = count1D(tofs)
tof_index = get_index(tof_counts)

# TODO: code in aggregation of data after resolution drop.
denser_tofs = reduce_resolution(
    tofs,
    3,
)

################################
# plt.scatter(np.arange(len(tof_counts)), tof_counts, s=0.1, alpha=0.1)
# plt.xlabel("TOF")
# plt.ylabel("COUNT")
# plt.show()

# denser_tof_counts = count1D(denser_tofs)
# plt.scatter(np.arange(len(denser_tof_counts)), denser_tof_counts, s=0.1, alpha=0.1)
# plt.xlabel("TOF / 3")
# plt.ylabel("COUNT")
# plt.show()
disc_intensities = discretize(intensities, transform=log2)
urt_scan_tof_intensity = Pivot.new(
    urt=urts,
    scan=scans,
    tof=tofs,
    log2intensity=disc_intensities,
)

urt_index = get_index(urt_counts)
urt_scan_tof_order = urt_scan_tof_intensity.argsort(urt=urt_index)
assert is_arange(urt_scan_tof_order)


# local densification

# i = 0
# index = urt_index
scan_radius = 2
tof_radius = 2


# somehow I did like the other approach more....

@numba.njit
def foo(_yzw, s, e, scans, tofs, scan_radius, tof_radius, results):
    for i in range(s,e):
        scan = scans[i]
        tof = tofs[i]
        results[i] = _yzw[
            max(scan-scan_radius,0):(scan+scan_radius),
            max(tof-tof_radius,0):(tof+tof_radius),
        ].max()

@numba.njit(parallel=True)
def densify(index, scans, tofs, intensities, shape, foo, foo_args, progress_proxy=None):
    yzw = np.zeros(shape=shape, dtype=intensities.dtype)
    assert len(intensities) == index[-1]
    for i in numba.prange(len(index)):
        s = index[i]
        e = index[i+1]
        _yzw = yzw.copy()
        for idx in range(s, e):
            _yzw[scans[idx], tofs[idx]] = intensities[idx]
        foo(_yzw, s, e, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(1)

results = np.zeros(shape=disc_intensities.shape, dtype=disc_intensities.dtype)
shape = urt_scan_tof_intensity.maxes[1:3]
with ProgressBar(total=len(urt_counts)) as progress_proxy:
    densify(urt_index, scans, tofs, disc_intensities, shape, foo, (scans, tofs, scan_radius, tof_radius, results), progress_proxy)



@numba.njit(parallel=True)
def densifytest(index, shape, progress_proxy):
    yzw = np.zeros(shape=shape, dtype=np.uint8)
    for i in numba.prange(len(index)):
        s = index[i]
        e = index[i+1]
        _yzw = yzw.copy()
        for idx in range(s, e):
            _yzw[0,0]=10
        if progress_proxy is not None:
            progress_proxy.update(1)

# so it is completely important what dims we have here to do dense thing.
with ProgressBar(total=len(tof_counts)) as progress_proxy:
    densifytest(tof_counts, (870,450), progress_proxy)




# this definitely facilitates computations.


# we could pass in the .array instead.


# fix something here, it is very important to have it working.
tof_scan_urt_intensity = Pivot.new(
    tof=tofs,
    scan=scans,
    urt=urts,
    # log2intensity=discretize(intensities, transform=log2),
)
tof_scan_urt_order = tof_scan_urt_intensity.argsort(urt=tof_index)
tof_scan_urt_intensity.permute(tof_scan_urt_order)
if paranoid:
    assert is_permutation(tof_scan_urt_order)
    # this needs to be checked...
    cols = tof_scan_urt_intensity.columns[:3]
    assert is_lex_nondecreasing(
        *tuple(tof_scan_urt_intensity.extract(c).astype(np.uint32) for c in )
    )
    for col in cols:
        if not f"{col}s" in globals():
            continue
        print(f"Comparing counts for `{col}`.")
        scan_counts = count1D(tof_scan_urt_intensity.extract(col))
        scan_counts_orig = count1D(globals()[f"{col}s"])
        np.testing.assert_equal(scan_counts, scan_counts_orig)
    is_lex_nondecreasing(tof_scan_urt_intensity.extract(cols[0]))
    # seems that adding intensity broke something... 

# nothing to do with intensities... how come?
tofs_2 = tof_scan_urt_intensity.extract(cols[0])
i = order_breaker(tofs_2)
tofs_2[i-2:i+2]



@numba.njit
def order_breaker(xx):
    x_prev = xx[0]
    for i,x in enumerate(xx):
        if x < x_prev:
            return i
        x_prev = x
    return -1

# need to implement better check for sortedness.


# we should make some check on data fitting.
tof_scan_urt_intensity.maxes

tof_scan_urt_intensity.array

radii = np.array((1, 4, 5))
shape = radii * 2 + 1
shape = np.append(shape, 2)

xx = urts
yy = scans


W = np.zeros(dtype=np.uintp, shape=shape)
Wflat = W.reshape(-1, 2)

radius = radii[1]
# we have the place, now what?
for widx in range(len(Wflat)):
    while yy


