"""
%load_ext autoreload
%autoreload 2
"""

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from dataclasses import dataclass
from mmapped_df import open_dataset
from pathlib import Path

from timstofu.tofu import CompactDataset
from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset

from opentimspy import OpenTIMS


# precursor_dataset = LexSortedDataset.from_tdf(
#     "/home/matteo/data_for_midiaID/F9477.d", "precursor"
# )
# fragment_dataset = LexSortedDataset.from_tdf(
#     "/home/matteo/data_for_midiaID/F9477.d", "fragment"
# )
# combined_dataset_directly = LexSortedDataset.from_tdf(
#     "/home/matteo/data_for_midiaID/F9477.d", "both"
# )


folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
folder_dot_d = "/home/matteo/data_for_midiaID/O11556.d"

# rm -rf /home/matteo/test.tofu
# rm -rf /home/matteo/O11556.tofu
%%time
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
    output_path="/home/matteo/O11556.tofu",
)

# rm -rf /home/matteo/O11556_with_mz.tofu
%%time
precursor_dataset = LexSortedDataset.from_tdf(
    folder_dot_d=folder_dot_d,
    level="precursor",
    output_path="/home/matteo/O11556_with_mz.tofu",
    satelite_data_dtypes = dict(
        tof=np.uint32,
        intensity=np.uint32,
        mz=np.float64,
    ),
)


precursor_dataset
# why precursor_dataset.columns.mz == 0?

"mz": ["<f4", 371439835],
np.memmap("/home/matteo/O11556_with_mz.tofu/mz.npy", mode="r", dtype="<f4")
# perhaps OpenTIMSpy did not output it.

x = LexSortedDataset.from_tofu("/home/matteo/O11556_with_mz.tofu")
# stupid m/z does not work.


# shape is not properly save, not saved at all as a matter of fuck.
# fix it .

# ls /home/matteo/test.tofu


LexSortedDataset.from_tofu("/home/matteo/test.tofu")

output_path = Path("/home/matteo/test.mmappet")

for file in output_path.glob("*.npy"):
    print(file)
# output_path = None

assert level in ("precursor", "fragment", "both")

raw_data = (
    folder_dot_d if isinstance(folder_dot_d, OpenTIMS) else OpenTIMS(folder_dot_d)
)

match level:
    case "precursor":
        frames = raw_data.ms1_frames
    case "fragment":
        frames = raw_data.ms2_frames
    case "both":
        frames = raw_data.frames["Id"]


if output_path is None:
    frame_scan_to_count = np.zeros(
        dtype=np.uint32,
        shape=(raw_data.max_frame + 1, raw_data.max_scan + 1),
    )
    columns = DotDict(raw_data.query(frames, columns=("tof", "intensity")))
else:
    size = np.sum(raw_data.frames["NumPeaks"][frames - 1])
    frame_scan_to_count = np.memmap(
        output_path / "frame_scan_counts.npy",
        dtype=np.uint32,
        shape=(raw_data.max_frame + 1, raw_data.max_scan + 1),
        mode="w+",
    )
    columns = dict(
        tof=np.memmap(
            output_path / "tof.npy",
            dtype=np.uint32,
            shape=size,
            mode="w+",
        ),
        intensity=np.memmap(
            output_path / "intensity.npy",
            dtype=np.uint32,
            shape=size,
            mode="w+",
        ),
    )
    columns = DotDict(raw_data.query(frames, columns=columns))

if "desc" not in tqdm_kwargs:
    tqdm_kwargs["desc"] = "Counting (frame,scan) pairs among events"
for frame in tqdm(frames, **tqdm_kwargs):
    frame_data = raw_data.query(frame, columns="scan")
    unique_scans, counts = np.unique(frame_data["scan"], return_counts=True)
    frame_scan_to_count[frame, unique_scans] = counts


index = get_precumsums(frame_scan_to_count)

# input_clusters_folder = "/home/matteo/tmp/test1.mmappet"
# clusters_df = open_dataset(input_clusters_folder)


self_assigned_cols = {
    col: np.memmap(
        f"/tmp/test/{col}.npy",
        dtype=np.uint32,
        shape=len(op),
        mode="w+",
    )
    for col in cols
}
