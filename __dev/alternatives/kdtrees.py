from opentimspy import OpenTIMS
from sklearn.neighbors import KDTree

import pandas as pd

folder_dot_d = "/home/matteo/data_for_midiaID/F9477.d"
raw_data = OpenTIMS(folder_dot_d)
X = raw_data.query(raw_data.ms1_frames, columns=("frame","scan", "tof"))
X = pd.DataFrame(X, copy=False)

%%time
kd = KDTree(X)
