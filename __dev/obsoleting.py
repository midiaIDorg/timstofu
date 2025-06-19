"""
%load_ext autoreload
%autoreload 2
"""
from dictodot import DotDict
from timstofu.tofu import CompactDataset

import numpy as np

cd = CompactDataset(counts=np.array([[0, 0, 1], [1, 2, 0]]), columns=DotDict())
