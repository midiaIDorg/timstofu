import numpy as np

from numba import njit


@njit
def process(*args):
    for i in range(len(args[0])):
        # Example: mix float and int arrays
        acc = 0.0
        for arr in args:
            acc += arr[i]  # Will be OK if the operation makes sense
        print(acc)


a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([1, 2, 3], dtype=np.int32)
c = np.array([10.0, 20.0, 30.0], dtype=np.float32)

process(a, b, c)
