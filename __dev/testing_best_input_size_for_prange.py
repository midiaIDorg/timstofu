import numba
import numpy as np
import time


@numba.njit(parallel=True)
def parallel_work(a, b):
    out = np.empty_like(a)
    for i in numba.prange(len(a)):
        out[i] = a[i] * b[i] + np.sin(a[i])  # artificial cost
    return out


sizes = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]

for n in sizes:
    a = np.random.rand(n)
    b = np.random.rand(n)
    start = time.time()
    parallel_work(a, b)
    print(f"{n:>10}: {(time.time() - start)/n*1000:.10f} ms")
