import numba as nb
import numpy as np


# --- Insertion-based top-k (good for small k) ---
@nb.njit(inline="always")
def _topk_insertion(values, k):
    n = values.shape[0]
    if k > n:
        k = n

    top_vals = np.empty(k, dtype=values.dtype)
    top_idx = np.empty(k, dtype=np.int64)
    for i in range(k):
        top_vals[i] = -np.inf
        top_idx[i] = -1

    for i in range(n):
        val = values[i]
        if val > top_vals[k - 1]:
            j = k - 1
            while j > 0 and val > top_vals[j - 1]:
                top_vals[j] = top_vals[j - 1]
                top_idx[j] = top_idx[j - 1]
                j -= 1
            top_vals[j] = val
            top_idx[j] = i
    return top_idx


# --- Heap helpers (good for large k) ---
@nb.njit(inline="always")
def _siftdown(heap_vals, heap_idx, startpos, pos):
    new_val = heap_vals[pos]
    new_idx = heap_idx[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        if new_val < heap_vals[parentpos]:
            heap_vals[pos] = heap_vals[parentpos]
            heap_idx[pos] = heap_idx[parentpos]
            pos = parentpos
            continue
        break
    heap_vals[pos] = new_val
    heap_idx[pos] = new_idx


@nb.njit(inline="always")
def _siftup(heap_vals, heap_idx, pos):
    endpos = heap_vals.shape[0]
    startpos = pos
    new_val = heap_vals[pos]
    new_idx = heap_idx[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and heap_vals[rightpos] < heap_vals[childpos]:
            childpos = rightpos
        if heap_vals[childpos] < new_val:
            heap_vals[pos] = heap_vals[childpos]
            heap_idx[pos] = heap_idx[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        else:
            break
    heap_vals[pos] = new_val
    heap_idx[pos] = new_idx
    _siftdown(heap_vals, heap_idx, startpos, pos)


@nb.njit(inline="always")
def _topk_heap(values, k):
    n = values.shape[0]
    if k > n:
        k = n

    heap_vals = values[:k].copy()
    heap_idx = np.arange(k)
    for i in range((k - 2) // 2, -1, -1):
        _siftup(heap_vals, heap_idx, i)

    for i in range(k, n):
        if values[i] > heap_vals[0]:
            heap_vals[0] = values[i]
            heap_idx[0] = i
            _siftup(heap_vals, heap_idx, 0)

    # sort descending before returning
    for i in range(k):
        for j in range(i + 1, k):
            if heap_vals[j] > heap_vals[i]:
                heap_vals[i], heap_vals[j] = heap_vals[j], heap_vals[i]
                heap_idx[i], heap_idx[j] = heap_idx[j], heap_idx[i]
    return heap_idx


# --- Hybrid selector ---
@nb.njit
def topk_indices_fast(values, k):
    n = values.shape[0]
    if k >= n // 2:
        # Full sort when k is large
        idx = np.arange(n)
        vals = values.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if vals[j] > vals[i]:
                    vals[i], vals[j] = vals[j], vals[i]
                    idx[i], idx[j] = idx[j], idx[i]
        return idx[:k]
    elif k < 64:
        # Small k → insertion method
        return _topk_insertion(values, k)
    else:
        # Large k but much smaller than n → heap method
        return _topk_heap(values, k)
