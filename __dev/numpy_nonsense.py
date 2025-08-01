import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

perm = np.array([2, 0, 1])
reordered = arr[:, perm]

print(np.shares_memory(arr, reordered))  # False — new view, not in-place

arr[:] = reordered  # assigns values back into the original buffer

print(np.all(arr == reordered))  # True
print(id(arr)) == id(arr)  # True
