import numpy as np
import time



@njit
def shift_op(arr, n):
    result = np.empty_like(arr)
    for i in range(arr.size):
        result[i] = arr[i] >> n
    return result


@njit
def div_op(arr, n):
    result = np.empty_like(arr)
    for i in range(arr.size):
        result[i] = arr[i] // (2**n)
    return result



arr = np.arange(1_000_000, dtype=np.int32)
n = 3

%%timeit
_ = shift_op(arr, n)

arr = np.arange(1_000_000, dtype=np.int32)
%%timeit
_ = div_op(arr, n)

# bit packing is more than 12 times faster!!!
# change to that.

@numba.njit
def encode_array(data: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Encode each row of `data` into a single uint64.
    `data`: shape (n_rows, n_fields)
    `widths`: shape (n_fields,), must sum to ≤ 64
    """
    n_rows, n_fields = data.shape
    output = np.zeros(n_rows, dtype=np.uint64)
    
    for i in range(n_rows):
        shift = 0
        value = 0
        for j in range(n_fields):
            val = data[i, j]
            width = widths[j]
            if val >= (1 << width):
                raise ValueError("Value exceeds bit width")
            value |= (val & ((1 << width) - 1)) << shift
            shift += width
        output[i] = value
    return output


@njit
def decode_array(packed: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Decode each uint64 in `packed` into a row of values.
    `packed`: shape (n_rows,)
    `widths`: shape (n_fields,)
    Returns array of shape (n_rows, n_fields)
    """
    n_rows = packed.shape[0]
    n_fields = widths.shape[0]
    output = np.zeros((n_rows, n_fields), dtype=np.uint64)
    
    for i in range(n_rows):
        shift = 0
        for j in range(n_fields):
            width = widths[j]
            mask = (1 << width) - 1
            output[i, j] = (packed[i] >> shift) & mask
            shift += width
    return output


values = np.array([
    [1, 3, 15],
    [0, 7, 255],
    [5, 0, 1],
], dtype=np.uint64)

import numba

from timstofu.math import bit_width

bit_widths = bit_width(maxes)

encode_array(index_data, bit_widths)


from numpy.typing import NDArray

@numba.njit
def bitpack(xx, bit_widths) -> np.uint64:
    """
    Encode an array of data-points `xx` into a single np.uint64.

    Parameters:
        xx (iterable): unsigned integers to pack.
        bit_widths: maximal number of bits needed to represent elements of xx.
    """
    shift = 0
    result = 0
    for x, width in zip(xx, bit_widths):
        if x >= (1 << width):
            raise ValueError("Value exceeds bit width")
        result |= (x & ((1 << width) - 1)) << shift
        shift += width
    return result


@numba.njit
def bitpack_arrays(bit_widths, xx, yy, zz, *other_arrays):
    temp = np.empty(3+len(other_arrays), dtype=np.uint64)
    bitpacked_numbers = np.empty(len(xx), np.uint64)
    for i in range(len(xx)):
        temp[0] = xx[i]
        temp[1] = yy[i]
        temp[2] = zz[i]
        if len(other_arrays):
            for j, arr in enumerate(*other_arrays):
                temp[3+j] = arr[j]
        bitpacked_numbers[i] = bitpack(temp, bit_widths)
    return bitpacked_numbers


maxes = tuple(map(np.max, (tofs, scans, urts)))

bit_widths = bit_width(maxes)

%%timeit
z = bitpack_arrays(bit_widths, tofs, scans, urts)


from timstofu.math import horner


maxes
maxes2 = 2**bit_widths
%%timeit
array = np.empty(len(tofs), np.uint64)
y = horner((tofs, scans, urts), maxes2, array)

x = bitpack_arrays(bit_widths[::-1], urts, scans, tofs)
mask = x != y
x[mask]
y[mask]

tofs[mask], scans[mask], urts[mask]


bitpack(np.array([302022,53,0]), )
bitpack([1], [1])

_input = [1, 3, 15]
bin_widths = list(map(bit_width,_input))
bit_packed_numbers = pack_bits(_input, bin_widths)
bit_packed_number_bin = bin(bit_packed_numbers)
bin_idx = get_index(bin_widths)



for i, bit_packed_number in enumerate(map(bin, _input)):
    extract = bit_packed_number_bin[bin_idx[i]:bin_idx[i+1]]
    print(bit_packed_number)




bin(x)[:3]
