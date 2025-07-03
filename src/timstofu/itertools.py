import itertools

from timstofu.math import pack


def iter_stencil_indices(*radii: int):
    for r in radii:
        assert isinstance(r, int)
        assert r >= 0
    for ii in itertools.product(*(range(-m, m + 1) for m in radii[:-1])):
        yield ii, radii[-1]

    # for i in range(0, radii[0] + 1):
    #     for ii in itertools.product(*(range(-m * (i > 0), m + 1) for m in radii[1:-1])):
    #         yield i, *ii
