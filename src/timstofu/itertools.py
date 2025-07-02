import itertools


def iter_stencil_indices(*radii: int):
    for r in radii:
        assert isinstance(r, int)
        assert r >= 0
    yield from itertools.product(*(range(-r, r + 1) for r in radii))
