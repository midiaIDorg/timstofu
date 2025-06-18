"""
%load_ext autoreload
%autoreload 2
"""
import numba
import numpy as np
import numpy.typing as npt
import shutil

from math import inf
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from numba_progress import ProgressBar
from pathlib import Path
from timstofu.sort_and_pepper import argcountsort3D
from timstofu.sort_and_pepper import is_lex_nondecreasing
from timstofu.sort_and_pepper import lexargcountsort2D
from timstofu.sort_and_pepper import lexargcountsort2D_to_3D
from timstofu.sort_and_pepper import test_count_unique_for_indexed_data
from timstofu.stats import _count_unique
from timstofu.stats import count_unique_for_indexed_data
from timstofu.stats import zeros_copy
from timstofu.tofu import LexSortedClusters
from timstofu.tofu import LexSortedDataset

from timstofu.mmapuccino import MmapedArrayValuedDict
from timstofu.mmapuccino import empty

sorted_clusters = LexSortedClusters.from_tofu("/tmp/test_blah.tofu")
deduplicated_clusters_in_ram = sorted_clusters.deduplicate() # TODO: missing memmapped equivalent.

memmaped_folder = Path("/tmp/test.tofu")
shutil.rmtree(memmaped_folder)
memmaped_folder.mkdir()
md = MmapedArrayValuedDict(memmaped_folder)
deduplicated_clusters_on_disk = sorted_clusters.deduplicate(_empty=md.get_empty())










# TODO: it would be nice to have a mechanism to make the decision about the memmapped serializer outside the function, to support mine and Michals when he does it.
# Have it!

# inaczej: po prostu podać ścieżkę?

x = None
y = 10 if x else 2








# OK, so what should read this folder?

memmaped_dir = MemmapedDir("/home/matteo/test2.tofu", force=True)
empty = memmaped_dir.get_memmapped_empty()
test = empty('test', shape=1000, dtype=np.uint32)
memmaped_dir.data

t = lambda name,*args,**kwargs: print(name, args, kwargs)
t(10,20,30,a=232)

h = empty(shape=100000000, dtype=np.int32)
%%timeit
h[:] = 0

from timstofu.numba_helper import overwrite
overwrite(h)



%%timeit
zero_out(h)



test = empty(shape=2,dtype=np.uint32, name=..)
dsfdstes = empty(shape=2,dtype=np.uint32, name=..)

with MemmapedAllocator(folder="/dfadfaf") as ta:
    ta["test"] = ta.empty(shape=2,dtype=np.uint32)
    ta.test
    ta.data





@contextmanager
def TrivialAllocatorContextManager():
    yield 

# this leaves allocation to the user
context_provider = Context(folder="dupa")

with context_provider as context:
    context["variable_1"] = context.zeros(shape, dtype)
    context["variable_2"] = context.empty(shape, dtype)





# OK, simply need to provide some functions to create data.
def test(path):
    return np.memmap(
        path,
        mode="w+",
        dtype=np.uint32,
        shape=2,
    )


# no, we need a context that will make those instead! So exactly like MemmappedContext.
Context = IdentityContext(
    dedup_tofs=np.empty(
        dtype=sorted_tofs.dtype,
        shape=deduplicated_event_count,
    ),
    dedup_intensities=np.zeros(
        dtype=sorted_intensities.dtype,
        shape=deduplicated_event_count,
    ),
)


class NoneField:
    def __getattr__(self, name):
        return None


x = NoneField()
print(x.field)  # None
print(x.anything)  # None
print(x["anything"])  # None


with NoneContext() as mm:
    print(mm.a)
    print(mm["b"])

# somehow, the contexts should work differently.
# For example, they should automate some procedures.
# context.empty()
# problem is to leave the programme space to do things.
# likely provide differnt code path for a context and not context.
