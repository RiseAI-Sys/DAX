import torch

from ...comm.split_gather import _split
from ...dist import parallel_state


@torch.compiler.disable
def split(x, dim):
    return _split(x, dim, parallel_state.get_sequence_parallel_group())
