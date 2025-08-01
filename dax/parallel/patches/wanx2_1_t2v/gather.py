import torch

from ...comm.split_gather import _gather
from ...dist import parallel_state


@torch.compiler.disable
def gather(x, dim):
    return _gather(x, dim, parallel_state.get_sequence_parallel_group())
