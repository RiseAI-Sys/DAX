import torch

from ...comm.all_to_all import _all_to_all_4D
from ...dist import parallel_state


@torch.compiler.disable
def collect_tokens(x):
    group = parallel_state.get_sequence_parallel_group()
    # (bs, s/N, hc, hd) -> (bs, s, hc/N, hd)
    return _all_to_all_4D(x, 2, 1, group)


@torch.compiler.disable
def collect_heads(x):
    group = parallel_state.get_sequence_parallel_group()
    # (bs, s, hc/N, hd) -> (bs, s/N, hc, hd)
    return _all_to_all_4D(x, 1, 2, group)
