import os

from .. import dist
from ..config import ParallelConfig

INITIALIZED = False


def _get_configs(sp_parallel_size):
    parallel_config = ParallelConfig(sp_degree=sp_parallel_size)
    return parallel_config


def initialize(sp_parallel_size=None):
    if is_launched_with_torchrun:
        if not sp_parallel_size:
            sp_parallel_size = int(os.environ["LOCAL_WORLD_SIZE"])

        parallel_config = _get_configs(sp_parallel_size)

        dist.initialize(parallel_config=parallel_config)

        global INITIALIZED
        INITIALIZED = True
        return sp_parallel_size
    else:
        raise ValueError("Not launched with torchrun")


def is_initialized():
    return INITIALIZED


def is_launched_with_torchrun():
    return "TORCHELASTIC_RUN_ID" in os.environ
