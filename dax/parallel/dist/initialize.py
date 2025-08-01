from ..config import ParallelConfig
from .coordinator import initialize_dist_coordinator, Launcher, is_dist_coordinator_initialized
from .parallel_state import initialize_parallel_state, is_parallel_state_initialized
from .runtime_state import initialize_runtime_state, is_runtime_state_initialized


def initialize(launcher: Launcher = None, parallel_config: ParallelConfig = None, launcher_kwargs: dict = {}):
    if parallel_config is None:
        parallel_config = ParallelConfig()

    initialize_dist_coordinator(launcher, **launcher_kwargs)
    initialize_parallel_state(parallel_config)
    initialize_runtime_state(parallel_config)


def is_initialized():
    return is_dist_coordinator_initialized() and is_parallel_state_initialized() and is_runtime_state_initialized()
