from ..config import ParallelConfig
from loguru import logger
from .coordinator import is_main_process

RUNTIME_STATE = None


class RuntimeState:
    def __init__(self, parallel_config):
        self.parallel_config = parallel_config

def is_runtime_state_initialized():
    global RUNTIME_STATE
    return RUNTIME_STATE is not None


def initialize_runtime_state(parallel_config: ParallelConfig):
    if not is_runtime_state_initialized():
        global RUNTIME_STATE

        RUNTIME_STATE = RuntimeState(parallel_config)

    else:
        if is_main_process():
            logger.warning("runtime_state has been initialized")


def get_parallel_config():
    return RUNTIME_STATE.parallel_config
