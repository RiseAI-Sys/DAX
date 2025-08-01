import os
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from typing import Optional, TypeVar

from loguru import logger

DIST_COORDINATOR = None

# Helper to turn Optional[T] into T when we know None either isn't
# possible or should trigger an exception.
T = TypeVar("T")


def not_none(obj: Optional[T]) -> T:
    if obj is None:
        raise TypeError(
            "Invariant encountered: value was None when it should not be")
    return obj


class Launcher:
    torchrun: str = "torchrun"
    accelerator: str = "accelerator"


class AcceleratorLauncher:
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def get_world_size(self):
        return self.accelerator.num_processes

    def get_rank(self):
        return self.accelerator.process_index

    def get_local_rank(self):
        return self.accelerator.local_process_index

    def is_main_process(self):
        return self.accelerator.is_main_process


class TorchLauncher:
    def get_world_size(self):
        return int(os.environ["WORLD_SIZE"])

    def get_rank(self):
        return int(os.environ["RANK"])

    def get_local_rank(self):
        return int(os.environ["LOCAL_RANK"])

    def is_main_process(self):
        return self.get_rank() == 0


class DistCoordinator:
    def __init__(self, launcher: Launcher, **kwargs):
        if launcher == Launcher.torchrun:
            self.launcher = TorchLauncher()
        elif launcher == Launcher.accelerator:
            accelerator = kwargs["accelerator"]
            self.launcher = AcceleratorLauncher(accelerator)
        else:
            raise TypeError(
                f"launcher {launcher} not support")

    def get_world_size(self):
        return self.launcher.get_world_size()

    def get_rank(self):
        return self.launcher.get_rank()

    def get_local_rank(self):
        return self.launcher.get_local_rank()

    def is_main_process(self):
        return self.launcher.is_main_process()


def is_dist_coordinator_initialized():
    global DIST_COORDINATOR
    return DIST_COORDINATOR is not None


def set_rank_env(k, v):
    os.environ[k] = v


def initialize_dist_coordinator(launcher: Launcher = None, **launcher_kwargs):
    if not is_dist_coordinator_initialized():
        global DIST_COORDINATOR
        if launcher is None:
            launcher = Launcher.torchrun

        DIST_COORDINATOR = DistCoordinator(launcher, **launcher_kwargs)

        if launcher == Launcher.accelerator and DIST_COORDINATOR.get_world_size() == 1:
            set_rank_env("WORLD_SIZE", "1")
            set_rank_env("RANK", "0")
            set_rank_env("LOCAL_RANK", "0")
            set_rank_env("LOCAL_WORLD_SIZE", "1")
            set_rank_env("MASTER_ADDR", "127.0.0.1")
            set_rank_env("MASTER_PORT", "29500")

    else:
        if DIST_COORDINATOR.is_main_process():
            logger.warning("coordinator has been initialized")


def init_default_group():
    dist.init_process_group(backend="nccl")


def get_world_size():
    return DIST_COORDINATOR.get_world_size()


def get_rank():
    return DIST_COORDINATOR.get_rank()


def get_local_rank():
    return DIST_COORDINATOR.get_local_rank()


def is_main_process():
    return DIST_COORDINATOR.is_main_process()


def get_default_group():
    return not_none(_get_default_group())


def get_group_size(group):
    return dist.get_world_size(group)


def get_group_rank(group):
    return dist.get_group_rank(group)


def get_group_ranks(group):
    return dist.get_process_group_ranks(group)


def destroy_process_group():
    dist.destroy_process_group()
