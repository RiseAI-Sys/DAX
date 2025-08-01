import torch
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from loguru import logger

from .coordinator import get_world_size, get_local_rank, is_main_process
from ..config import ParallelConfig


PARALLEL_STATE = None


class MeshDim:
    dp: str = "dp"
    sp: str = "sp"


class ParallelState:
    def __init__(self, parallel_config: ParallelConfig):
        self.parallel_config = parallel_config
        self.world_size = get_world_size()

        self.validate()
        self.device_type = "cuda"
        self.mesh = self.build_mesh()

    def validate(self):
        dp, sp = self.parallel_config.dp_degree, self.parallel_config.sp_degree
        assert (sp  <=
                self.world_size), f"Invalid parallel dims: {sp} > WORLD_SIZE({self.world_size})"
        if dp == -1:
            self.parallel_config.dp_degree = dp = self.world_size // (
                sp)
        assert dp >= 1, dp
        assert sp >= 1, sp
        assert (
            dp * sp == self.world_size
        ), f"Invalid parallel dims: {MeshDim.dp}({dp}) * {MeshDim.sp}({sp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self):
        mesh_shape = []
        mesh_dim_names = []
        for d, name in zip(
            [self.parallel_config.dp_degree, self.parallel_config.sp_degree], [MeshDim.dp, MeshDim.sp]
        ):
            if d >= 1:
                mesh_shape.append(d)
                mesh_dim_names.append(name)
        if is_main_process():
            logger.info(
                f"Building {len(mesh_shape)}-D device mesh with {mesh_dim_names}, {mesh_shape}")
        mesh_dim_names = tuple(mesh_dim_names)
        return init_device_mesh(self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names)

    def get_mesh(self, mesh_dim):
        return self.mesh[mesh_dim]

    def get_group(self, mesh_dim):
        return self.get_mesh(mesh_dim).get_group()

    def get_group_size(self, mesh_dim):
        return self.get_mesh(mesh_dim).size()

    def get_group_rank(self, mesh_dim):
        return self.get_mesh(mesh_dim).get_local_rank()


def is_parallel_state_initialized():
    global PARALLEL_STATE
    return PARALLEL_STATE is not None


def initialize_parallel_state(parallel_config: ParallelConfig):
    if not is_parallel_state_initialized():
        global PARALLEL_STATE

        PARALLEL_STATE = ParallelState(parallel_config)

        torch.cuda.set_device(get_local_rank())
    else:
        if is_main_process():
            logger.warning("parallel_state has been initialized")


def get_data_parallel_mesh():
    return PARALLEL_STATE.get_mesh(MeshDim.dp)


def get_data_parallel_group():
    return PARALLEL_STATE.get_group(MeshDim.dp)


def get_data_parallel_size():
    return PARALLEL_STATE.get_group_size(MeshDim.dp)


def get_data_parallel_all_ranks():
    return dist.get_process_group_ranks(get_data_parallel_group())


def get_data_parallel_rank():
    return PARALLEL_STATE.get_group_rank(MeshDim.dp)


def get_sequence_parallel_mesh():
    return PARALLEL_STATE.get_mesh(MeshDim.sp)


def get_sequence_parallel_group():
    return PARALLEL_STATE.get_group(MeshDim.sp)


def get_sequence_parallel_size():
    return PARALLEL_STATE.get_group_size(MeshDim.sp)


def get_sequence_parallel_all_ranks():
    return dist.get_process_group_ranks(get_sequence_parallel_group())


def get_sequence_parallel_rank():
    return PARALLEL_STATE.get_group_rank(MeshDim.sp)


@torch.compiler.disable
def is_enable_sequence_parallel():
    return is_parallel_state_initialized() and get_sequence_parallel_size() > 1
