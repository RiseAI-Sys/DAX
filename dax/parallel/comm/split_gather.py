import torch
import torch.distributed as dist


def _gather(inputs, dim=-1, group=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return inputs

    # all gather
    inputs = inputs.contiguous()
    tensor_list = [torch.empty_like(inputs) for _ in range(world_size)]
    dist.all_gather(tensor_list, inputs, group=group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _split(inputs, dim=-1, group=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return inputs

    # Split along dimension.
    dim_size = inputs.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(inputs, dim_size // world_size, dim=dim)
    rank = dist.get_rank(group)
    output = tensor_list[rank].contiguous()

    return output
