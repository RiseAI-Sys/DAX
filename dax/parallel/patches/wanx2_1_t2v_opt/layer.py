import torch
import torch.distributed as dist

from ...dist import parallel_state


def _generate_layout_params(scatter_idx, batch_dim_idx, seq_world_size, input):
    """
    Generate parameters required for `permute` and `reshape` operations,
    which are used to process data before and after `all2all` communication.
    """
    if batch_dim_idx == 0:
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [
                bs,
                seq_world_size,
                global_seq_len // seq_world_size,
                num_local_head,
                head_dim,
            ]
            pre_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [
                bs,
                global_seq_len // seq_world_size,
                seq_world_size * num_local_head,
                head_dim,
            ]
        else:  # scatter_idx >= 2
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, (
                f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            )
            pre_all2all_inp_shape = [
                bs,
                local_seq_len,
                seq_world_size,
                num_total_head // seq_world_size,
                head_dim,
            ]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)

            post_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_res_shape = [
                bs,
                seq_world_size * local_seq_len,
                num_total_head // seq_world_size,
                head_dim,
            ]
    else:  # batch_dim_idx != 0
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [
                seq_world_size,
                global_seq_len // seq_world_size,
                bs,
                num_local_head,
                head_dim,
            ]
            pre_all2all_permute_idx = None

            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [
                bs,
                seq_world_size * global_seq_len,
                num_local_head // seq_world_size,
                head_dim,
            ]
        else:  # scatter_idx >= 2
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, (
                f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            )
            pre_all2all_inp_shape = [
                local_seq_len,
                bs,
                seq_world_size,
                num_total_head // seq_world_size,
                head_dim,
            ]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)
            post_all2all_permute_idx = None
            post_all2all_res_shape = [
                local_seq_len * seq_world_size,
                bs,
                num_total_head // seq_world_size,
                head_dim,
            ]

    return (
        pre_all2all_permute_idx,
        pre_all2all_inp_shape,
        post_all2all_permute_idx,
        post_all2all_res_shape,
    )


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()
        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


@torch.compiler.disable
def all2all_impl(output, input_t, async_op=False):
    """
    Implementation of `all2all` communication.
    """
    group = parallel_state.get_sequence_parallel_group()
    # Documentation: https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
    # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -> all2all -> (P, bs x seq_len/P, hc/P, hs) scatter head
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
    return work


def single_all_to_all_pack_qkv(
    q,
    k,
    v,
    seq_world_size,
    scatter_idx,
    gather_idx,
    batch_dim_idx,
    async_op=False,
    handle=None,
):
    """
    Perform single all-to-all communication for packed Q/K/V tensors.
    """
    # Packed QKV
    input = torch.cat([q, k, v], dim=0).contiguous()

    # Only support MHA (Multi-Head Attention)
    # Get reshape and permute maps
    (
        pre_all2all_permute_idx,
        pre_all2all_inp_shape,
        post_all2all_permute_idx,
        post_all2all_res_shape,
    ) = _generate_layout_params(scatter_idx, batch_dim_idx, seq_world_size, input)

    # Input reshape
    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    # Output reshape function
    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)

    output = torch.empty_like(input_t)

    # Implementation
    work = all2all_impl(output, input_t, async_op=async_op)

    # Output reshape
    res = post_all2all_fun(output)

    # Unpack QKV
    q, k, v = torch.chunk(res, 3, dim=0)

    return q, k, v


def single_all_to_all(
    input,
    seq_world_size,
    scatter_idx,
    gather_idx,
    batch_dim_idx,
    async_op=False,
    handle=None,
):
    """
    Perform single all-to-all communication for input tensors.
    """
    # Only support MHA (Multi-Head Attention)
    # Get reshape and permute maps
    (
        pre_all2all_permute_idx,
        pre_all2all_inp_shape,
        post_all2all_permute_idx,
        post_all2all_res_shape,
    ) = _generate_layout_params(scatter_idx, batch_dim_idx, seq_world_size, input)

    # Input reshape
    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    # Output reshape function
    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)

    output = torch.empty_like(input_t)

    # Implementation
    work = all2all_impl(output, input_t, async_op=async_op)

    res = post_all2all_fun(output)

    # Async handle
    if async_op:
        handle["work"] = work
        handle["output"] = output
        handle["post_all2all_func"] = post_all2all_fun
        return output.view(post_all2all_res_shape)

    return res


@torch.compiler.disable
def _split_forward_impl(tensor_list):
    """
    Implementation for splitting tensors along the sequence dimension.
    """
    group = parallel_state.get_sequence_parallel_group()
    rank = dist.get_rank(group)
    output = tensor_list[rank].contiguous()
    return output


def split_forward(inputs, world_size=1, dim=-1):
    """
    Split input tensor evenly across the specified dimension.
    """
    if world_size == 1:
        return inputs

    # Split along dimension
    dim_size = inputs.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(inputs, dim_size // world_size, dim=dim)

    # Implement split
    output = _split_forward_impl(tensor_list)

    return output
