from .layer import single_all_to_all, single_all_to_all_pack_qkv


def collect_tokens_packed_qkv(q, k, v, seq_world_size, async_op=False, handle=None):
    """
    Collect packed Q/K/V tokens.

    This function is meant for inference only.
    - async_op is not currently supported and defaults to False.
    - If using async_op, ensure that a handle is provided.

    Usage example:
        handle_warpper = {}
        q, k, v = para_patches.wanx2_1_t2v_opt.collect_tokens_packed_qkv(q, k, v, sp_size)
    """
    return single_all_to_all_pack_qkv(
        q, k, v, seq_world_size, 2, 1, 0, async_op=async_op, handle=handle
    )


def collect_tokens(x, seq_world_size, async_op=False, handle=None):
    """
    Collect input tokens.

    This function is meant for inference only.
    - If using async_op, ensure that a handle is provided.

    Usage example:
        handle_warpper = {}
        x = para_patches.wanx2_1_t2v_opt.collect_tokens(x, sp_size, async_op=True, handle=handle_warpper)

        # Await completion
        handle_warpper['work'].wait()
        default_stream.wait_stream()
        all2all_output_x = handle_warpper['output']
        x = handle_warpper['post_all2all_func'](all2all_output_x)
    """
    return single_all_to_all(
        x, seq_world_size, 2, 1, 0, async_op=async_op, handle=handle
    )


def collect_heads(x, seq_world_size, async_op=False, handle=None):
    """
    Collect heads from the input.

    This function is meant for inference only.
    - async_op is not currently supported and defaults to False.
    - If using async_op, ensure that a handle is provided.

    Usage example:
        x = para_patches.wanx2_1_t2v_opt.collect_heads(x, sp_size)
    """
    return single_all_to_all(
        x, seq_world_size, 1, 2, 0, async_op=async_op, handle=handle
    )
