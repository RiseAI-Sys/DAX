from .layer import split_forward


def split(x, world_size, dim):
    """
    Split the input tensor along the specified dimension.

    This function is intended for inference only.

    Parameters:
    - x: The input tensor to be split.
    - world_size: The number of splits to perform.
    - dim: The dimension along which to split the tensor.

    Returns:
    - The split tensors.

    Notes:
    - async_op is not currently supported and defaults to False.
    - If using async_op, it should be used with a handle.

    Example usage:
        k = para_patches.wanx2_1_t2v_opt.split(k, 8, 2)
    """
    return split_forward(x, world_size=world_size, dim=dim)
