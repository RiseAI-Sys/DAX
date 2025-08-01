def micro_batch_project_O_strategy(tensor, micro_batch_times, gpu_nums):
    """
    Generate micro-batch blocks from the input tensor based on the specified micro-batch times
    and the number of GPUs.

    Parameters:
    - tensor: Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
    - micro_batch_times: Number of micro-batches.
    - gpu_nums: Number of GPUs available.

    Returns:
    - A list of tensors, each representing a micro-batch block.
    """
    b, s, n, d = tensor.shape
    p = gpu_nums
    m = micro_batch_times
    block_size = s // (p * m)

    blocks = []

    for i in range(m):
        indices = []
        for j in range(p):
            start_idx = i * block_size + j * m * block_size
            end_idx = start_idx + block_size
            indices.extend(range(start_idx, end_idx))
        blocks.append(tensor[:, indices, :, :])

    return blocks
