from typing import Tuple, Union

import torch

per_tensor_symmetric = "dax.quant.per_tensor_symmetric"
per_token_symmetric = "dax.quant.per_token_symmetric"
per_channel_symmetric = "dax.quant.per_channel_symmetric"


def get_block_size(
    input_shape: Tuple[int, ...], qscheme: Union[str, torch.qscheme]
) -> Tuple[int, ...]:
    """Get the block size based on the input shape and qscheme type.

    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        qscheme: The qscheme type of the quantization
    """
    if qscheme == per_tensor_symmetric:
        return input_shape
    elif qscheme == per_token_symmetric:
        return (1,) * (len(input_shape) - 1) + (input_shape[-1],)
    elif qscheme == per_channel_symmetric:
        return input_shape[:-1] + (1,)

    raise ValueError(f"Unsupported QScheme: {qscheme}")


FP8_QSCHEME = [
    per_tensor_symmetric,
    per_token_symmetric,
    per_channel_symmetric,
]
INT8_QSCHEME = [
    per_tensor_symmetric,
    per_token_symmetric,
    per_channel_symmetric,
]
