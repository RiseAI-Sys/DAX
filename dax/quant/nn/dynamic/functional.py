from typing import List, Optional, Tuple, Union

import torch

from ...qscheme import FP8_QSCHEME, INT8_QSCHEME, get_block_size, per_tensor_symmetric
from ..quantized.functional import (
    FP8_ALLOWED_DTYPES,
    quantize_to_fp8,
    quantize_to_int8,
)


def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, (
                f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            )
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims


def _dynamic_quantize_func(
    input: Optional[torch.Tensor],
    block_size: List[int],
    target_dtype: torch.dtype,
    quant_max: Optional[Union[int, float, bool]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library

    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. find min_val/max_val based on the dimension for reduction
    3. calculate quantization parameters based on min_val/max_val
    4. quantize the input based on the quantization parameters
    5. reshape the quantized result to origianl shape

    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
          e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of scale_dtype.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
        quantized input and its scale Tensor with requested dtype
    """

    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"

    if scale_dtype is None:
        scale_dtype = input.dtype
    if eps is None:
        eps = torch.finfo(scale_dtype.dtype).eps

    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)

    ## alternative implementation, hard to read
    # min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    # max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    # amax = torch.max(-min_val, max_val)

    amax = input.abs().amax(dim=reduction_dims, keepdim=False)

    scale = amax / float(quant_max)
    scale = torch.clamp(scale, min=eps)
    if scale.dtype != scale_dtype:
        scale = scale.to(scale_dtype)

    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1

    scale = scale.view(shape_after_reduction)

    if target_dtype in FP8_ALLOWED_DTYPES:
        return (
            quantize_to_fp8(input, scale, target_dtype).view(original_shape),
            scale,
        )
    elif target_dtype == torch.int8:
        return (
            quantize_to_int8(input, scale, target_dtype).view(original_shape),
            scale,
        )
    else:
        raise NotImplementedError(f"Unsupported dtype {target_dtype}")


# @torch.compile()
def dynamic_quantize_to_fp8(
    input_float: torch.Tensor,
    qscheme: str = per_tensor_symmetric,
    block_size: List[int] = None,
    target_dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
    scale_dtype: Optional[torch.dtype] = torch.float32,
):
    if block_size is None:
        assert qscheme is not None and qscheme in FP8_QSCHEME, (
            f"Invalid qscheme: {qscheme}, expected {FP8_QSCHEME}"
        )

        block_size = get_block_size(input_float.shape, qscheme)
    if target_dtype in FP8_ALLOWED_DTYPES:
        return _dynamic_quantize_func(
            input_float,
            block_size=block_size,
            target_dtype=target_dtype,
            quant_max=torch.finfo(target_dtype).max,
            eps=torch.finfo(scale_dtype).eps,
            scale_dtype=scale_dtype,
        )
    else:
        raise NotImplementedError(
            f"Unsupported dtype {target_dtype} for dynamic_quantize_to_fp8"
        )


# @torch.compile()
def dynamic_quantize_to_int8(
    input_float: torch.Tensor,
    qscheme: str = per_tensor_symmetric,
    block_size: List[int] = None,
    target_dtype: Optional[torch.dtype] = torch.int8,
    scale_dtype: Optional[torch.dtype] = torch.float32,
):
    if block_size is None:
        assert qscheme is not None and qscheme in INT8_QSCHEME, (
            f"Invalid qscheme: {qscheme}, expected {INT8_QSCHEME}"
        )

        block_size = get_block_size(input_float.shape, qscheme)
    if target_dtype == torch.int8:
        return _dynamic_quantize_func(
            input_float,
            block_size=block_size,
            target_dtype=target_dtype,
            quant_max=torch.iinfo(target_dtype).max,
            eps=torch.finfo(scale_dtype).eps,
            scale_dtype=scale_dtype,
        )
    else:
        raise NotImplementedError(
            f"Unsupported dtype {target_dtype} for dynamic_quantize_to_int8"
        )
