import torch
from torch import Tensor

FP8_ALLOWED_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


# @torch.compile()
def quantize_to_fp8(x: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    quant_max = torch.finfo(dtype).max
    return (x / scale).clamp(-quant_max, quant_max).to(dtype)


# @torch.compile()
def quantize_to_int8(x: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    quant_min = torch.iinfo(dtype).min
    quant_max = torch.iinfo(dtype).max
    return torch.clamp(torch.round(x * (1.0 / scale)), quant_min, quant_max).to(dtype)
