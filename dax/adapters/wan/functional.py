import torch
from diffusers import DiffusionPipeline

from ...parallel.initialize import initialize
from .attn_adapter import replace_wan_attention
from .transformer_adapter import quantize_transformer, replace_transformer


def optimize_pipe(
    pipe: DiffusionPipeline,
    sequence_parallel=True,
    compile=True,
    overlap_comm=True,
    int8_linear=False,
    cache_strategy=None,
    **cache_params,
):
    if sequence_parallel:
        # sp parallel init
        sp_size = initialize()

    replace_transformer(pipe.transformer, cache_strategy, **cache_params)

    if int8_linear:
        quantize_transformer(pipe.transformer)

    if overlap_comm and not sequence_parallel:
        raise ValueError("Overlap communication is only valid in SP mode")
    replace_wan_attention(
        pipe, sp_size if sequence_parallel else 1, overlap_comm=overlap_comm
    )

    if compile:
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs"
        )

    torch.cuda.empty_cache()

    return pipe
