import torch
from diffusers import DiffusionPipeline, WanPipeline, WanImageToVideoPipeline
from diffusers.models.attention_processor import Attention

from ...parallel.initialize import initialize
from .self_attn_adapter import DAXWanSelfAttnProcessor2_0
from .cross_attn_adapter import (
    DAXWanT2VCrossAttnProcessor2_0,
    DAXWanI2VCrossAttnProcessor2_0,
)
from .transformer_adapter import quantize_transformer, replace_transformer


def _replace_wan2_1_attention(pipe, sp_size=1, overlap_comm=False):
    for name, m in pipe.transformer.named_modules():
        if isinstance(m, Attention):
            if name.endswith("attn1"):  # self attn
                m.set_processor(
                    DAXWanSelfAttnProcessor2_0(
                        sp_size=sp_size, overlap_comm=overlap_comm
                    )
                )
            elif name.endswith("attn2"):  # cross attn
                if isinstance(pipe, WanImageToVideoPipeline):
                    m.set_processor(
                        DAXWanI2VCrossAttnProcessor2_0(
                            sp_size=sp_size, overlap_comm=overlap_comm
                        )
                    )
                elif isinstance(pipe, WanPipeline):
                    m.set_processor(
                        DAXWanT2VCrossAttnProcessor2_0(
                            sp_size=sp_size, overlap_comm=overlap_comm
                        )
                    )
                else:
                    raise NotImplementedError()


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
    _replace_wan2_1_attention(
        pipe, sp_size if sequence_parallel else 1, overlap_comm=overlap_comm
    )

    if compile:
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs"
        )

    torch.cuda.empty_cache()

    return pipe
