import os
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from ...parallel import dist as para_dist
from ...parallel import patches as para_patches

ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "TORCH_SDPA")

if ATTENTION_BACKEND == "SAGE_ATTN":
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
    except ModuleNotFoundError:
        raise Exception(
            "SageAttention is not installed. To use SageAttention 2.1.1, please compile from source."
        )
elif ATTENTION_BACKEND == "FLASH_ATTN":
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_v3

        ATTENTION_BACKEND = "FLASH_ATTN_V3"
    except ModuleNotFoundError:
        try:
            from flash_attn import flash_attn_func as flash_attn_func_v2

            ATTENTION_BACKEND = "FLASH_ATTN_V2"
        except ModuleNotFoundError:
            raise Exception("FlashAttention is not installed.")
elif ATTENTION_BACKEND != "TORCH_SDPA":
    raise Exception(f"ATTENTION_BACKEND:{ATTENTION_BACKEND} not support.")

print(f"[PreInfo] Use attention backend={ATTENTION_BACKEND}")


class DAXWanT2VCrossAttnProcessor2_0:
    def __init__(self, sp_size=1, overlap_comm=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.sp_size = sp_size
        self.overlap_comm = overlap_comm

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn.add_k_proj is None
        assert encoder_hidden_states is not None

        if not self.overlap_comm:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # batch, num_head, seq_len, head_dim
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if para_dist.parallel_state.is_enable_sequence_parallel():
                query = para_patches.wanx2_1_t2v.collect_tokens(
                    query.transpose(1, 2)
                ).transpose(1, 2)
                key = para_patches.wanx2_1_t2v.split(key, 1)
                value = para_patches.wanx2_1_t2v.split(value, 1)
        else:
            handle_warpper_q = {}

            # compute q
            query = attn.to_q(hidden_states)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # comm q
            query = para_patches.wanx2_1_t2v_opt.collect_tokens(
                query.transpose(1, 2),
                self.sp_size,
                async_op=True,
                handle=handle_warpper_q,
            )

            # compute k
            key = attn.to_k(encoder_hidden_states)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # compute v
            value = attn.to_v(encoder_hidden_states)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # comm k, v
            key = para_patches.wanx2_1_t2v_opt.split(key, self.sp_size, 1)
            value = para_patches.wanx2_1_t2v_opt.split(value, self.sp_size, 1)

            # wait q
            handle_warpper_q["work"].wait()
            all2all_output_q = handle_warpper_q["output"]
            query = handle_warpper_q["post_all2all_func"](all2all_output_q).transpose(
                1, 2
            )

        if ATTENTION_BACKEND == "SAGE_ATTN":
            if attention_mask is not None:
                raise NotImplementedError(
                    "SageAttention does not support `attention_mask`"
                )

            hidden_states = sageattn_qk_int8_pv_fp8_cuda_sm90(
                query,
                key,
                value,
                is_causal=False,
            )
        elif ATTENTION_BACKEND == "FLASH_ATTN_V3":
            hidden_states = flash_attn_func_v3(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                causal=False,
            )[0].transpose(1, 2)
        elif ATTENTION_BACKEND == "FLASH_ATTN_V2":
            hidden_states = flash_attn_func_v2(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                causal=False,
            ).transpose(1, 2)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        if not self.overlap_comm:
            if para_dist.parallel_state.is_enable_sequence_parallel():
                hidden_states = para_patches.wanx2_1_t2v.collect_heads(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

            hidden_states = attn.to_out[0](hidden_states)
        else:
            micro_batch_time = 2
            # make micro batch strategpy
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states_split = (
                para_patches.wanx2_1_t2v_opt.micro_batch_project_O_strategy(
                    hidden_states,
                    micro_batch_time,
                    self.sp_size,
                )
            )
            # wrapper for double buffer
            handle_wrapper = {0: {}, 1: {}}
            # double buffer
            hidden_states_temp_buffer = [None, None]
            current_buffer = 0
            hidden_states_processed = []
            # send first
            hidden_states_temp_buffer[current_buffer] = (
                para_patches.wanx2_1_t2v_opt.collect_heads(
                    hidden_states_split[0],
                    self.sp_size,
                    async_op=True,
                    handle=handle_wrapper[current_buffer],
                )
            )
            # switch buffer
            current_buffer ^= 1
            for i in range(1, micro_batch_time + 1):
                if i < micro_batch_time:
                    # send next
                    hidden_states_temp_buffer[current_buffer] = (
                        para_patches.wanx2_1_t2v_opt.collect_heads(
                            hidden_states_split[i],
                            self.sp_size,
                            async_op=True,
                            handle=handle_wrapper[current_buffer],
                        )
                    )
                    # switch buffer
                    current_buffer ^= 1
                # wait previous
                handle_wrapper[current_buffer]["work"].wait()
                all2all_output_temp = handle_wrapper[current_buffer]["output"]
                hidden_states_temp_buffer[current_buffer] = handle_wrapper[
                    current_buffer
                ]["post_all2all_func"](all2all_output_temp)
                hidden_states_temp_buffer[current_buffer] = (
                    hidden_states_temp_buffer[current_buffer]
                    .flatten(2, 3)
                    .type_as(query)
                )
                # compute previous
                hidden_states_temp_buffer[current_buffer] = attn.to_out[0](
                    hidden_states_temp_buffer[current_buffer]
                )
                hidden_states_processed.append(
                    hidden_states_temp_buffer[current_buffer]
                )
                # switch buffer
                current_buffer ^= 1
            # reshape to origin
            hidden_states = torch.cat(hidden_states_processed, dim=1).contiguous()

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class DAXWanI2VCrossAttnProcessor2_0:
    def __init__(self, sp_size=1, overlap_comm=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.sp_size = sp_size
        self.overlap_comm = overlap_comm

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert attn.add_k_proj is not None
        encoder_hidden_states_img = encoder_hidden_states[:, :257]
        encoder_hidden_states = encoder_hidden_states[:, 257:]
        assert encoder_hidden_states is not None
        assert encoder_hidden_states_img is not None

        if not self.overlap_comm:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # I2V task
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if para_dist.parallel_state.is_enable_sequence_parallel():
                query = para_patches.wanx2_1_t2v.collect_tokens(
                    query.transpose(1, 2)
                ).transpose(1, 2)
                key = para_patches.wanx2_1_t2v.split(key, 1)
                value = para_patches.wanx2_1_t2v.split(value, 1)

                key_img = para_patches.wanx2_1_t2v.split(key_img, 1)
                value_img = para_patches.wanx2_1_t2v.split(value_img, 1)
        else:
            handle_warpper_q = {}

            # compute q
            query = attn.to_q(hidden_states)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # comm q
            query = para_patches.wanx2_1_t2v_opt.collect_tokens(
                query.transpose(1, 2),
                self.sp_size,
                async_op=True,
                handle=handle_warpper_q,
            )

            # compute k
            key = attn.to_k(encoder_hidden_states)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # compute v
            value = attn.to_v(encoder_hidden_states)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # comm k, v
            key = para_patches.wanx2_1_t2v.split(key, 1)
            value = para_patches.wanx2_1_t2v.split(value, 1)

            # compute k_img, v_img
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # comm k_img, v_img
            key_img = para_patches.wanx2_1_t2v.split(key_img, 1)
            value_img = para_patches.wanx2_1_t2v.split(value_img, 1)

            # wait q
            handle_warpper_q["work"].wait()
            all2all_output_q = handle_warpper_q["output"]
            query = handle_warpper_q["post_all2all_func"](all2all_output_q).transpose(
                1, 2
            )

        if ATTENTION_BACKEND == "SAGE_ATTN":
            if attention_mask is not None:
                raise NotImplementedError(
                    "SageAttention does not support `attention_mask`"
                )
            hidden_states_img = sageattn_qk_int8_pv_fp8_cuda_sm90(
                query,
                key_img,
                value_img,
                is_causal=False,
            )
            hidden_states = sageattn_qk_int8_pv_fp8_cuda_sm90(
                query,
                key,
                value,
                is_causal=False,
            )
        elif ATTENTION_BACKEND == "FLASH_ATTN_V3":
            hidden_states_img = flash_attn_func_v3(
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                causal=False,
            )[0].transpose(1, 2)
            hidden_states = flash_attn_func_v3(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                causal=False,
            )[0].transpose(1, 2)
        elif ATTENTION_BACKEND == "FLASH_ATTN_V2":
            hidden_states_img = flash_attn_func_v2(
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                causal=False,
            ).transpose(1, 2)
            hidden_states = flash_attn_func_v2(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                causal=False,
            ).transpose(1, 2)
        else:
            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        if para_dist.parallel_state.is_enable_sequence_parallel():
            hidden_states_img = para_patches.wanx2_1_t2v.collect_heads(
                hidden_states_img.transpose(1, 2)
            ).transpose(1, 2)

            hidden_states = para_patches.wanx2_1_t2v.collect_heads(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
        hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
