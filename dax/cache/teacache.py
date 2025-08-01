import numpy as np
import torch


@torch.compiler.disable
def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.is_even:
        return hidden_states + self.previous_residual_even
    else:
        return hidden_states + self.previous_residual_odd


@torch.compiler.disable
def maybe_cache_states(
    self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
) -> None:
    if self.is_even:
        self.previous_residual_even = hidden_states.squeeze(0) - original_hidden_states
    else:
        self.previous_residual_odd = hidden_states.squeeze(0) - original_hidden_states


@torch.compiler.disable
def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
    if not self.enable_teacache:
        return False

    # initialize the coefficients, cutoff_steps, and ret_steps
    coefficients = self.coefficients
    use_ref_steps = self.use_ref_steps
    cutoff_steps = self.cutoff_steps
    ret_steps = self.ret_steps
    teacache_thresh = self.teacache_thresh

    # print(coefficients)
    # print(use_ref_steps)
    # print(cutoff_steps)
    # print(ret_steps)
    # print(teacache_thresh)

    timestep_proj = kwargs["timestep_proj"]
    temb = kwargs["temb"]
    modulated_inp = timestep_proj if use_ref_steps else temb

    if self.cnt % 2 == 0:  # even -> condition
        self.is_even = True
        if self.cnt < ret_steps or self.cnt >= cutoff_steps:
            self.should_calc_even = True
            self.accumulated_rel_l1_distance_even = 0
        else:
            assert self.previous_e0_even is not None, (
                "previous_e0_even is not initialized"
            )
            assert self.accumulated_rel_l1_distance_even is not None, (
                "accumulated_rel_l1_distance_even is not initialized"
            )
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance_even += rescale_func(
                (
                    (modulated_inp - self.previous_e0_even).abs().mean()
                    / self.previous_e0_even.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance_even < teacache_thresh:
                self.should_calc_even = False
            else:
                self.should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
        self.previous_e0_even = modulated_inp.clone()

    else:  # odd -> unconditon
        self.is_even = False
        if self.cnt < ret_steps or self.cnt >= cutoff_steps:
            self.should_calc_odd = True
            self.accumulated_rel_l1_distance_odd = 0
        else:
            assert self.previous_e0_odd is not None, (
                "previous_e0_odd is not initialized"
            )
            assert self.accumulated_rel_l1_distance_odd is not None, (
                "accumulated_rel_l1_distance_odd is not initialized"
            )
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance_odd += rescale_func(
                (
                    (modulated_inp - self.previous_e0_odd).abs().mean()
                    / self.previous_e0_odd.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance_odd < teacache_thresh:
                self.should_calc_odd = False
            else:
                self.should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_odd = modulated_inp.clone()
    self.cnt += 1
    should_skip_forward = False
    if self.is_even:
        if not self.should_calc_even:
            should_skip_forward = True
    else:
        if not self.should_calc_odd:
            should_skip_forward = True

    return should_skip_forward


def reset_cache(pipe):
    if hasattr(pipe.transformer, "cnt"):
        pipe.transformer.cnt = 0
        pipe.transformer.accumulated_rel_l1_distance_even = 0
        pipe.transformer.accumulated_rel_l1_distance_odd = 0
        pipe.transformer.previous_e0_even = None
        pipe.transformer.previous_e0_odd = None
        pipe.transformer.previous_residual_even = None
        pipe.transformer.previous_residual_odd = None
