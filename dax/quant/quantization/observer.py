import warnings
from typing import Any, Dict, List

import torch
from torch import __version__
from torch.ao.quantization import ObserverBase
from torch.ao.quantization.utils import check_min_max_valid

ALLOWED_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
)

AT_LEAST_TORCH_2_5 = __version__ >= (2, 5)
from ..qscheme import per_channel_symmetric, per_tensor_symmetric, per_token_symmetric


class UniformQuantizationObserverBase(ObserverBase):
    eps: torch.Tensor

    def __init__(
        self,
        dtype=torch.float8_e4m3fn,
        qscheme=per_tensor_symmetric,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        prefix="",
        sync_state=False,
        **kwargs,
    ) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme

        self.register_buffer("eps", torch.tensor([eps], **factory_kwargs))
        assert self.qscheme in (
            per_tensor_symmetric,
            per_channel_symmetric,
            per_token_symmetric,
        ), (
            "Default Observer only works for per_tensor_symmetric/per_channel_symmetric/per_token_symmetric quantization scheme"
        )

        assert self.dtype in ALLOWED_DTYPES, (
            f"Default Observer only works for {ALLOWED_DTYPES} data type"
        )

        self.quant_max = (
            torch.iinfo(self.dtype).max
            if dtype == torch.int8
            else torch.finfo(self.dtype).max
        )
        self.prefix = prefix
        self.sync_state = sync_state

    def _sync_state(self, reduce_op="max"):
        if torch.distributed.is_initialized() and self.sync_state:
            if reduce_op == "max":
                reduce_op = torch.distributed.ReduceOp.MAX
            elif reduce_op == "sum":
                reduce_op = torch.distributed.ReduceOp.SUM
            else:
                raise NotImplementedError("reduce_op must be 'max' or 'sum'")

            torch.distributed.all_reduce(self.min_val, op=reduce_op)
            torch.distributed.all_reduce(self.max_val, op=reduce_op)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per tensor/channel
            max_val: Maximum values per tensor/channel

        Returns:
            scales: Scales tensor of shape (#channels,)
        """
        # Functionally equivalent to 'determine_qparams' in utils.py. Observers must be torchscriptable however and qscheme
        # as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
        # to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code
        # seems unlikey to change (last update over 1 year ago) and when torchscript is fully deprecated we can refactor.
        # TODO(jakeszwe, jerryzh168)
        if not check_min_max_valid(min_val, max_val):
            # return torch.tensor(
            #     [1.0], device=min_val.device.type
            # ), torch.tensor([0], device=min_val.device.type)
            return None

        quant_max = self.quant_max
        amax = torch.max(-min_val, max_val)

        scale = amax / float(quant_max)
        scale = torch.max(scale, self.eps)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=scale.device)

        return scale

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError("Cannot reset min/max values in the given observer.")


class MinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.float8_e4m3fn,
        qscheme=per_tensor_symmetric,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        assert qscheme == per_tensor_symmetric, (
            "MinMaxObserver's qscheme only support per_tensor_symmetric"
        )

        # TODO: MinMaxObserver by itself doesn't support dynamic quantization, but
        # if it's inherited by MovingAverageObserver, and averaging_constant is 1, it
        # supports dynamic quantization, we may need to better error checking here

        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self._sync_state()
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))


class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.float8_e4m3fn,
        qscheme=per_tensor_symmetric,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not qscheme == per_channel_symmetric:
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    per_channel_symmetric,"
            )

        assert AT_LEAST_TORCH_2_5, (
            "per-channel weight scale is only supported on PyTorch 2.5 or later, please use per-tensor instead."
        )

        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self._sync_state()
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.min_val = torch.rand(
            0,
        )
        self.max_val = torch.rand(
            0,
        )

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "min_val":
                    self.min_val.resize_(val.shape)
                elif name == "max_val":
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(
                        f"Observer load_from_state_dict got unexpected name {name}"
                    )
            elif strict:
                missing_keys.append(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class DistributionProfileObserver(MinMaxObserver):
    def __init__(self, output_dir: str, max_sample_num: int, bins: int, **kwargs):
        super().__init__(**kwargs)
        self.cached_tensor = None
        self.output_dir = output_dir
        self.max_sample_num = max_sample_num
        self.bins = bins
        self.orig_shape = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        tensor = x_orig.detach()  # avoid keeping autograd tape
        if self.cached_tensor is None:
            self.cached_tensor = tensor.cpu()
            self.orig_shape = tuple(tensor.shape)
        else:
            self.cached_tensor = torch.cat([self.cached_tensor, tensor.cpu()])
        return super().forward(x_orig)

    @torch.jit.export
    def calculate_qparams(self):
        import os
        import warnings

        import matplotlib.pyplot as plt
        import seaborn as sns

        orig_shape = self.orig_shape
        assert orig_shape is not None, "must run observer before compute histogram"

        tensor = self.cached_tensor.view(-1)
        print(f"generating distribution for {self.prefix}, orig_shape: {orig_shape}")
        if tensor.numel() > self.max_sample_num:
            tensor = tensor[torch.randperm(tensor.numel())[: self.max_sample_num]]
            warnings.warn(f"exceed max_sample_num, random resample to {tensor.numel()}")

        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        # 将 tensor 转换为 NumPy 数组
        data = tensor.float().numpy()
        # 使用 Seaborn 绘制分布图
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=self.bins, kde=True, color="purple")
        plt.title(f"Tensor distribution for {self.prefix}")
        plt.xlabel("value")
        plt.ylabel("number")
        plt.legend(
            loc="best",  # 位置
            fontsize="large",  # 字体大小
            frameon=False,  # 是否显示边框
            shadow=True,  # 是否显示阴影
            title=f"Mean: {mean:.4g}\nStd: {std:.4g}\nMin: {min_val:.4g}\nMax: {max_val:.4g}",  # 图例标题
        )
        images_dir = os.path.join(self.output_dir, "_images")
        os.makedirs(images_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "distribution.md"), "a") as f:
            f.write(f"# {self.prefix}\n\n")
            f.write(f"orig_shape: {orig_shape}\n\n")
            f.write(
                f"Mean: {mean:.4g}\n\nStd: {std:.4g}\n\nMin: {min_val:.4g}\n\nMax: {max_val:.4g}\n\n"
            )
            f.write(f"![](_images/{self.prefix}.png)\n\n")
        image_file = os.path.join(images_dir, f"{self.prefix}.png")
        if os.path.exists(image_file):
            warnings.warn(f"image file {image_file} already exists, skipping")
        else:
            plt.savefig(image_file)

        return super().calculate_qparams()


class DynamicObserverPlaceholder(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    """

    def forward(self, x_orig):
        pass

    @torch.jit.export
    def calculate_qparams(self):
        pass
