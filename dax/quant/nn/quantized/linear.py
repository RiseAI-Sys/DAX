import warnings

import torch
from torch import Tensor, __version__, nn
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.version import cuda

from ...qscheme import per_channel_symmetric
from .functional import FP8_ALLOWED_DTYPES, quantize_to_fp8

AT_LEAST_TORCH_2_5 = __version__ >= (2, 5)

IS_TORCH_2_4 = __version__ < (2, 4, 9)

LT_TORCH_2_4 = __version__ < (2, 4)
if LT_TORCH_2_4:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError(
            "This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later."
        )
CUDA_VERSION = float(cuda) if cuda else 0
if CUDA_VERSION < 12.4:
    raise RuntimeError(
        f"This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later got torch version {__version__} and CUDA version {cuda}."
    )


class Linear(nn.Module):
    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        weight_dtype=None,
        input_dtype=None,
        out_dtype=None,
        row_wise_scale=False,
    ) -> None:
        if row_wise_scale:
            assert AT_LEAST_TORCH_2_5, (
                "per-channel weight scale is only supported on PyTorch 2.5 or later, please use per-tensor instead."
            )
        factory_kwargs = {"device": device, "dtype": out_dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.qweight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=weight_dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=(
                        torch.bfloat16 if row_wise_scale else out_dtype
                    ),  # row-wise scale only supports bfloat16 bias dtype
                )
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "weight_scale",
            torch.ones(
                ((1, out_features) if row_wise_scale else 1),
                device=device,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "input_scale",
            torch.tensor([1.0], device=device, dtype=torch.float32),
            persistent=True,
        )
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.input_dtype = input_dtype
        self.row_wise_scale = row_wise_scale

    @property
    def weight(self):
        return self.qweight.to(self.out_dtype) * self.weight_scale.to(self.out_dtype)

    # @torch.compiler.disable
    def forward(self, input: Tensor) -> Tensor:
        qinput = quantize_to_fp8(input, self.input_scale, self.input_dtype)
        prev_dims = qinput.shape[:-1]
        qinput = qinput.view(-1, self.in_features)
        if self.row_wise_scale:
            # for now, row_wise scale only supports both act and weight to be row_wise,
            # so we expand per-tensor scale of act for now.
            scale_a = self.input_scale.expand(qinput.shape[0], 1).contiguous()
            # row-wise scale only supports bfloat16 bias dtype
            out_dtype = torch.bfloat16
        else:
            scale_a = self.input_scale
            out_dtype = self.out_dtype
        out = torch._scaled_mm(
            qinput,
            self.qweight.T,
            scale_a=scale_a,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        if IS_TORCH_2_4:
            out = out[0]
        out = out.view(*prev_dims, self.out_features)
        if out.dtype != self.out_dtype:
            out = out.to(self.out_dtype)
        return out

    def extra_repr(self) -> str:
        # return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        s = "in_features={in_features}, out_features={out_features}"
        if self.bias is None:
            s += ", bias=False"
        s += ", weight_scale={weight_scale}, input_scale={input_scale}"
        return s.format(
            **self.__dict__,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
        )

    @classmethod
    @torch.no_grad
    def get_qlinear(
        cls,
        mod,
        activation_post_process,
        weight_post_process,
        reserve_float_weight,
    ):
        r"""Creates a qlinear object and returns it."""
        if (
            activation_post_process is None
            or activation_post_process.dtype == torch.float
        ):
            raise ValueError("Input must be quantized!")

        try:
            qlinear = cls(
                mod.in_features,
                mod.out_features,
                mod.bias is not None,
                mod.weight.device,
                weight_post_process.dtype,
                activation_post_process.dtype,
                mod.weight.dtype,
                weight_post_process.qscheme == per_channel_symmetric,
            )
        except NotImplementedError as e:
            warnings.warn(
                "Quantized convolution is not supported for this combination of arguments. "
                "Falling back to float module. "
                f"Message: {e}"
            )
            return mod

        device = mod.weight.device

        input_scale = activation_post_process.calculate_qparams()
        input_dtype = activation_post_process.dtype
        if input_scale is not None:
            qlinear.input_scale.copy_(input_scale)
        qlinear.input_dtype = input_dtype

        # if load from pre-quantized model, skip weight calibcation
        if input_scale is not None:
            weight_post_process(mod.weight)

        if weight_post_process.dtype in FP8_ALLOWED_DTYPES:
            weight_scale = weight_post_process.calculate_qparams()
            weight_dtype = weight_post_process.dtype
            if weight_scale is not None:
                if weight_scale.numel() > 1:
                    assert mod.weight.shape[0] == weight_scale.numel()
                    scale_shape = [mod.weight.shape[0]] + [1] * (mod.weight.ndim - 1)
                else:
                    scale_shape = weight_scale.shape
                qweight = quantize_to_fp8(
                    mod.weight,
                    scale=weight_scale.reshape(scale_shape),
                    dtype=weight_dtype,
                )
                qlinear.qweight = nn.Parameter(qweight)
                weight_scale = weight_scale.reshape(qlinear.weight_scale.shape)
                qlinear.weight_scale.copy_(weight_scale)
            qlinear.weight_dtype = weight_dtype

        elif weight_post_process.dtype == torch.qint8:
            raise NotImplementedError
            qweight = _quantize_weight(mod.weight.float(), weight_post_process)
            qweight, weight_scale = qweight.int_repr(), qweight.q_scale()

        else:
            raise ValueError("Unexpected dtype " + weight_post_process.dtype)

        if qlinear.bias is not None:
            qlinear.bias.copy_(mod.bias)

        # important to release device memory
        if not reserve_float_weight:
            del mod.weight
            del mod.bias

        return qlinear

    @classmethod
    def from_float(cls, mod, reserve_float_weight=False):
        # for QAT
        if hasattr(mod, "weight_fake_quant"):
            assert hasattr(mod, "activation_post_process"), (
                "Input QAT module must have observer attached"
            )
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        # for PTQ
        else:
            assert type(mod) == cls._FLOAT_MODULE, (
                " nnq."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
                + " but got:"
                + str(type(mod))
            )
            assert hasattr(mod, "qconfig"), (
                "Input float module must have qconfig defined."
            )
            activation_post_process = (
                None
                if not hasattr(mod, "activation_post_process")
                else mod.activation_post_process
            )

            weight_post_process = mod.qconfig.weight()
        return cls.get_qlinear(
            mod,
            activation_post_process,
            weight_post_process,
            reserve_float_weight,
        )
