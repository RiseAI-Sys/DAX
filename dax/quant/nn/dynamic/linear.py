import torch
from loguru import logger
from torch import Tensor, __version__, nn
from torch.version import cuda

from ...qscheme import per_channel_symmetric, per_tensor_symmetric, per_token_symmetric
from .functional import (
    FP8_ALLOWED_DTYPES,
    dynamic_quantize_to_fp8,
    dynamic_quantize_to_int8,
    quantize_to_fp8,
    quantize_to_int8,
)

AT_LEAST_TORCH_2_5 = __version__ >= (2, 5)
AT_LEAST_TORCH_2_2 = __version__ >= (2, 2)
if AT_LEAST_TORCH_2_2:
    from torch._dynamo import is_compiling
    from torch._higher_order_ops.out_dtype import out_dtype


IS_TORCH_2_4 = __version__ < (2, 4, 9)

LT_TORCH_2_4 = __version__ < (2, 4)


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
        weight_qscheme=None,
        input_qscheme=None,
    ) -> None:
        compute_dtype = None
        if weight_dtype in FP8_ALLOWED_DTYPES and input_dtype in FP8_ALLOWED_DTYPES:
            if LT_TORCH_2_4:
                if not hasattr(torch, "_scaled_mm"):
                    raise RuntimeError(
                        "FP8 linear is not supported before PyTorch 2.4. "
                        "Please upgrade to PyTorch 2.4 with CUDA 12.4 or later. "
                        f"got torch version {__version__} and CUDA version {cuda}."
                    )
            CUDA_VERSION = float(cuda) if cuda else 0
            if CUDA_VERSION < 12.4:
                raise RuntimeError(
                    "FP8 linear is not supported before CUDA 12.4. "
                    "Please upgrade to PyTorch 2.4 with CUDA 12.4 or later. "
                    f"got torch version {__version__} and CUDA version {cuda}."
                )

            if (
                input_qscheme == per_token_symmetric
                and weight_qscheme == per_channel_symmetric
            ):
                assert AT_LEAST_TORCH_2_5, (
                    "per-channel weight and per-token act scale for fp8 is only supported on PyTorch 2.5 or later, please use per-tensor instead."
                )
            elif (
                input_qscheme == per_tensor_symmetric
                and weight_qscheme == per_tensor_symmetric
            ):
                pass
            else:
                raise ValueError(
                    f"Invalid qscheme combination! "
                    f"input_qscheme: {input_qscheme}, weight_qscheme: {weight_qscheme}"
                )
            compute_dtype = "FP8"
        elif weight_dtype == torch.int8 and input_dtype == torch.int8:
            # torch._int_mm doesn't exist before 2.2
            assert AT_LEAST_TORCH_2_2, (
                "INT8 linear is not supported before PyTorch 2.2. ",
                "Please upgrade to PyTorch 2.2 or later. "
                f"got torch version {__version__}",
            )
            compute_dtype = "INT8"
        else:
            raise ValueError(
                "Unsupported dtype combination! "
                f"weight_dtype: {weight_dtype}, input_dtype: {input_dtype}"
            )

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.qweight = nn.Parameter(
        #     torch.empty(
        #         (out_features, in_features), device=device, dtype=weight_dtype
        #     )
        # )
        self.register_buffer(
            "qweight",
            torch.empty((out_features, in_features), device=device, dtype=weight_dtype),
            persistent=True,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=(
                        torch.bfloat16
                        if compute_dtype == "FP8"
                        and weight_qscheme == per_channel_symmetric
                        else out_dtype
                    ),  # row-wise scale only supports bfloat16 bias dtype
                )
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "weight_scale",
            torch.ones(
                ((1, out_features) if weight_qscheme == per_channel_symmetric else 1),
                device=device,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.input_dtype = input_dtype
        self.compute_dtype = compute_dtype
        self.weight_qscheme = weight_qscheme
        self.input_qscheme = input_qscheme

    @property
    def weight(self):
        return self.qweight.to(self.out_dtype) * self.weight_scale.T.to(self.out_dtype)

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def forward(self, input: Tensor) -> Tensor:
        if self.compute_dtype == "FP8":
            qinput, input_scale = dynamic_quantize_to_fp8(
                input,
                qscheme=self.input_qscheme,
                target_dtype=self.input_dtype,
            )

            prev_dims = qinput.shape[:-1]
            qinput = qinput.view(-1, self.in_features)

            out = torch._scaled_mm(
                qinput,
                self.qweight.T,
                # row-wise scale for input is expect to be (num_row, 1)
                scale_a=(
                    input_scale.reshape(-1, 1)
                    if self.input_qscheme == per_token_symmetric
                    else input_scale
                ),
                scale_b=self.weight_scale,
                bias=self.bias,
                # row-wise scale only supports bfloat16 bias dtype
                out_dtype=(
                    torch.bfloat16
                    if self.input_qscheme == per_token_symmetric
                    else self.out_dtype
                ),
                use_fast_accum=True,
            )
            if IS_TORCH_2_4:
                out = out[0]
            out = out.view(*prev_dims, self.out_features)
            if out.dtype != self.out_dtype:
                out = out.to(self.out_dtype)

        elif self.compute_dtype == "INT8":
            prev_dims = input.shape[:-1]
            flattened_input = input.view(-1, self.in_features)
            M, K = flattened_input.shape
            K, N = self.qweight.T.shape

            m_is_strictly_greater_than_16 = M > 16
            k_is_nonzero_multiple_of_8 = (K % 8 == 0) and (K > 0)
            n_is_nonzero_multiple_of_8 = (N % 8 == 0) and (N > 0)
            bad_dimensions_for_cublas = not (
                m_is_strictly_greater_than_16
                and k_is_nonzero_multiple_of_8
                and n_is_nonzero_multiple_of_8
            )
            if bad_dimensions_for_cublas and not is_compiling():
                logger.warning(
                    "cublas int8 matmul is not supported for this combination of dimensions. "
                    "Falling back to float implementation. This could degrade performance. "
                    "Use torch.compiler with max-autotune to enable triton "
                    "int8 matmul generation to avoid this. "
                    "M: {}, K: {}, N: {}".format(M, K, N)
                )
                return torch.nn.functional.linear(input, self.weight, self.bias)

            qinput, input_scale = dynamic_quantize_to_int8(
                flattened_input,
                qscheme=self.input_qscheme,
                target_dtype=self.input_dtype,
            )

            input_scale = input_scale.reshape(-1, 1)

            assert input_scale.numel() == 1 or input_scale.size(0) == M
            assert self.weight_scale.numel() == 1 or self.weight_scale.size(1) == N
            input_scale = input_scale.expand((M, N))

            # output will be torch.int32
            # out = torch._int_mm(qinput, self.qweight.T)
            out = out_dtype(
                torch.ops.aten.mm.default, torch.int32, qinput, self.qweight.T
            )
            out = out * input_scale
            out = out * self.weight_scale

            out = out.view(*prev_dims, self.out_features)
            if out.dtype != self.out_dtype:
                out = out.to(self.out_dtype)
            if self.bias is not None:
                out = out + self.bias

        return out

    def extra_repr(self) -> str:
        # return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        s = "in_features={in_features}, out_features={out_features}"
        if self.bias is None:
            s += ", bias=False"
        s += ", weight_scale={weight_scale}"
        return s.format(**self.__dict__, weight_scale=self.weight_scale)

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
                weight_post_process.qscheme,
                activation_post_process.qscheme,
            )
        except NotImplementedError as e:
            logger.warning(
                "Quantized convolution is not supported for this combination of arguments. "
                "Falling back to float module. "
                f"Message: {e}"
            )
            return mod

        weight_post_process(mod.weight)

        if weight_post_process.dtype in FP8_ALLOWED_DTYPES:
            quantize_func = quantize_to_fp8
        elif weight_post_process.dtype == torch.int8:
            quantize_func = quantize_to_int8
        else:
            raise NotImplementedError(f"Unsupported dtype {weight_post_process.dtype}")

        weight_scale = weight_post_process.calculate_qparams()
        weight_dtype = weight_post_process.dtype
        if weight_scale is not None:
            if weight_post_process.qscheme == per_channel_symmetric:
                assert mod.weight.shape[0] == weight_scale.numel()
                scale_shape = [mod.weight.shape[0]] + [1] * (mod.weight.ndim - 1)
                weight_scale = weight_scale.reshape(scale_shape)
            qweight = quantize_func(
                mod.weight,
                scale=weight_scale,
                dtype=weight_dtype,
            )
            qlinear.qweight = qweight
            weight_scale = weight_scale.reshape(qlinear.weight_scale.shape)
            qlinear.weight_scale.copy_(weight_scale)

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

            activation_post_process = mod.qconfig.activation()
            weight_post_process = mod.qconfig.weight()
        return cls.get_qlinear(
            mod,
            activation_post_process,
            weight_post_process,
            reserve_float_weight,
        )
