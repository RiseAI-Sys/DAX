import torch
from torch.ao.quantization import QConfig

from ..qscheme import per_channel_symmetric, per_tensor_symmetric, per_token_symmetric
from .observer import (
    DistributionProfileObserver,
    DynamicObserverPlaceholder,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


def get_default_fp8_qconfig(
    activation_dtype=torch.float8_e5m2, weight_dtype=torch.float8_e4m3fn
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
            sync_state=True,
        ),
        weight=MinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_tensor_symmetric
        ),
    )

    return qconfig


def get_fp8_per_channel_weight_qconfig(
    activation_dtype=torch.float8_e5m2, weight_dtype=torch.float8_e4m3fn
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
            sync_state=True,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_channel_symmetric
        ),
    )

    return qconfig


def get_default_dynamic_fp8_qconfig(
    activation_dtype=torch.float8_e5m2, weight_dtype=torch.float8_e4m3fn
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DynamicObserverPlaceholder.with_args(
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
            is_dynamic=True,
        ),
        weight=MinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_tensor_symmetric
        ),
    )

    return qconfig


def get_dynamic_fp8_per_token_act_per_channel_weight_qconfig(
    activation_dtype=torch.float8_e5m2, weight_dtype=torch.float8_e4m3fn
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DynamicObserverPlaceholder.with_args(
            dtype=activation_dtype,
            qscheme=per_token_symmetric,
            is_dynamic=True,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_channel_symmetric
        ),
    )

    return qconfig


def get_default_dynamic_int8_qconfig(
    activation_dtype=torch.int8, weight_dtype=torch.int8
):
    """
    Returns the default int8 qconfig for the specified int8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DynamicObserverPlaceholder.with_args(
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
            is_dynamic=True,
        ),
        weight=MinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_tensor_symmetric
        ),
    )

    return qconfig


def get_dynamic_int8_per_channel_weight_qconfig(
    activation_dtype=torch.int8, weight_dtype=torch.int8
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DynamicObserverPlaceholder.with_args(
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
            is_dynamic=True,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_channel_symmetric
        ),
    )

    return qconfig


def get_dynamic_int8_per_token_act_per_channel_weight_qconfig(
    activation_dtype=torch.int8, weight_dtype=torch.int8
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DynamicObserverPlaceholder.with_args(
            dtype=activation_dtype,
            qscheme=per_token_symmetric,
            is_dynamic=True,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=per_channel_symmetric
        ),
    )

    return qconfig


def get_default_dist_profile_qconfig(
    output_dir=".",
    max_sample_num=1000000,
    bins=1000,
    activation_dtype=torch.float8_e5m2,
    weight_dtype=torch.float8_e4m3fn,
):
    """
    Returns the default fp8 qconfig for the specified fp8 dtype.

    Return:
        qconfig
    """

    qconfig = QConfig(
        activation=DistributionProfileObserver.with_args(
            output_dir=output_dir,
            max_sample_num=max_sample_num,
            bins=bins,
            dtype=activation_dtype,
            qscheme=per_tensor_symmetric,
        ),
        weight=DistributionProfileObserver.with_args(
            output_dir=output_dir,
            max_sample_num=max_sample_num,
            bins=bins,
            dtype=weight_dtype,
            qscheme=per_tensor_symmetric,
        ),
    )

    return qconfig
