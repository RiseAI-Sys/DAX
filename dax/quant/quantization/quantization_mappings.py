import copy
from typing import Any, Callable, Dict, Set

from torch import nn

from ..nn import dynamic as nndq
from ..nn import quantized as nnsq

__all__ = [
    "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_QAT_MODULE_MAPPINGS",
]


# Default map for swapping float module to quantized ones
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Linear: nnsq.Linear,
}

# Default map for swapping float module to quantized ones
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Linear: nndq.Linear,
}


def get_default_qconfig_propagation_list() -> Set[Callable]:
    """Get the default list of module types that we'll attach qconfig
    attribute to in prepare
    """
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = set(
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys()
    )
    return copy.deepcopy(QCONFIG_PROPAGATE_MODULE_CLASS_LIST)


def get_default_static_quant_module_mappings() -> Dict[Callable, Any]:
    """Get module mapping for post training static quantization"""
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)


def get_default_dynamic_quant_module_mappings() -> Dict[Callable, Any]:
    """Get module mapping for post training dynamic quantization"""
    return copy.deepcopy(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS)
