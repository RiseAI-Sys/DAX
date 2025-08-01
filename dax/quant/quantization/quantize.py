import copy
import warnings

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.quantization
import torch.nn as nn
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.observer import (
    _is_activation_post_process,
)  # noqa: F401
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr
from torch.ao.quantization.quantization_mappings import no_observer_set
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
    has_no_children_ignoring_parametrizations,
)
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.quantization.quantize import _get_unique_devices_

from .quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qconfig_propagation_list,
    get_default_static_quant_module_mappings,
)


def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the output"""
    self.activation_post_process(input[0])
    return None


def _register_activation_post_process_hook(module: nn.Module, pre_hook=False):
    assert hasattr(module, "activation_post_process"), (
        "Expect activation_post_process attribute already attached to the module"
    )

    handle = module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)


def _propagate_qconfig_helper(
    module,
    qconfig_dict,
    qconfig_parent=None,
    prefix="",
    prepare_custom_config_dict=None,
):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict
        prepare_custom_config_dict: dictionary for custom handling of modules
                                    see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """

    module_qconfig = qconfig_dict.get(
        type_before_parametrizations(module), qconfig_parent
    )
    # debug
    # if prefix in qconfig_dict:
    #     print(f"'{prefix}' is set by `qconfig_dict`: {qconfig_dict[prefix]}")
    # debug
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)

    module_qconfig = getattr(module, "qconfig", module_qconfig)

    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)

    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(module_qconfig, module)
    module.qconfig = qconfig_with_device_check

    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        #  do no not propagate qconfig to child if child is non traceable
        if prepare_custom_config_dict is None or not (
            name in prepare_custom_config_dict.get("non_traceable_module_name", [])
            or type(child)
            in prepare_custom_config_dict.get("non_traceable_module_class", [])
        ):
            _propagate_qconfig_helper(
                child, qconfig_dict, qconfig_with_device_check, module_prefix
            )


def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        prepare_custom_config_dict: dictionary for custom handling of modules
            see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(
        module,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
    )


def _add_observer_(
    module,
    qconfig_propagation_list=None,
    non_leaf_module_list=None,
    device=None,
    custom_module_class_mapping=None,
):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize
        qconfig_propagation_list: a list of quantizable modules that will have observers added to them
            if they are leaf nodes
        device: parent device, if any
        non_leaf_module_list: list of non-leaf modules we want to add observer

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    # respect device affinity when adding observers
    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        activation = (
            qconfig.activation()
            if special_act_post_process is None
            else special_act_post_process()
        )
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        return hasattr(m, "qconfig") and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        """Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        # We don't insert observer/fake_quantize for DeQuantStub
        if needs_observation(m) and not isinstance(m, DeQuantStub):
            # observer and hook will be gone after we swap the module
            m.add_module(
                "activation_post_process",
                get_activation_post_process(
                    m.qconfig, device, special_act_post_process
                ),
            )
            # Register observer as the first entry in the hook list
            # All post forward hooks are preserved and will be executed after the observer before convert
            _register_activation_post_process_hook(m)

    for name, child in module.named_children():
        # TODO remove Dropout special after codebase stable
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        elif issubclass(
            type_before_parametrizations(child),
            (nnq.FloatFunctional, nnq.QFunctional),
        ):
            if needs_observation(child):
                assert hasattr(child, "activation_post_process"), (
                    f"functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`"
                )
                child.activation_post_process = get_activation_post_process(
                    child.qconfig, device
                )
        elif isinstance(child, _FusedModule):
            # activation_post_process are now added directly to nn.Sequential/_FusedModule
            if needs_observation(child):
                insert_activation_post_process(child)
        elif (
            non_leaf_module_list is not None
            and type_before_parametrizations(child) in non_leaf_module_list
        ):
            if needs_observation(child):
                insert_activation_post_process(child)
        # elif _has_special_act_post_process(child):
        #     special_act_post_process = _get_special_act_post_process(child)
        #     insert_activation_post_process(child, special_act_post_process)
        elif (
            needs_observation(child)
            and type_before_parametrizations(child) in custom_module_class_mapping
        ):
            observed_child = custom_module_class_mapping[
                type_before_parametrizations(child)
            ].from_float(child)
            setattr(module, name, observed_child)
            # TODO: These are the modules that cannot be observed
            #       Once there are more, we should move them to a separate list
            if (
                custom_module_class_mapping[type_before_parametrizations(child)]
                not in no_observer_set()
            ):
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(
                child,
                qconfig_propagation_list,
                non_leaf_module_list,
                device,
                custom_module_class_mapping,
            )

    # Insert observers only for leaf nodes
    if (
        has_no_children_ignoring_parametrizations(module)
        and not isinstance(module, torch.nn.Sequential)
        and type_before_parametrizations(module) in qconfig_propagation_list
    ):
        insert_activation_post_process(module)


def prepare(model, qconfig_dict=None, inplace=True, observer_non_leaf_module_list=None):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        `model`: input model to be modified in-place
        `qconfig_dict`: dictionary that maps from model name and prefix to qconfig
            for example:
                {
                    "transformer": {
                        "": get_fp8_per_channel_weight_qconfig(),
                        "context_embedder": None,
                        "proj_out": None,
                    },
                    "controlnet": {
                        "": get_fp8_per_channel_weight_qconfig(),
                        "context_embedder": None,
                    },
                }
        `inplace`: carry out model transformations in-place, the original module is mutated
        `observer_non_leaf_module_list`: list of non-leaf modules we want to add observer



    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare")

    if not inplace:
        model = copy.deepcopy(model)

    qconfig_propagation_list = get_default_qconfig_propagation_list()
    propagate_qconfig_(model, qconfig_dict=qconfig_dict)

    # sanity check common API misusage
    if not any(hasattr(m, "qconfig") and m.qconfig for m in model.modules()):
        warnings.warn(
            "None of the submodule got qconfig applied. Make sure you "
            "passed correct configuration through `qconfig_dict` or "
            "by assigning the `.qconfig` attribute directly on submodules"
        )

    _add_observer_(
        model,
        qconfig_propagation_list,
        observer_non_leaf_module_list,
    )
    return model


def _remove_activation_post_process(
    module, remove_activation_post_process=True, remove_observer_hook=True
):
    # TODO: maybe we should change activation_post_process to _activation_post_process
    # to prevent it from being used by user
    if (
        hasattr(module, "activation_post_process")
        and _is_activation_post_process(module.activation_post_process)
        and remove_activation_post_process
    ):
        delattr(module, "activation_post_process")

    # remove activation_post_process pre and post hooks
    def remove_hooks():
        hook_map = module._forward_pre_hooks
        observer_hook = _observer_forward_pre_hook
        handle_ids_to_remove = set()
        for handle_id, hook_fn in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

    if remove_observer_hook:
        remove_hooks()


# TODO: rename to something more general
def _remove_qconfig(
    module,
    remove_qconfig=True,
    remove_activation_post_process=True,
    remove_observer_hook=True,
):
    r"""Clean up the qconfig left in the module so that new qconfig can be
    propagated.

    Args:
        module: module to be cleaned up
    """
    for child in module.children():
        _remove_qconfig(
            child,
            remove_qconfig,
            remove_activation_post_process,
            remove_observer_hook,
        )

    if hasattr(module, "qconfig") and remove_qconfig:
        del module.qconfig

    _remove_activation_post_process(
        module, remove_activation_post_process, remove_observer_hook
    )


def convert(
    model,
    mapping=None,
    inplace=True,
    remove_qconfig=True,
):
    r"""Converts submodules in input model to a different module according to `mapping`
    by calling `from_float` method on the target module class. And remove qconfig at the
    end if remove_qconfig is set to True.

    Args:
        `model`: prepared and calibrated model
        `mapping`: a dictionary that maps from source module type to target
                   module type, can be overwritten to allow swapping user defined
                   Modules
        `inplace`: carry out model transformations in-place, the original module
                   is mutated


    """
    torch._C._log_api_usage_once("quantization_api.quantize.convert")

    if mapping is None:
        mapping = get_default_static_quant_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)

    _convert(model, mapping)
    if remove_qconfig:
        _remove_qconfig(model)
    return model


def quantize_dynamic(
    model,
    qconfig_dict=None,
    mapping=None,
    inplace=True,
    remove_qconfig=True,
):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig_dict: dictionary that maps from model name and prefix to qconfig
            for example:
                {
                    "transformer": {
                        "": get_fp8_per_channel_weight_qconfig(),
                        "context_embedder": None,
                        "proj_out": None,
                    },
                    "controlnet": {
                        "": get_fp8_per_channel_weight_qconfig(),
                        "context_embedder": None,
                    },
                }

        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced
        inplace: carry out model transformations in-place, the original module is mutated

    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_dynamic")

    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig_dict)
    _convert(model, mapping)
    if remove_qconfig:
        _remove_qconfig(model)
    return model


def _convert(module, mapping=None):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class

    Args:
        module: input module
        is_dynamic: whether to do dynamic quantization
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules

    """
    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not isinstance(mod, _FusedModule):
            _convert(mod, mapping)
        reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def swap_module(mod, mapping, reserve_float_weight=False):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    if hasattr(mod, "qconfig") and mod.qconfig is not None:
        if type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            new_mod = qmod.from_float(mod, reserve_float_weight)

            # Preserve module's pre forward hooks except _observer_forward_pre_hook
            # They'll be called before input is quantized
            for pre_hook_fn in mod._forward_pre_hooks.values():
                if pre_hook_fn is not _observer_forward_pre_hook:
                    new_mod.register_forward_pre_hook(pre_hook_fn)
            # Preserve module's post forward hooks.
            # After convert they'll work with dequantized output
            for hook_fn in mod._forward_hooks.values():
                new_mod.register_forward_hook(hook_fn)

            # respect device affinity when swapping modules
            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1, (
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod
