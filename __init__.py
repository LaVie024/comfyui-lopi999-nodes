from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from types import ModuleType
from typing import Dict, Type

import comfy.samplers as cs
from comfy.k_diffusion import sampling as kdiff

def camel_to_title(name: str) -> str:
    return re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name
    ).replace("_", " ")


def register_custom_samplers(module: ModuleType) -> None:
    for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
        if not fn_name.startswith("sample_"):
            continue

        short = fn_name[len("sample_") :]

        if hasattr(cs, "KSAMPLER_NAMES") and short not in cs.KSAMPLER_NAMES:
            cs.KSAMPLER_NAMES.append(short)
        if hasattr(cs.KSampler, "SAMPLERS") and short not in cs.KSampler.SAMPLERS:
            cs.KSampler.SAMPLERS.append(short)

        setattr(cs.KSampler, fn_name, fn)
        setattr(kdiff, fn_name, fn)

sampler_mod = importlib.import_module(".sampler_list", package=__package__)
register_custom_samplers(sampler_mod)

NODE_CLASS_MAPPINGS: Dict[str, Type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

nodes_pkg = importlib.import_module(".nodes", package=__package__)

for mod_info in pkgutil.iter_modules(nodes_pkg.__path__):
    full_name = f"{nodes_pkg.__name__}.{mod_info.name}"
    module = importlib.import_module(full_name)

    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if not cls_name.startswith("node_"):
            continue
        if cls.__module__ != module.__name__:
            continue

        key = cls_name[5:]
        NODE_CLASS_MAPPINGS[key]        = cls
        NODE_DISPLAY_NAME_MAPPINGS[key] = camel_to_title(key)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
