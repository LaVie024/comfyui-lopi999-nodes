import importlib
import inspect
import re

lopi_utils = importlib.import_module(".lopi999_utils", package=__package__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def camel_to_title(name):
    return re.sub(
        r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])',
        ' ',
        name
    ).replace('_', ' ')

for name, cls in inspect.getmembers(lopi_utils, inspect.isclass):
    if name.startswith("node_"):
        node_key = name[5:]
        NODE_CLASS_MAPPINGS[node_key] = cls
        display_name = camel_to_title(node_key)
        NODE_DISPLAY_NAME_MAPPINGS[node_key] = display_name

from .lopi999_img import ImgBurn
from .lopi999_img import HueSaturationLightLevels
NODE_CLASS_MAPPINGS["BurnImage"] = ImgBurn
NODE_CLASS_MAPPINGS["HueSaturationLightLevels"] = HueSaturationLightLevels
NODE_DISPLAY_NAME_MAPPINGS["BurnImage"] = "Burn Image"
NODE_DISPLAY_NAME_MAPPINGS["HueSaturationLightLevels"] = "Hue Saturation Light Levels"

from . import sampler_list
lopi_utils.register_custom_samplers(sampler_list)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
