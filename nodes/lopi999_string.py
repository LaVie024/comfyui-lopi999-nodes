import re
import random
import comfy.sd
import comfy.samplers as cs
from comfy.samplers import SchedulerHandler
from comfy.samplers import KSAMPLER_NAMES

class node_Concatenate_With_Prefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "delimiter": ("STRING", {"default": ", ", "multiline": False}),
            "prefix": ("STRING", {"default": "%n%: ", "multiline": False})
            },
            "optional": {
                "string_1": ("STRING", {"forceInput": True}),
                "string_2": ("STRING", {"forceInput": True}),
                "string_3": ("STRING", {"forceInput": True}),
                "string_4": ("STRING", {"forceInput": True}),
                "string_5": ("STRING", {"forceInput": True}),
                "string_6": ("STRING", {"forceInput": True}),
                "string_7": ("STRING", {"forceInput": True}),
                "string_8": ("STRING", {"forceInput": True}),
                }
            }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "lopi999/string"

    def concat(self, delimiter, prefix, **kwargs):
        all_strings = [
            value
            for key, value in kwargs.items()
            if key.startswith("string_")
        ]
        non_empty = [s for s in all_strings if s]

        result = ""
        for idx, s in enumerate(non_empty, start=1):
            numbered = prefix.replace("%n%", str(idx)) + s
            if idx == 1:
                result = numbered
            else:
                result += delimiter + numbered

        return (result,)

class node_AdvancedTextSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        base_types = {
            "required": {
                "mode": (["index", "random"], {"default": "index"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "visible": False}),
            },
            "optional": {
                "concat_text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

        for i in range(10):
            base_types["required"][f"text_{i}"] = ("STRING", {"default": "", "multiline": True})
            base_types["required"][f"prob_{i}"] = ("INT", {"default": 1, "min": 0, "visible": False})

        return base_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "lopi999/string"

    def execute(self, mode, index, seed, concat_text="", **kwargs):
        if mode == "random":
            random.seed(seed)
            weighted_texts = []
            for i in range(10):
                text = kwargs.get(f"text_{i}", "")
                prob = kwargs.get(f"prob_{i}", 0)
                if prob > 0 and text:
                    weighted_texts.extend([text] * prob)

            selected = random.choice(weighted_texts) if weighted_texts else ""
        else:
            selected = kwargs.get(f"text_{index}", "")

        if concat_text and selected:
            output = f"{concat_text}, {selected}"
        elif concat_text:
            output = concat_text
        else:
            output = selected

        return (output,)

class node_ParametersToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "prefix": ("STRING", {"multiline": False, "default": ", ", "tooltip": "Both concatenates and prefixes all items in the string."}),
                "include_seed": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "lopi999/string"

    def to_string(self, seed, steps, cfg, sampler, scheduler, denoise, prefix, include_seed):
        initialPrefix = re.sub(
            r'^(?:[,\s]+(?=[A-Za-z])|[,\s]+$)',
            '',
            prefix
        )
        output = (initialPrefix + ((f"Seed: {seed}" + prefix) if include_seed else "") + prefix.join((f"Steps: {steps}", f"CFG: {cfg}", f"Sampler: {sampler}", f"Scheduler: {scheduler}", f"Denoise: {denoise:.2f}")))

        return (output,)

class node_ListWildcard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_newline_as_separator": ("BOOLEAN", {"default": True}),
                "separator_if_not_newline": ("STRING", {"multiline": False, "default": " "})
                }
            }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "lopi999/string"

    def execute(self, text, seed, use_newline_as_separator, separator_if_not_newline):
        if use_newline_as_separator:
            wildcard = text.splitlines()
        else:
            wildcard = text.split(separator_if_not_newline)

        if not wildcard:
            return (""),

        random.seed(seed)
        i = random.randint(0, (len(wildcard)-1))
        return (wildcard[i]),
