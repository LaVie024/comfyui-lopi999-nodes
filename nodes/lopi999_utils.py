import re
import torch
import folder_paths
import comfy.model_management
from nodes import MAX_RESOLUTION

class node_SDXLEmptyLatentSizePicker_v2:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "resolution": (["512x2048 (0.25)","512x1984 (0.26)","512x1920 (0.27)","512x1856 (0.28)","576x1792 (0.32)","576x1728 (0.33)","576x1664 (0.35)","640x1600 (0.4)","640x1536 (0.42)","704x1472 (0.48)","704x1408 (0.5)","704x1344 (0.52)","768x1344 (0.57)","768x1280 (0.6)","832x1216 (0.68)","832x1152 (0.72)","896x1152 (0.78)","896x1088 (0.82)","960x1088 (0.88)","960x1024 (0.94)","1024x1024 (1.0)","1024x960 (1.8)","1088x960 (1.14)","1088x896 (1.22)","1152x896 (1.30)","1152x832 (1.39)","1216x832 (1.47)","1280x768 (1.68)","1344x768 (1.76)","1408x704 (2.0)","1472x704 (2.10)","1536x640 (2.4)","1600x640 (2.5)","1664x576 (2.90)","1728x576 (3.0)","1792x576 (3.12)","1856x512 (3.63)","1920x512 (3.76)","1984x512 (3.89)","2048x512 (4.0)",], {"default": "1024x1024 (1.0)"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "resolution_multiplier": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100, "step": 0.05}),
            "swap_dimensions": ("BOOLEAN", {"default": False}),
            }}

    RETURN_TYPES = ("LATENT","INT","INT","STRING","INT")
    RETURN_NAMES = ("LATENT","width","height","resolution_text","batch_size")
    FUNCTION     = "execute"
    CATEGORY     = "lopi999/utils"

    def execute(self, resolution, batch_size, swap_dimensions, width_override=0, height_override=0, resolution_multiplier=1.0):
        width, height = resolution.split(" ")[0].split("x")
        width  = width_override  if width_override  > 0 else int(float(width)  * resolution_multiplier)
        height = height_override if height_override > 0 else int(float(height) * resolution_multiplier)

        if swap_dimensions:
            width, height = height, width

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples":latent}, width, height,f"{width}x{height}", batch_size)

class node_LuminaEmptyLatentPicker:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "resolution": (["768x1532 (0.5)","896x1792 (0.5)","896x1664 (0.52)","960x1664 (0.57)","960x1600 (0.6)","1024x1536 (0.68)","1024x1472 (0.72)","968x1332 (0.73)","1152x1472 (0.78)","1152x1344 (0.82)","1216x1344 (0.88)","1216x1280 (0.94)","1024x1024 (1.0)","1280x1280 (1.0)","1280x1216 (1.8)","1344x1216 (1.14)","1344x1152 (1.22)","1472x1152 (1.3)","1332x968 (1.38)","1472x1024 (1.39)","1536x1024 (1.47)","1600x960 (1.68)","1664x960 (1.76)","1792x896 (2.0)","1532x768 (2.0)"],{"default": "1280x1280 (1.0)"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "resolution_multiplier": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100, "step": 0.05}),
            "swap_dimensions": ("BOOLEAN", {"default": False}),
            }}

    RETURN_TYPES = ("LATENT","INT","INT","STRING","INT")
    RETURN_NAMES = ("LATENT","width","height","resolution_text","batch_size")
    FUNCTION     = "execute"
    CATEGORY     = "lopi999/utils"

    def execute(self, resolution, batch_size, swap_dimensions, width_override=0, height_override=0, resolution_multiplier=1.0):
        width, height = resolution.split(" ")[0].split("x")
        width  = width_override  if width_override  > 0 else int(float(width)  * resolution_multiplier)
        height = height_override if height_override > 0 else int(float(height) * resolution_multiplier)

        if swap_dimensions:
            width, height = height, width

        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)

        return ({"samples":latent}, width, height,f"{width}x{height}", batch_size)

class node_ModelParameters:
    ckpt_list = folder_paths.get_filename_list("checkpoints")
    vae_list  = folder_paths.get_filename_list("vae")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    cls.ckpt_list,
                    {
                        "default": cls.ckpt_list[0] if cls.ckpt_list else "",
                        "tooltip": "Which checkpoint to use"
                    }
                ),
                "vae_name": (
                    cls.vae_list,
                    {
                        "default": cls.vae_list[0] if cls.vae_list else "",
                        "tooltip": "Which VAE to use"
                    }
                ),
            }
        }

    RETURN_TYPES = (ckpt_list, ckpt_list + ["None"], vae_list, ["Baked VAE"] + vae_list,)
    RETURN_NAMES = ("ckpt_name_sans_none", "ckpt_name", "vae_name_sans_baked", "vae_name",)
    FUNCTION = "get_names"
    CATEGORY = "lopi999/utils"

    def get_names(self, ckpt_name, vae_name):
        return (
            ckpt_name,
            ckpt_name,
            vae_name,
            vae_name,
        )

class node_TokenCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"forceInput": True}),
                }
            }

    RETURN_TYPES = ("STRING","INT",)
    RETURN_NAMES = ("tkn_count_str","tkn_count_int",)
    FUNCTION = "count_tokens"
    CATEGORY = "lopi999/utils"

    def count_tokens(self, clip, text):
        # Based on the original code used in the CLIPTokenCounter node,
        # made by pamparamm in ComfyUI-ppm.
        lengths = []
        blocks = []
        special_tokens = set(clip.cond_stage_model.clip_l.special_tokens.values())
        vocab = clip.tokenizer.clip_l.inv_vocab
        prompts = text.split("BREAK")
        for prompt in prompts:
            if len(prompt) > 0:
                tokens_pure = clip.tokenize(prompt)
                tokens_concat = sum(tokens_pure["l"], [])
                block = [t for t in tokens_concat if t[0] not in special_tokens]
                blocks.append(block)

        if len(blocks) > 0:
            lengths = [str(len(b)) for b in blocks]
        else:
            lengths = ["0"]

        result = lengths[0]
        return (result, int(result),)
