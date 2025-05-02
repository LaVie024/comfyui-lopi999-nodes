import random
import folder_paths
from .utils import AnyType
import comfy.model_management
import torch
import numpy as np
import comfy.sd
import comfy.samplers as cs
import inspect
from comfy.samplers import SchedulerHandler
from comfy.samplers import KSAMPLER_NAMES
from comfy.k_diffusion import sampling as kdiff
from nodes import MAX_RESOLUTION

def zipf_linear_scheduler(model_sampling, steps: int, x_start=3.2, x_end=2.75):
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    device = model_sampling.sigmas.device

    x_curve = torch.linspace(x_start, x_end, steps,
                             dtype=torch.float32, device=device)
    ranks = torch.arange(1, steps + 1,
                         dtype=torch.float32, device=device)

    weights = 1.0 / (ranks ** x_curve)
    weights = weights / weights.sum()

    cum_weights = torch.cat([
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.cumsum(weights, dim=0)
    ])
    cum_weights = 1.0 - cum_weights / cum_weights[-1]
    sigmas = cum_weights.mul_(sigma_max - sigma_min).add_(sigma_min)
    sigmas[-1] = sigma_min

    return sigmas

def zeta_scheduler(model_sampling, steps: int, x_start: float = 0.5, x_end: float = 3.0):
    """
    Zeta‐based scheduler with a sliding exponent:
      exponent(t) = linspace(x_start → x_end) over steps.

    Returns steps+1 noise values, monotonically decreasing.
    """
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    dev = model_sampling.sigmas.device

    # 1) build a per-step exponent curve
    exponents = torch.linspace(x_start, x_end, steps, device=dev, dtype=torch.float32)

    # 2) compute raw weights ∝ k^(–exponent[t])
    ranks = torch.arange(1, steps + 1, device=dev, dtype=torch.float32)
    w = ranks.pow(-exponents)
    w = w / w.sum()

    # 3) build a decreasing cumulative decay
    cum   = torch.cat([torch.tensor([0.0], device=dev), torch.cumsum(w, dim=0)])
    decay = 1.0 - cum / cum[-1]   # starts @1, ends @0

    # 4) map into [σ_max → σ_min], enforce exact final min
    out = sigma_min + (sigma_max - sigma_min) * decay
    out[-1] = sigma_min

    return out

for name, fn in (
    ("zipf_linear", zipf_linear_scheduler),
    ("zeta",        zeta_scheduler),
):
    if name not in cs.SCHEDULER_HANDLERS:
        cs.SCHEDULER_HANDLERS[name] = SchedulerHandler(fn)

for name in ("zipf_linear", "zeta"):
    if name not in cs.SCHEDULER_NAMES:
        cs.SCHEDULER_NAMES.append(name)
    if name not in cs.KSampler.SCHEDULERS:
        cs.KSampler.SCHEDULERS.append(name)

def register_custom_samplers(module):
    """
    Registers sample_<foo> both in cs.KSampler and in the k‐diffusion namespace,
    so that Comfy’s ksampler() lookup will find it.
    """
    for attr, fn in inspect.getmembers(module, inspect.isfunction):
        if not attr.startswith("sample_"):
            continue
        name = attr[len("sample_"):]              # e.g. "euler_extsig_cfg_pp"
        short_fn = fn

        # 1) make it selectable in the UI
        if hasattr(cs, "KSAMPLER_NAMES") and name not in cs.KSAMPLER_NAMES:
            cs.KSAMPLER_NAMES.append(name)
        if hasattr(cs.KSampler, "SAMPLERS") and name not in cs.KSampler.SAMPLERS:
            cs.KSampler.SAMPLERS.append(name)

        # 2) patch it onto the KSampler class (in case other code calls it)
        setattr(cs.KSampler, attr, short_fn)

        # 3) patch it into comfy.k_diffusion.sampling so ksampler() can find it
        setattr(kdiff, attr, short_fn)

class RandomSDXLLatentSize:
    # Class-level resolution definitions
    landscape_res = [
        ("1024x960", 1.07), ("1088x960", 1.13), ("1088x896", 1.21),
        ("1152x896", 1.29), ("1152x832", 1.38), ("1216x832", 1.46),
        ("1280x768", 1.67), ("1344x768", 1.75), ("1344x704", 1.91),
        ("1408x704", 2.0), ("1472x704", 2.09), ("1536x640", 2.4),
        ("1600x640", 2.5), ("1664x576", 2.89), ("1728x576", 3.0)
    ]

    portrait_res = [
        ("576x1728", 0.33), ("576x1664", 0.35), ("640x1600", 0.4),
        ("640x1536", 0.42), ("704x1472", 0.48), ("704x1408", 0.5),
        ("704x1344", 0.52), ("768x1344", 0.57), ("768x1280", 0.6),
        ("832x1216", 0.68), ("832x1152", 0.72), ("896x1152", 0.78),
        ("896x1088", 0.82), ("960x1088", 0.88), ("960x1024", 0.94)
    ]

    square_res = [("1024x1024", 1.0)]

    # Create separate dropdown options
    landscape_options = [f"{res[0]} ({res[1]:.2f})" for res in landscape_res + square_res]
    portrait_options = [f"{res[0]} ({res[1]:.2f})" for res in portrait_res + square_res]

    # Default indices
    default_landscape_min = landscape_options.index("1024x1024 (1.00)")
    default_landscape_max = landscape_options.index("1728x576 (3.00)")
    default_portrait_min = portrait_options.index("576x1728 (0.33)")
    default_portrait_max = portrait_options.index("1024x1024 (1.00)")

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "RESOLUTION_TEXT", "SHOW_HELP")
    FUNCTION = "generate"
    CATEGORY = "lopi999/utils"

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "landscape": ("BOOLEAN", {"default": True}),
                "portrait": ("BOOLEAN", {"default": True}),
                "resolution_multiplier": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100, "step": 0.05}),
                "landscape_min_res": (cls.landscape_options, {"default": cls.landscape_options[cls.default_landscape_min]}),
                "landscape_max_res": (cls.landscape_options, {"default": cls.landscape_options[cls.default_landscape_max]}),
                "portrait_min_res": (cls.portrait_options, {"default": cls.portrait_options[cls.default_portrait_min]}),
                "portrait_max_res": (cls.portrait_options, {"default": cls.portrait_options[cls.default_portrait_max]}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
        }

    def parse_resolution(self, res_str):
        """Extract resolution and aspect ratio from dropdown string"""
        parts = res_str.split(" ")
        resolution = parts[0]
        ratio = float(parts[1][1:-1])  # Remove parentheses
        return resolution, ratio

    def generate(self, seed, landscape, portrait, resolution_multiplier,
                landscape_min_res, landscape_max_res,
                portrait_min_res, portrait_max_res, batch_size):
        # Set the random seed for reproducibility
        random.seed(seed)

        # Parse selected resolutions and get their indices
        landscape_min_res, landscape_min_ratio = self.parse_resolution(landscape_min_res)
        landscape_max_res, landscape_max_ratio = self.parse_resolution(landscape_max_res)
        portrait_min_res, portrait_min_ratio = self.parse_resolution(portrait_min_res)
        portrait_max_res, portrait_max_ratio = self.parse_resolution(portrait_max_res)

        # Get current indices for failsafe checks
        current_landscape_min_idx = self.landscape_options.index(f"{landscape_min_res} ({landscape_min_ratio:.2f})")
        current_landscape_max_idx = self.landscape_options.index(f"{landscape_max_res} ({landscape_max_ratio:.2f})")
        current_portrait_min_idx = self.portrait_options.index(f"{portrait_min_res} ({portrait_min_ratio:.2f})")
        current_portrait_max_idx = self.portrait_options.index(f"{portrait_max_res} ({portrait_max_ratio:.2f})")

        # Failsafe: Ensure max >= min for landscape
        if current_landscape_max_idx < current_landscape_min_idx:
            if current_landscape_min_idx < len(self.landscape_options) - 1:
                landscape_max_res, landscape_max_ratio = self.parse_resolution(
                    self.landscape_options[current_landscape_min_idx + 1])
            else:
                landscape_max_res, landscape_max_ratio = landscape_min_res, landscape_min_ratio

        # Failsafe: Ensure max >= min for portrait
        if current_portrait_max_idx < current_portrait_min_idx:
            if current_portrait_min_idx < len(self.portrait_options) - 1:
                portrait_max_res, portrait_max_ratio = self.parse_resolution(
                    self.portrait_options[current_portrait_min_idx + 1])
            else:
                portrait_max_res, portrait_max_ratio = portrait_min_res, portrait_min_ratio

        # Filter resolutions based on parameters
        valid_groups = []
        if landscape:
            valid_landscape = [res for res in self.landscape_res
                             if landscape_min_ratio <= res[1] <= landscape_max_ratio]
            if valid_landscape:
                valid_groups.append(("landscape", valid_landscape))

        if portrait:
            valid_portrait = [res for res in self.portrait_res
                            if portrait_min_ratio <= res[1] <= portrait_max_ratio]
            if valid_portrait:
                valid_groups.append(("portrait", valid_portrait))

        if landscape and portrait:
            # Check if square resolution fits within either range
            square_fits_landscape = (landscape_min_ratio <= 1.0 <= landscape_max_ratio)
            square_fits_portrait = (portrait_min_ratio <= 1.0 <= portrait_max_ratio)

            if square_fits_landscape or square_fits_portrait:
                valid_groups.append(("square", self.square_res))

        if not valid_groups:
            # Fallback to default resolution
            w, h = 1024, 1024
        else:
            # Randomly select an orientation group
            selected_group = random.choice(valid_groups)
            # Randomly select a resolution from the chosen group
            selected_res = random.choice(selected_group[1])
            w, h = map(int, selected_res[0].split('x'))

        # Apply resolution multiplier resolution if enabled
        if resolution_multiplier != 1:
            w = int(float(w) * resolution_multiplier)
            h = int(float(h) * resolution_multiplier)

        # Generate latent (still using 8x downscaling)
        latent_height = h // 8
        latent_width = w // 8
        latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=self.device)

        # Create help text
        help_text = "=== Random SDXL Latent Size Help ===\n"
        help_text += f"Landscape Range: {landscape_min_res} to {landscape_max_res}\n"
        help_text += f"Portrait Range: {portrait_min_res} to {portrait_max_res}\n"
        help_text += f"Selected Resolution: {w}x{h}"
        if resolution_multiplier != 1:
            help_text += f" ({resolution_multiplier}x scaled)"

        return (
            {"samples": latent},
            w,
            h,
            f"{w}x{h}" + (f" ({resolution_multiplier}x)" if resolution_multiplier != 1 else ""),
            help_text,
        )

class RandomNormalDistribution:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mean": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "std_dev": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1e9, "step": 0.01}),
                "enable_min_max": ("BOOLEAN", {"default": False}),
                "output_type": (["float", "integer"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),  # Changed max seed value
            },
            "optional": {
                "minimum": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "maximum": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("float", "integer", "show_help")
    FUNCTION = "generate_random"
    CATEGORY = "lopi999/utils"

    def generate_random(self, mean, std_dev, enable_min_max, output_type, seed, minimum=0.0, maximum=1.0):
        # Set random seeds (seed is already validated to be 32-bit unsigned)
        random.seed(seed)
        np.random.seed(seed % (2**32))  # Ensure 32-bit seed for numpy

        # Generate normally distributed random number
        while True:  # Ensure we get a valid number
            value = np.random.normal(mean, std_dev)
            if not np.isnan(value):  # Check for NaN
                break

        # Apply min/max constraints if enabled
        if enable_min_max:
            value = max(minimum, min(maximum, value))

        # Convert to integer if requested
        int_value = int(round(value))

        # Create help text
        help_text = (
            "=== Normal Distribution Random Number ===\n"
            f"Mean: {mean:.2f}\n"
            f"Standard Deviation: {std_dev:.2f}\n"
            f"Output Type: {output_type}\n"
            f"Min/Max {'Enabled' if enable_min_max else 'Disabled'}\n"
            f"Current Value: {value:.4f}\n"
            f"Integer Value: {int_value}\n"
            "Note: Values are clamped when min/max is enabled"
        )

        if output_type == "integer":
            return (float(int_value), int_value, help_text)
        else:
            return (value, int_value, help_text)

class AdvancedTextSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        # Create input types with concat at top (optional)
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

        # Add text and probability widgets to required
        for i in range(10):
            base_types["required"][f"text_{i}"] = ("STRING", {"default": "", "multiline": True})
            base_types["required"][f"prob_{i}"] = ("INT", {"default": 1, "min": 0, "visible": False})

        return base_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "lopi999/utils"

    def execute(self, mode, index, seed, concat_text="", **kwargs):
        # Get the selected text based on mode
        if mode == "random":
            random.seed(seed)
            # Build weighted list
            weighted_texts = []
            for i in range(10):
                text = kwargs.get(f"text_{i}", "")
                prob = kwargs.get(f"prob_{i}", 0)
                if prob > 0 and text:
                    weighted_texts.extend([text] * prob)

            selected = random.choice(weighted_texts) if weighted_texts else ""
        else:
            # Index mode
            selected = kwargs.get(f"text_{index}", "")

        # Apply concatenation (concat_text comes first)
        if concat_text and selected:
            output = f"{concat_text}, {selected}"
        elif concat_text:
            output = concat_text
        else:
            output = selected

        return (output,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Update visibility based on mode
        mode = kwargs.get("mode", "index")
        visibility = {
            "index": mode == "index",
            "seed": mode == "random",
        }

        # Update widget visibility
        for i in range(10):
            visibility[f"prob_{i}"] = mode == "random"

        return visibility

class ZipfSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "automatic": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "x_start": ("FLOAT", {"default": 3.2, "min": 1.0, "max": 5.0, "step": 0.01}),
                "x_end": ("FLOAT", {"default": 2.75, "min": 1.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "lopi999/utils"

    def get_sigmas(self, steps, denoise, automatic, x_start=3.2, x_end=2.75, model=None):
        model_sampling = model.get_model_object("model_sampling")
        sigma_max = float(model_sampling.sigma_max.cpu().item())
        sigma_min = float(model_sampling.sigma_min.cpu().item())
        sigma_min = max(sigma_min, 1e-5)  # avoid 0

        if denoise < 1.0 and denoise > 0.0:
            total_steps = int(steps / denoise)
            full_sigmas = zipf_linear_scheduler(
                model_sampling,
                total_steps,
                x_start=x_start,
                x_end=x_end
            )
            sigmas = full_sigmas[-(steps + 1):]
        else:
            sigmas = zipf_linear_scheduler(
                model_sampling,
                steps,
                x_start=x_start,
                x_end=x_end
            )

        return (sigmas,)

class ZetaSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "automatic": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "x_start": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 100.0, "step": 0.01}),
                "x_end": ("FLOAT", {"default": 3.00, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "lopi999/utils"

    def get_sigmas(self, model, automatic, steps: int,
                   denoise: float = 1.0,
                   x_start: float = 0.5,
                   x_end:   float = 3.0):

        model_sampling = model.get_model_object("model_sampling")

        if denoise < 1.00 and denoise > 0.00:
            steps = int(steps / denoise)

        if automatic:
            sigmas = zeta_scheduler(
                    model_sampling,
                    steps,
                    x_start = 0.5,
                    x_end = 3.0
                )
        else:
            sigmas = zeta_scheduler(
                    model_sampling,
                    steps,
                    x_start,
                    x_end
                )

        return (sigmas,)

class SDXLEmptyLatentSizePicker_v2:
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

    RETURN_TYPES = ("LATENT","INT","INT","STRING")
    RETURN_NAMES = ("LATENT","width","height","resolution_text")
    FUNCTION     = "execute"
    CATEGORY     = "lopi999/utils"

    def execute(self, resolution, batch_size, swap_dimensions, width_override=0, height_override=0, resolution_multiplier=1.0):
        width, height = resolution.split(" ")[0].split("x")
        width  = width_override  if width_override  > 0 else int(float(width)  * resolution_multiplier)
        height = height_override if height_override > 0 else int(float(height) * resolution_multiplier)

        if swap_dimensions:
            width, height = height, width

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples":latent}, width, height,f"{width}x{height}")

class Lopi999InputParameters:
    # Credits to giris and alexopus for this code, modified a bit
    RETURN_TYPES = (
        "INT",                                               # steps
        "FLOAT",                                             # cfg
        comfy.samplers.KSampler.SAMPLERS,                    # sampler
        cs.KSampler.SCHEDULERS,                              # scheduler (core)
        cs.KSampler.SCHEDULERS + ['align_your_steps','gits'] # scheduler+extra (extended)
    )
    RETURN_NAMES = (
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "scheduler+extra",
    )
    OUTPUT_TOOLTIPS = (
        "steps (INT)",
        "cfg (FLOAT)",
        "sampler (SAMPLERS)",
        "scheduler (SCHEDULERS)",
        "scheduler+extra (SCHEDULERS + extra)",
    )
    FUNCTION = "get_values"

    CATEGORY = "lopi999/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process."
                    }
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt."
                    }
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "The algorithm used when sampling."}
                ),
                "scheduler": (
                    cs.KSampler.SCHEDULERS,  # only core list here
                    {"tooltip": "The scheduler controls how noise is gradually removed."}
                ),
            }
        }

    def get_values(self, steps, cfg, sampler, scheduler):
        # We simply mirror the chosen scheduler into both outputs:
        return (
            steps,
            cfg,
            sampler,
            scheduler,  # core
            scheduler,  # extra
        )



class ModelParameters:
    ckpt_list = folder_paths.get_filename_list("checkpoints")
    vae_list  = folder_paths.get_filename_list("vae")

    RETURN_TYPES = (
        ckpt_list,                # first output: ckpt_name
        ckpt_list + ["None"],     # second output: ckpt_name_none
        vae_list,                 # third output: vae_name
        ["Baked VAE"] + vae_list, # fourth output: vae_name_baked
    )
    RETURN_NAMES = (
        "ckpt_name_sans_none",
        "ckpt_name",
        "vae_name_sans_baked",
        "vae_name",
    )
    FUNCTION = "get_names"
    CATEGORY = "lopi999/utils"

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

    def get_names(self, ckpt_name, vae_name):
        return (
            ckpt_name,
            ckpt_name,
            vae_name,
            vae_name,
        )
