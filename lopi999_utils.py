import random
from .utils import AnyType
import comfy.model_management
import torch
import numpy as np
import comfy.samplers as cs
from comfy.samplers import SchedulerHandler

def zipf_linear_scheduler(model_sampling, steps: int, x_start = 3.2, x_end = 2.75):
    """
    Generates a sigma schedule using a Zipf-based weighting function where the exponent
    changes gradually from x_start to x_end over the course of the schedule.
    """

    sigma_min = model_sampling.sigma_min
    sigma_max = model_sampling.sigma_max

    # Build a smooth exponent curve from x_start to x_end
    x_curve = torch.linspace(x_start, x_end, steps, dtype=torch.float32)
    ranks = torch.arange(1, steps + 1, dtype=torch.float32)
    weights = 1.0 / (ranks ** x_curve)
    weights /= weights.sum()

    # Convert to decreasing cumulative distribution
    cum_weights = torch.cat([torch.tensor([0.0]), torch.cumsum(weights, dim=0)])
    cum_weights = 1.0 - cum_weights / cum_weights[-1]

    # Convert cumulative weights to sigma values
    sigmas = sigma_min + (sigma_max - sigma_min) * cum_weights
    sigmas[-1] = sigma_min  # enforce exact min

    return sigmas.to(torch.float32)

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

cs.SCHEDULER_HANDLERS["zipf_linear"] = SchedulerHandler(zipf_linear_scheduler)
cs.SCHEDULER_HANDLERS["zeta"]       = SchedulerHandler(zeta_scheduler)
cs.SCHEDULER_NAMES = list(cs.SCHEDULER_HANDLERS)
cs.KSampler.SCHEDULERS = cs.SCHEDULER_NAMES

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
        ("704x1408", 0.5), ("704x1344", 0.52), ("768x1344", 0.57),
        ("768x1280", 0.6), ("832x1216", 0.68), ("832x1152", 0.72),
        ("896x1152", 0.78), ("896x1088", 0.82), ("960x1088", 0.88),
        ("960x1024", 0.94)
    ]

    square_res = [("1024x1024", 1.0)]

    # Create separate dropdown options
    landscape_options = [f"{res[0]} ({res[1]:.2f})" for res in landscape_res + square_res]
    portrait_options = [f"{res[0]} ({res[1]:.2f})" for res in portrait_res + square_res]

    # Default indices
    default_landscape_min = landscape_options.index("1024x1024 (1.00)")
    default_landscape_max = landscape_options.index("1728x576 (3.00)")
    default_portrait_min = portrait_options.index("704x1408 (0.50)")
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
                "enable_1_5x_resolution": ("BOOLEAN", {"default": False}),
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

    def generate(self, seed, landscape, portrait, enable_1_5x_resolution,
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

        # Apply 1.5x resolution if enabled
        if enable_1_5x_resolution:
            w = int(w * 1.5)
            h = int(h * 1.5)

        # Generate latent (still using 8x downscaling)
        latent_height = h // 8
        latent_width = w // 8
        latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=self.device)

        # Create help text
        help_text = "=== Random SDXL Latent Size Help ===\n"
        help_text += f"Landscape Range: {landscape_min_res} to {landscape_max_res}\n"
        help_text += f"Portrait Range: {portrait_min_res} to {portrait_max_res}\n"
        help_text += f"Selected Resolution: {w}x{h}"
        if enable_1_5x_resolution:
            help_text += " (1.5x scaled)"

        return (
            {"samples": latent},
            w,
            h,
            f"{w}x{h}" + (" (1.5x)" if enable_1_5x_resolution else ""),
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

