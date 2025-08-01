import random
import torch
import comfy.model_management
import numpy as np

class node_RandomBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "n": ("INT", {"default": 0, "min": -0x8000000000000000, "max": 0x7FFFFFFFFFFFFFFF})
                }}

    RETURN_TYPES = ("BOOLEAN", "INT",)
    RETURN_NAMES = ("BOOLEAN", "INT+n",)
    FUNCTION = "randBool"
    CATEGORY = "lopi999/random"

    def randBool(self, seed, n):
        random.seed(seed)
        boole = random.choice([True, False])
        result = bool(boole)
        return (result, (int(result)+n),)

class node_RandomSDXLLatentSize:
    RES_GROUPS = {
        "landscape": [
            ("1024x1024", 1.00),
            ("1024x960", 1.07), ("1088x960", 1.13), ("1088x896", 1.21),
            ("1152x896", 1.29), ("1152x832", 1.38), ("1216x832", 1.46),
            ("1280x768", 1.67), ("1344x768", 1.75), ("1344x704", 1.91),
            ("1408x704", 2.00), ("1472x704", 2.09), ("1536x640", 2.40),
            ("1600x640", 2.50), ("1664x576", 2.89), ("1728x576", 3.00),
        ],
        "portrait": [
            ("576x1728", 0.33), ("576x1664", 0.35), ("640x1600", 0.40),
            ("640x1536", 0.42), ("704x1472", 0.48), ("704x1408", 0.50),
            ("704x1344", 0.52), ("768x1344", 0.57), ("768x1280", 0.60),
            ("832x1216", 0.68), ("832x1152", 0.72), ("896x1152", 0.78),
            ("896x1088", 0.82), ("960x1088", 0.88), ("960x1024", 0.94),
        ],
        "square": [("1024x1024", 1.00)],
    }

    @staticmethod
    def _make_opts(lst):
        return [f"{r} ({a:.2f})" for r, a in lst]

    landscape_options = _make_opts.__func__(RES_GROUPS["landscape"] + RES_GROUPS["square"])
    portrait_options  = _make_opts.__func__(RES_GROUPS["portrait"] + RES_GROUPS["square"])

    def _def_idx(opts, txt):
        return opts.index(txt)

    default_landscape_min = _def_idx(landscape_options, "1024x1024 (1.00)")
    default_landscape_max = _def_idx(landscape_options, "1728x576 (3.00)")
    default_portrait_min  = _def_idx(portrait_options,  "576x1728 (0.33)")
    default_portrait_max  = _def_idx(portrait_options,  "1024x1024 (1.00)")

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "RESOLUTION_TEXT")
    FUNCTION      = "generate"
    CATEGORY      = "lopi999/random"

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        num = lambda d, mn, mx, st=1: ("INT", {"default": d, "min": mn, "max": mx, "step": st})
        flt = lambda d, mn, mx, st: ("FLOAT", {"default": d, "min": mn, "max": mx, "step": st})

        return {"required": {
            "seed"              : num(0, 0, 0xffffffffffffffff),
            "landscape"         : ("BOOLEAN", {"default": True}),
            "portrait"          : ("BOOLEAN", {"default": True}),
            "resolution_multiplier": flt(1.00, 0.01, 100, 0.05),
            "landscape_min_res" : (cls.landscape_options, {"default": cls.landscape_options[cls.default_landscape_min]}),
            "landscape_max_res" : (cls.landscape_options, {"default": cls.landscape_options[cls.default_landscape_max]}),
            "portrait_min_res"  : (cls.portrait_options,  {"default": cls.portrait_options[cls.default_portrait_min]}),
            "portrait_max_res"  : (cls.portrait_options,  {"default": cls.portrait_options[cls.default_portrait_max]}),
            "batch_size"        : num(1, 1, 4096),
        }}

    @staticmethod
    def _parse(opt):
        res, ratio = opt.split(" ")
        return res, float(ratio[1:-1])

    def generate(self, seed, landscape, portrait, resolution_multiplier,
                 landscape_min_res, landscape_max_res,
                 portrait_min_res,  portrait_max_res,  batch_size):

        random.seed(seed)

        bounds = {
            "landscape": tuple(self._parse(v) for v in (landscape_min_res, landscape_max_res)),
            "portrait" : tuple(self._parse(v) for v in (portrait_min_res,  portrait_max_res)),
        }

        for key, (lo, hi) in bounds.items():
            opts = getattr(self, f"{key}_options")
            lo_idx = opts.index(f"{lo[0]} ({lo[1]:.2f})")
            hi_idx = opts.index(f"{hi[0]} ({hi[1]:.2f})")
            if hi_idx < lo_idx:  # swap to nearest valid
                hi_idx = lo_idx
                hi = self._parse(opts[hi_idx])
            bounds[key] = (lo, hi)

        candidates = []
        def _in(ratio, lohi): return lohi[0][1] <= ratio <= lohi[1][1]

        if landscape:
            lo, hi = bounds["landscape"]
            candidates.append(("landscape",
                               [r for r in self.RES_GROUPS["landscape"] if _in(r[1], bounds["landscape"])]))

        if portrait:
            candidates.append(("portrait",
                               [r for r in self.RES_GROUPS["portrait"] if _in(r[1], bounds["portrait"])]))

        if landscape and portrait and any(_in(1.0, b) for b in bounds.values()):
            candidates.append(("square", self.RES_GROUPS["square"]))

        if not candidates:
            res = "1024x1024"
        else:
            non_empty = [g for _, g in candidates if g]
            if not non_empty:
                res = "1024x1024"
            else:
                res = random.choice(random.choice(non_empty))[0]

        w, h = map(int, res.split("x"))
        if resolution_multiplier != 1:
            w = int(w * resolution_multiplier)
            h = int(h * resolution_multiplier)

        latent = torch.zeros([batch_size, 4, h // 8, w // 8], device=self.device)

        return (
            {"samples": latent}, w, h,
            f"{w}x{h}" + (f" ({resolution_multiplier}×)" if resolution_multiplier != 1 else ""),
        )

class node_RandomNormalDistribution:
    @classmethod
    def INPUT_TYPES(cls):
        num = lambda d, mn, mx, st: ("FLOAT", {"default": d, "min": mn, "max": mx, "step": st})
        return {
            "required": {
                "mean"          : num(0.0, -1e9, 1e9, 0.01),
                "std_dev"       : num(1.0,  0.0, 1e9, 0.01),
                "enable_min_max": ("BOOLEAN", {"default": False}),
                "seed"          : ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "minimum"       : num(0.0, -1e9, 1e9, 0.01),
                "maximum"       : num(1.0, -1e9, 1e9, 0.01),
            },
        }

    RETURN_TYPES  = ("FLOAT", "INT", "STRING")
    RETURN_NAMES  = ("float", "integer", "show_help")
    FUNCTION      = "generate_random"
    CATEGORY      = "lopi999/random"

    def generate_random(self, mean, std_dev, enable_min_max,
                        seed, minimum=0.0, maximum=1.0):

        seed64 = seed & 0xffffffffffffffff
        random.seed(seed64)
        rng = np.random.default_rng(seed64)

        value = rng.normal(mean, std_dev)

        if enable_min_max:
            if minimum > maximum:
                minimum, maximum = maximum, minimum
            value = np.clip(value, minimum, maximum)

        int_value = int(round(value))

        help_txt = (
            "=== Normal Distribution Random Number ===\n"
            f"Mean: {mean}\n"
            f"Std Dev: {std_dev}\n"
            f"Min/Max {'ON' if enable_min_max else 'OFF'}\n"
            f"Value: {value}\n"
            f"Integer: {int_value}"
        )

        return (float(value), int_value, help_txt)
