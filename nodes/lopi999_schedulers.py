import torch
import math
import comfy.samplers as cs
from comfy.samplers import SchedulerHandler

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

def zeta_scheduler(
    model_sampling,
    steps: int,
    *,
    x_start: float = 0.5,
    x_end:   float = 3.0,
    path_type: str = "linear",
    kappa: float = 1.0,
    gamma: float = 1.0,
    pivot: float | None = None,
    clip_frac: float | None = None,
    range_scale: float = 1.0
):
    if steps < 1:
        raise ValueError("steps must be ≥ 1")
    if range_scale <= 0:
        raise ValueError("range_scale must be positive")
    if gamma < 0:
        raise ValueError("gamma must be ≥ 0")

    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    dev       = model_sampling.sigmas.device
    r         = torch.arange(1, steps + 1, dtype=torch.float32, device=dev)

    if pivot is None:
        if path_type == "linear":
            alpha = torch.linspace(x_start, x_end, steps,
                                   device=dev, dtype=torch.float32)
        else:
            t = torch.linspace(0.0, 1.0, steps, device=dev, dtype=torch.float32)

            if path_type == "poly":
                t = t.pow(max(kappa, 1e-6))
            elif path_type == "cosine":
                t = 0.5 - 0.5 * torch.cos(t * math.pi)
            elif path_type == "exp":
                denom = torch.expm1(torch.tensor(max(kappa, 1e-6), device=dev))
                t = torch.expm1(kappa * t) / denom
            else:
                raise ValueError(f"unknown path_type: {path_type!r}")

            alpha = x_start + t * (x_end - x_start)

    else:
        if not (0.0 < pivot < 1.0):
            raise ValueError("pivot must lie strictly between 0 and 1")
        t = (r - 1) / (steps - 1)
        alpha = torch.where(
            t <= pivot,
            x_start + (t / pivot)       * (x_end - x_start),
            x_end   + (t - pivot) / (1 - pivot) * (x_end - x_start)
        )

    w = r.pow(-alpha)
    if gamma != 1.0:
        w = w.pow(gamma)
    w = w / w.sum()

    cdf   = torch.cat([torch.tensor([0.0], device=dev), torch.cumsum(w, dim=0)])
    decay = 1.0 - cdf / cdf[-1]

    sigmas = sigma_min + range_scale * (sigma_max - sigma_min) * decay
    if clip_frac is not None and clip_frac > 0:
        lo = sigma_min * (1.0 - clip_frac)
        hi = sigma_max * (1.0 + clip_frac)
        sigmas.clamp_(min=lo, max=hi)

    sigmas[-1] = sigma_min
    return sigmas

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

class node_ZipfSchedulerNode:
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
    CATEGORY = "lopi999/schedulers"

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

class node_ZetaSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "x_start": ("FLOAT", {"default": 2.75, "min": 0.01, "step": 0.01}),
                "x_end": ("FLOAT", {"default": 3.20, "min": 0.01, "step": 0.01}),
                "path_type": (["linear","poly","cosine","exp"],),
                "kappa": ("FLOAT", {"default": 1.00, "min": 0.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.00, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 0.99, "step": 0.01}),
                "clip_frac": ("FLOAT", {"default": 0.00, "min": 0.00, "step": 0.01}),
                "range_scale": ("FLOAT", {"default": 1.00, "min": 0.01, "step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "lopi999/schedulers"

    def get_sigmas(
        self,
        model,
        steps: int,
        denoise: float,
        x_start: float,
        x_end: float,
        path_type: str,
        kappa: float,
        gamma: float,
        pivot: float,
        clip_frac: float,
        range_scale: float
    ):
        model_sampling = model.get_model_object("model_sampling")
        sigma_max = float(model_sampling.sigma_max.cpu().item())
        sigma_min = float(model_sampling.sigma_min.cpu().item())

        if path_type == "poly":
            kappa  = max(kappa, 0.05)
        if gamma < 0.0:
            gamma  = 0.0
        range_scale = max(range_scale, 0.01)

        sigmas = zeta_scheduler(
            model_sampling=model_sampling,
            steps=steps,
            x_start=x_start,
            x_end=x_end,
            path_type=path_type,
            kappa=kappa,
            gamma=gamma,
            pivot=pivot if pivot > 0.0 else None,
            clip_frac=clip_frac if clip_frac > 0.0 else None,
            range_scale=range_scale,
        )

        if denoise < 1.0:
            keep = max(1, round((len(sigmas) - 1) * denoise)) + 1
            sigmas = sigmas[-keep:]

        return (sigmas,)
