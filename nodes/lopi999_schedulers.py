import torch
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

def zeta_scheduler(model_sampling, steps: int, x_start: float = 0.5, x_end: float = 3.0):
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    dev = model_sampling.sigmas.device

    exponents = torch.linspace(x_start, x_end, steps, device=dev, dtype=torch.float32)
    ranks = torch.arange(1, steps + 1, device=dev, dtype=torch.float32)
    w = ranks.pow(-exponents)
    w = w / w.sum()

    cum   = torch.cat([torch.tensor([0.0], device=dev), torch.cumsum(w, dim=0)])
    decay = 1.0 - cum / cum[-1]

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
                "automatic": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "x_start": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 100.0, "step": 0.01}),
                "x_end": ("FLOAT", {"default": 3.00, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "lopi999/schedulers"

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
