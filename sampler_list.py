import math
import folder_paths
import comfy

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm

# ComfyUI core modules
import comfy.utils as utils
import comfy.model_patcher as model_patcher
import comfy.model_sampling as model_sampling
from comfy.k_diffusion import utils as kdiff_utils

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.special.expm1(beta_d * t ** 2 / 2 + beta_min * t))
    return append_zero(sigmas)


def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas



def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / kdiff_utils.append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

@torch.no_grad()
def sample_euler_extsig_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    """
    A multipass variant of Euler-Ancestral sampling.
    - For each i in [0, len(sigmas)-2], we check if sigma_i is in [pass_sigma_min, pass_sigma_max].
      If so, subdivide the step from sigma_i -> sigma_{i+1} into 'pass_steps' sub-steps.
      Otherwise, do a single standard step.
    - Each sub-step calls 'get_ancestral_step(...)' with the sub-interval's start & end sigmas,
      then applies the usual Euler-Ancestral update:
         x = x + d*dt + (noise * sigma_up)
    """
    if extra_args is None:
        extra_args = {}

    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Decide how many sub-steps to do
        if pass_sigma_min <= sigma_i <= pass_sigma_max:
            n_sub = pass_steps
        else:
            n_sub = 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            # Current sub-step range:
            sub_sigma_curr = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            # Denoise at the current sub-sigma
            denoised = model(x, sub_sigma_curr * s_in, **extra_args)

            if callback is not None:
                callback({
                    'x': x,
                    'i': i,
                    'sub_step': sub_step,
                    'sigma': sub_sigma_curr,
                    'denoised': denoised
                })

            # Compute the ancestral step parameters for this sub-interval
            sigma_down, sigma_up = get_ancestral_step(sub_sigma_curr, sub_sigma_next, eta=eta)
            if sigma_down == 0.0:
                # If we're stepping down to 0, we typically just take the final denoised
                x = denoised
            else:
                # Normal Euler-Ancestral logic
                d = to_d(x, sub_sigma_curr, denoised)
                dt = sigma_down - sub_sigma_curr
                x = x + d * dt
                if sigma_up != 0.0:
                    # Add noise for the "ancestral" part
                    x = x + noise_sampler(sub_sigma_curr, sub_sigma_next) * (s_noise * sigma_up)

    return x

@torch.no_grad()
def sample_euler_extsig(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.,
    s_tmin=0.,
    s_tmax=float('inf'),
    s_noise=1.0,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    """
    A multipass variant of Euler sampling.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Decide how many sub-steps to do
        if pass_sigma_min <= sigma_i <= pass_sigma_max:
            n_sub = pass_steps
        else:
            n_sub = 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            # Current sub-step range:
            sub_sigma_curr = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            if s_churn > 0:
                gamma = min(s_churn / (n_sub - 1), 2 ** 0.5 - 1) if s_tmin <= sub_sigma_curr < s_tmax else 0
                sigma_hat = sub_sigma_curr * (gamma + 1)
            else:
                gamma = 0
                sigma_hat = sub_sigma_curr

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat ** 2 - sigma_hat ** 2) ** 0.5

            # Denoise at the current sub-sigma
            denoised = model(x, sub_sigma_curr * s_in, **extra_args)

            if callback is not None:
                callback({
                    'x': x,
                    'i': i,
                    'sub_step': sub_step,
                    'sigma': sub_sigma_curr,
                    'sigma_hat': sigma_hat,
                    'denoised': denoised,
                })

            d = to_d(x, sigma_hat, denoised)
            dt = sub_sigma_next - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_extsig_cfg_pp(
    model, x, sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    s_churn=0.,
    s_tmin=0.,
    s_tmax=float('inf'),
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    """
    CFG++-enabled multipass Euler sampler.
    """
    if extra_args is None:
        extra_args = {}
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    # CFG++ wrapper
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        n_sub = pass_steps if pass_sigma_min <= sigma_i <= pass_sigma_max else 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            sub_sigma = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            if s_churn > 0:
                gamma = min(s_churn / (n_sub - 1), 2 ** 0.5 - 1) if s_tmin <= sub_sigma < s_tmax else 0
                sigma_hat = sub_sigma * (gamma + 1)
            else:
                gamma = 0
                sigma_hat = sub_sigma

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat ** 2 - sub_sigma ** 2).sqrt()

            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, temp[0])
            dt = sub_sigma_next - sigma_hat

            x = x + (denoised - temp[0]) + d * dt

            if callback is not None:
                callback({
                    'x': x, 'i': i, 'sub_step': sub_step,
                    'sigma': sub_sigma, 'sigma_hat': sigma_hat,
                    'denoised': denoised, 'uncond_denoised': temp[0]
                })

    return x

@torch.no_grad()
def sample_euler_extsig_ancestral_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pass_steps=2,
    pass_sigma_max=float("inf"),
    pass_sigma_min=12.0,
):
    """
    CFG++-enabled multipass ancestral Euler sampler.
    """
    if extra_args is None:
        extra_args = {}
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    # CFG++ wrapper
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Subdivision
        n_sub = pass_steps if pass_sigma_min <= sigma_i <= pass_sigma_max else 1
        sub_sigmas = torch.linspace(sigma_i, sigma_ip1, n_sub + 1, device=sigmas.device)

        for sub_step in range(n_sub):
            sub_sigma = sub_sigmas[sub_step]
            sub_sigma_next = sub_sigmas[sub_step + 1]

            # Compute ancestral steps
            sigma_down, sigma_up = get_ancestral_step(sub_sigma, sub_sigma_next, eta=eta)

            # CFG++ denoise
            denoised = model(x, sub_sigma * s_in, **extra_args)
            d = to_d(x, sub_sigma, temp[0])
            dt = sigma_down - sub_sigma

            # Main ancestral Euler update with CFG++
            x = x + (denoised - temp[0]) + d * dt

            # Noise injection
            if sub_sigma_next > 0:
                x = x + noise_sampler(sub_sigma, sub_sigma_next) * s_noise * sigma_up

            if callback is not None:
                callback({
                    'x': x, 'i': i, 'sub_step': sub_step,
                    'sigma': sub_sigma, 'sigma_hat': sub_sigma,
                    'denoised': denoised, 'uncond_denoised': temp[0]
                })

    return x
