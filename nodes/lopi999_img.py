import math
import torch
import kornia as K

def _gauss(x, μ, σ=0.25):
    return torch.exp(-0.5 * ((x - μ) / σ) ** 2)

_2PI = 2.0 * math.pi
_EPS = 1e-8

def _rgb_to_hsl(img: torch.Tensor):
    r, g, b = img.unbind(1)
    c_max = torch.max(img, 1)[0]
    c_min = torch.min(img, 1)[0]
    delta  = c_max - c_min

    l = 0.5 * (c_max + c_min)

    s = torch.zeros_like(l)
    mask = delta > _EPS
    s[mask] = delta[mask] / (1.0 - torch.abs(2. * l[mask] - 1.0))

    h = torch.zeros_like(l)
    rc = (((g - b) / delta) % 6.0)
    gc = ((b - r) / delta) + 2.0
    bc = ((r - g) / delta) + 4.0
    h[r == c_max] = rc[r == c_max]
    h[g == c_max] = gc[g == c_max]
    h[b == c_max] = bc[b == c_max]
    h = (h / 6.) % 1.0

    return torch.stack((h, s, l), dim=1)

def _hsl_to_rgb(hsl: torch.Tensor):
    h, s, l = hsl.unbind(1)
    q = torch.where(l < 0.5, l * (1. + s), l + s - l * s)
    p = 2. * l - q

    def _f(t):
        t = (t % 1.0)
        cond1 = t < 1/6
        cond2 = (1/6 <= t) & (t < 1/2)
        cond3 = (1/2 <= t) & (t < 2/3)
        return torch.where(cond1, p + (q - p) * 6. * t,
               torch.where(cond2, q,
               torch.where(cond3, p + (q - p) * (2/3 - t) * 6., p)))
    r = _f(h + 1/3)
    g = _f(h)
    b = _f(h - 1/3)
    return torch.stack((r, g, b), dim=1)

def _apply_mask(base: torch.Tensor, new: torch.Tensor, mask: torch.Tensor | None):
    if mask is None:
        return new
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return new * mask + base * (1. - mask)

class node_BurnImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadows": ("INT", {"default": 0, "min": 0, "max": 99}),
                "midtones": ("INT", {"default": 0, "min": 0, "max": 99}),
                "highlights": ("INT", {"default": 0, "min": 0, "max": 99}),
                },
            "optional": {
                "mask": ("MASK",),
                }
            }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "burn_image"
    CATEGORY = "lopi999/image"

    def burn_image(self, image, shadows, midtones, highlights, mask=None):
        img = image.float()
        B, H, W, _ = img.shape
        dev = img.device

        # luminance (Rec.709)
        Y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        w_sh  = _gauss(Y, 0.25)
        w_mid = _gauss(Y, 0.50)
        w_hi  = _gauss(Y, 0.75)

        S = (shadows/100.0) * w_sh \
        + (midtones/100.0) * w_mid \
        + (highlights/100.0) * w_hi

        S = S.unsqueeze(-1)

        burned = img * (1.0 - S)

        if mask is None:
            return (burned,)

        m = mask.float().unsqueeze(-1)
        out = img * (1.0 - m) + burned * m
        return (out,)

class node_HueSaturationLightLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": ("INT", {"default": 0, "min": -180, "max": 180}),
                "saturation": ("INT", {"default": 0, "min": -100, "max": 100}),
                "light": ("INT", {"default": 0, "min": -100, "max": 100}),
                "light_type": (["value","intensity","luma"],),
                },
            "optional": {
                "mask": ("MASK",),
                }
            }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hsv_adjustment"
    CATEGORY = "lopi999/image"

    def hsv_adjustment(self, image, hue, saturation, light, light_type, mask=None):
        nhwc = False
        if image.dim() == 4 and image.shape[-1] == 3:
            nhwc = True
            image = image.permute(0, 3, 1, 2).contiguous()
            if mask is not None and mask.dim() == 4 and mask.shape[-1] == 1:
                mask = mask.permute(0, 3, 1, 2).contiguous()

        img = torch.clamp(image.float(), 0., 1.)

        hue_shift = math.radians(hue)
        sat_gain  = 1. + saturation / 100.0
        light_gain = 1. + light / 100.0

        if light_type == "value":
            hsv = K.color.rgb_to_hsv(img)
            h, s, v = hsv.unbind(1)
            h = (h + hue_shift) % _2PI
            s = torch.clamp(s * sat_gain, 0., 1.)
            v = torch.clamp(v * light_gain, 0., 1.)
            out = K.color.hsv_to_rgb(torch.stack((h, s, v), dim=1))

        elif light_type == "intensity":
            hsv = K.color.rgb_to_hsv(img)
            h, s, _ = hsv.unbind(1)
            i_old = img.mean(1, keepdim=True)
            i_new = torch.clamp(i_old * light_gain, 0., 1.)
            scale = i_new / (i_old + _EPS)
            out = torch.clamp(img * scale, 0., 1.)

            if hue or saturation:
                hsv = K.color.rgb_to_hsv(out)
                h = (hsv[:,0] + hue_shift) % _2PI
                s = torch.clamp(hsv[:,1] * sat_gain, 0., 1.)
                out = K.color.hsv_to_rgb(torch.stack((h, s, hsv[:,2]), dim=1))

        else:
            y_old = (0.299 * img[:,0] + 0.587 * img[:,1] + 0.114 * img[:,2]).unsqueeze(1)
            y_new = torch.clamp(y_old * light_gain, 0., 1.)
            scale = y_new / (y_old + _EPS)
            out = torch.clamp(img * scale, 0., 1.)
            if hue or saturation:
                hsv = K.color.rgb_to_hsv(out)
                h = (hsv[:,0] + hue_shift) % _2PI
                s = torch.clamp(hsv[:,1] * sat_gain, 0., 1.)
                out = K.color.hsv_to_rgb(torch.stack((h, s, hsv[:,2]), dim=1))

        out = torch.clamp(out, 0., 1.)
        out = _apply_mask(img, out, mask)
        if nhwc:
            out = out.permute(0, 2, 3, 1).contiguous()
        return (out,)

