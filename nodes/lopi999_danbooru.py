import io
import os
import random
from typing import List, Optional, Tuple

import requests
from PIL import Image
import numpy as np
import torch

DANBOORU_API = "https://danbooru.donmai.us/posts.json"
USER_AGENT = "ComfyUI-DanbooruRandomImageNode/1.2"
ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_CREDENTIALS_FILE = os.path.join(BASE_DIR, "API.txt")

def _read_credentials() -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(API_CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            line = f.read().strip()
        if "," not in line:
            return None, None
        u, k = line.split(",", 1)
        u, k = u.strip(), k.strip()
        return (u or None, k or None)
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

def _split_tags(multiline: str) -> List[str]:
    if not multiline:
        return []
    raw = multiline.replace(",", " ").replace("\r", " ").replace("\n", " ")
    toks = [t.strip() for t in raw.split(" ") if t.strip()]
    return [t.replace(" ", "_") for t in toks]

def _choose_positive_rating(allow_g: bool, allow_s: bool, allow_q: bool, allow_e: bool) -> Optional[str]:
    allowed = []
    if allow_g: allowed.append("g")
    if allow_s: allowed.append("s")
    if allow_q: allowed.append("q")
    if allow_e: allowed.append("e")
    # If all ratings are allowed, we don't need to constrain rating at all
    if len(allowed) == 4:
        return None
    # Guarantee at least one positive tag by choosing one rating from the allowed set
    if allowed:
        return f"rating:{random.choice(allowed)}"
    # Failsafe: default to general
    return "rating:g"

def _compose_tag_query(include_tags: List[str],
                       exclude_tags: List[str],
                       allow_g: bool, allow_s: bool, allow_q: bool, allow_e: bool) -> str:
    tags: List[str] = []
    # positive user tags
    tags.extend(include_tags)

    # positive rating (one) to avoid "only negatives" and keep tag count low
    pr = _choose_positive_rating(allow_g, allow_s, allow_q, allow_e)
    if pr:
        tags.append(pr)

    # user excludes (prefixed with -)
    for t in exclude_tags:
        tags.append(t if t.startswith("-") else f"-{t}")

    # NOTE: don't add order:random; we use random=true query parameter instead
    return " ".join(tags)

def _pick_valid_post(params: dict, tries: int = 8) -> Optional[dict]:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    for _ in range(tries):
        resp = s.get(DANBOORU_API, params=params, timeout=20)
        # Handle 422 with helpful hint
        if resp.status_code == 422:
            raise RuntimeError(
                f"Danbooru returned 422 (Unprocessable). Likely too many/only-negative tags. "
                f"Query tags='{params.get('tags','')}'"
            )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            continue
        post = items[0]
        ext = (post.get("file_ext") or "").lower()
        if ext in ALLOWED_EXTS and (post.get("file_url") or post.get("large_file_url")):
            return post
    return None

def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)
    img = Image.open(data)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img

def _pil_to_image_tensor(img: Image.Image) -> torch.Tensor:
    rgb = img.convert("RGB") if img.mode == "RGBA" else img
    arr = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def _pil_to_mask_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode == "RGBA":
        alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        return torch.from_numpy(alpha)[None, ...]
    h, w = img.height, img.width
    return torch.ones((1, h, w), dtype=torch.float32)

class node_DanbooruRandomImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "include_tags": ("STRING", {"default": "", "multiline": True}),
                "exclude_tags": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0}),
                "include_rating_g": ("BOOLEAN", {"default": True}),
                "include_rating_s": ("BOOLEAN", {"default": True}),
                "include_rating_q": ("BOOLEAN", {"default": False}),
                "include_rating_e": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "download_random"
    CATEGORY = "lopi999"
    OUTPUT_NODE = False

    def download_random(self,
                        include_tags: str,
                        exclude_tags: str,
                        seed: int,
                        include_rating_g: bool,
                        include_rating_s: bool,
                        include_rating_q: bool,
                        include_rating_e: bool):
        inc = _split_tags(include_tags)
        exc = _split_tags(exclude_tags)

        tags_query = _compose_tag_query(
            inc, exc,
            allow_g=include_rating_g,
            allow_s=include_rating_s,
            allow_q=include_rating_q,
            allow_e=include_rating_e
        )

        params = {
            "limit": 1,
            "tags": tags_query,
            "random": "true",  # use API-side randomness (no order:random metatag)
        }

        username, api_key = _read_credentials()
        if username and api_key:
            params["login"] = username
            params["api_key"] = api_key

        post = _pick_valid_post(params, tries=10)
        if not post:
            raise RuntimeError("No suitable Danbooru post found for the given filters.")

        file_url = post.get("file_url") or post.get("large_file_url")
        if not file_url:
            raise RuntimeError("Post did not contain a downloadable file URL.")

        img = _download_image(file_url)
        return (_pil_to_image_tensor(img), _pil_to_mask_tensor(img))
