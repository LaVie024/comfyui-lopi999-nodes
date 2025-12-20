import base64
import binascii
import os
import re
import tempfile
from typing import Optional, Tuple

import torchaudio


def _mime_to_ext(mime: str, default_ext: str) -> str:
    mime = (mime or "").lower().strip()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/flac": ".flac",
        "audio/x-flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/aiff": ".aiff",
        "audio/x-aiff": ".aiff",
    }
    return mapping.get(mime, default_ext)


def _sniff_ext(data: bytes, default_ext: str) -> str:
    # Best-effort magic-byte detection to pick a correct suffix for torchaudio.load()
    # (we still allow the user-provided hint).
    if len(data) >= 12:
        if data[0:4] == b"RIFF" and data[8:12] == b"WAVE":
            return ".wav"
        if data[0:4] == b"fLaC":
            return ".flac"
        if data[0:4] == b"OggS":
            return ".ogg"
        if data[0:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
            return ".aiff"
    # MP3: ID3 tag or frame sync (0xFFE...).
    if len(data) >= 3 and data[0:3] == b"ID3":
        return ".mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"

    return default_ext


def _split_data_uri(s: str) -> Tuple[Optional[str], str]:
    # data:<mime>;base64,<payload>
    m = re.match(r"^\s*data:(?P<mime>[-\w.+/]+);base64,(?P<payload>.*)\s*$", s, flags=re.DOTALL)
    if not m:
        return None, s
    return m.group("mime"), m.group("payload")


def _decode_base64(s: str, strict: bool) -> bytes:
    # Remove whitespace/newlines.
    compact = "".join(s.split())
    # Add missing padding.
    compact += "=" * (-len(compact) % 4)

    try:
        return base64.b64decode(compact, validate=strict)
    except (binascii.Error, ValueError):
        # Try URL-safe alphabet as fallback.
        return base64.urlsafe_b64decode(compact)


class node_LoadAudioBase64:
    """
    Decode Base64-encoded audio and output ComfyUI AUDIO dict.

    Supports the same extensions advertised by ComfyUI's LoadAudio node:
    .wav, .mp3, .ogg, .flac, .aiff, .aif
    """
    SUPPORTED_FORMATS = (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_audio": ("STRING", {"multiline": False, "default": ""}),
                "hint_extension": (list(cls.SUPPORTED_FORMATS), {"default": ".wav"}),
                "strict_base64": ("BOOLEAN", {"default": True}),
                "parse_data_uri": ("BOOLEAN", {"default": True}),
                "auto_detect_extension": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(
        self,
        base64_audio: str,
        hint_extension: str = ".wav",
        strict_base64: bool = True,
        parse_data_uri: bool = True,
        auto_detect_extension: bool = True,
    ):
        if not isinstance(base64_audio, str) or not base64_audio.strip():
            raise ValueError("base64_audio is empty.")

        payload = base64_audio
        ext = hint_extension

        if parse_data_uri:
            mime, payload = _split_data_uri(base64_audio)
            if mime:
                ext = _mime_to_ext(mime, default_ext=ext)

        audio_bytes = _decode_base64(payload, strict=strict_base64)

        if auto_detect_extension:
            ext = _sniff_ext(audio_bytes, default_ext=ext)

        # Write to a temporary file so torchaudio can infer format from extension,
        # which avoids needing to pass format for file-like objects.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            waveform, sample_rate = torchaudio.load(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # ComfyUI AUDIO expects waveform shaped [B, C, T].
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
        return (audio,)
