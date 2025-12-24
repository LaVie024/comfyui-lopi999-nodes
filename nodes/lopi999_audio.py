import base64
import binascii
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from typing import Optional, Tuple

import numpy as np
import torch


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
        "audio/x-aifc": ".aif",
        "audio/aif": ".aif",
    }
    return mapping.get(mime, default_ext)


def _sniff_ext(data: bytes, default_ext: str) -> str:
    # Best-effort magic-byte detection to pick a correct suffix for decoders.
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
    compact = "".join(s.split())
    compact += "=" * (-len(compact) % 4)
    try:
        return base64.b64decode(compact, validate=strict)
    except (binascii.Error, ValueError):
        return base64.urlsafe_b64decode(compact)


def _to_torch_waveform(ch_first: np.ndarray) -> torch.Tensor:
    """
    Expect numpy array shaped [C, T] float32 in [-1, 1] (or close).
    Return torch Tensor [C, T] float32 contiguous.
    """
    if ch_first.ndim == 1:
        ch_first = ch_first[None, :]
    if ch_first.ndim != 2:
        raise ValueError(f"Decoded audio must be 1D/2D, got shape {ch_first.shape}")
    if ch_first.dtype != np.float32:
        ch_first = ch_first.astype(np.float32, copy=False)
    t = torch.from_numpy(ch_first)
    return t.contiguous()


def _load_audio_pyav(path: str) -> Tuple[torch.Tensor, int]:
    import av  # optional dependency

    container = av.open(path)
    astream = next((s for s in container.streams if s.type == "audio"), None)
    if astream is None:
        raise ValueError("No audio stream found")

    # Decode to float32 via resampler. Use stream's layout/rate when known.
    layout = astream.layout.name if astream.layout is not None else None
    rate = int(astream.rate) if astream.rate else None

    kwargs = {"format": "fltp"}
    if layout:
        kwargs["layout"] = layout
    if rate:
        kwargs["rate"] = rate
    resampler = av.audio.resampler.AudioResampler(**kwargs)

    chunks = []
    sample_rate = rate or 0

    for packet in container.demux(astream):
        for frame in packet.decode():
            frame = resampler.resample(frame)
            if frame is None:
                continue
            arr = frame.to_ndarray()  # typically [C, T] for planar formats
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1] and arr.shape[1] <= 8:
                # Sometimes returns [T, C]
                arr = arr.T
            if sample_rate == 0 and getattr(frame, "sample_rate", None):
                sample_rate = int(frame.sample_rate)
            chunks.append(arr)

    if not chunks:
        raise ValueError("Decoded audio is empty")

    audio_np = np.concatenate(chunks, axis=1)
    return _to_torch_waveform(audio_np), int(sample_rate) if sample_rate else 0


def _ffprobe_stream_info(path: str) -> Tuple[int, int]:
    """
    Return (channels, sample_rate). Prefer ffprobe JSON; fallback to parsing ffmpeg -i stderr.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate",
            "-of",
            "json",
            path,
        ]
        out = subprocess.check_output(cmd)
        data = json.loads(out.decode("utf-8", errors="replace"))
        streams = data.get("streams") or []
        if not streams:
            raise ValueError("ffprobe: no audio stream")
        ch = int(streams[0].get("channels", 1))
        sr = int(streams[0].get("sample_rate", 0))
        return ch, sr

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("Neither ffprobe nor ffmpeg found on PATH")

    # Parse stderr from `ffmpeg -i` (best-effort).
    p = subprocess.run([ffmpeg, "-v", "error", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = p.stderr.decode("utf-8", errors="replace")

    # sample rate
    m_sr = re.search(r"(\d+)\s*Hz", txt)
    sr = int(m_sr.group(1)) if m_sr else 0

    # channels
    ch = 0
    if re.search(r"\bmono\b", txt):
        ch = 1
    elif re.search(r"\bstereo\b", txt):
        ch = 2
    else:
        m_ch = re.search(r"(\d+)\s*channels", txt)
        ch = int(m_ch.group(1)) if m_ch else 0

    if ch <= 0:
        ch = 1
    return ch, sr


def _load_audio_ffmpeg(path: str) -> Tuple[torch.Tensor, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH")

    channels, sample_rate = _ffprobe_stream_info(path)
    if sample_rate <= 0:
        # Let ffmpeg pick; we'll still output the probe result if available (0 otherwise).
        sample_rate = 0

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
    ]
    if sample_rate > 0:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-"]  # stdout

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode('utf-8', errors='replace').strip()}")

    raw = p.stdout
    if not raw:
        raise ValueError("ffmpeg decoded audio is empty")

    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size % channels != 0:
        # Trim partial frame
        audio = audio[: audio.size - (audio.size % channels)]
    audio = audio.reshape(-1, channels).T  # [C, T]

    return _to_torch_waveform(audio), int(sample_rate) if sample_rate > 0 else int(_ffprobe_stream_info(path)[1] or 0)


def _load_audio_wav_stdlib(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if sw == 1:
        # unsigned 8-bit PCM
        x = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw} bytes")

    x = x.reshape(-1, ch).T  # [C, T]
    return _to_torch_waveform(x), int(sr)


def _load_audio_any(path: str) -> Tuple[torch.Tensor, int]:
    # Prefer PyAV when available (ComfyUI is moving toward PyAV in some workflows) [4].
    try:
        return _load_audio_pyav(path)
    except Exception:
        pass

    # Fallback to ffmpeg.
    try:
        return _load_audio_ffmpeg(path)
    except Exception:
        pass

    # Last resort: WAV-only via stdlib.
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return _load_audio_wav_stdlib(path)

    raise RuntimeError(
        "Unable to decode audio. Install PyAV (python package `av`) or ensure ffmpeg/ffprobe are on PATH."
    )


class node_LoadAudioBase64:
    """
    Decode Base64-encoded audio and output ComfyUI AUDIO dict.

    Outputs AUDIO as:
      {"waveform": torch.Tensor[B, C, T], "sample_rate": int}
    per ComfyUI datatypes documentation [1].
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
            return (None,)

        payload = base64_audio
        ext = hint_extension

        if parse_data_uri:
            mime, payload = _split_data_uri(base64_audio)
            if mime:
                ext = _mime_to_ext(mime, default_ext=ext)

        audio_bytes = _decode_base64(payload, strict=strict_base64)

        if auto_detect_extension:
            ext = _sniff_ext(audio_bytes, default_ext=ext)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            waveform_ct, sample_rate = _load_audio_any(tmp_path)

        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # ComfyUI expects waveform shaped [B, C, T] [1].
        audio = {"waveform": waveform_ct.unsqueeze(0), "sample_rate": int(sample_rate)}
        return (audio,)
