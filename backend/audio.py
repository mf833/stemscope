"""Audio decoding, resampling and the small numeric summaries the UI needs."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to float32 of shape (channels, samples).

    libsndfile reads MP3/WAV/FLAC/OGG itself. For anything else (M4A, AAC, ...)
    fall back to macOS's built-in afconvert, or ffmpeg if it is installed.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), sr
    except Exception as first_error:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "decoded.wav"
            for cmd in _fallback_decoders(path, wav):
                if subprocess.run(cmd, capture_output=True).returncode == 0 and wav.exists():
                    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                    return np.ascontiguousarray(data.T), sr
        raise ValueError(f"Could not decode {path.name}: {first_error}") from first_error


def _fallback_decoders(src: Path, dst: Path) -> list[list[str]]:
    cmds = []
    if shutil.which("afconvert"):
        cmds.append(["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(dst)])
    if shutil.which("ffmpeg"):
        cmds.append(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", str(dst)])
    return cmds


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    return np.ascontiguousarray(x[:2])


def write_flac(path: Path, x: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0).T, sr, format="FLAC", subtype="PCM_16")


def peaks(x: np.ndarray, sr: int, per_second: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket peak amplitude and RMS (channels combined) for waveform drawing."""
    bucket = sr // per_second
    n = int(np.ceil(x.shape[-1] / bucket))
    padded = np.zeros((x.shape[0], n * bucket), dtype=np.float32)
    padded[:, : x.shape[-1]] = x
    frames = padded.reshape(x.shape[0], n, bucket)
    peak = np.abs(frames).max(axis=(0, 2))
    rms = np.sqrt((frames**2).mean(axis=2).mean(axis=0))
    return peak, rms


def loudness_db(x: np.ndarray, sr: int, hop_s: float, n_frames: int, win_s: float = 0.5) -> np.ndarray:
    """Short-term level in dBFS: mean power over `win_s` around each hop-spaced frame."""
    power = (x.astype(np.float64) ** 2).mean(axis=0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    hop = hop_s * sr
    half = int(win_s * sr / 2)
    centers = (np.arange(n_frames) * hop + hop / 2).astype(int)
    lo = np.clip(centers - half, 0, len(power))
    hi = np.clip(centers + half, 0, len(power))
    mean_power = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    return (10 * np.log10(mean_power + 1e-10)).astype(np.float32)


def extract_cover(source: Path, dest_dir: Path) -> Path | None:
    """Save embedded album art (ID3, MP4 or FLAC pictures) as dest_dir/cover.*, if present."""
    try:
        from mutagen import File

        f = File(str(source))
    except Exception:
        return None
    if f is None:
        return None
    data, mime, tags = None, "image/jpeg", f.tags
    if tags is not None and hasattr(tags, "getall") and tags.getall("APIC"):  # MP3 (ID3)
        pic = tags.getall("APIC")[0]
        data, mime = pic.data, pic.mime
    elif getattr(f, "pictures", None):  # FLAC
        data, mime = f.pictures[0].data, f.pictures[0].mime
    elif tags is not None and "covr" in tags:  # M4A
        data = bytes(tags["covr"][0])
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    if not data:
        return None
    path = dest_dir / ("cover.png" if "png" in mime else "cover.jpg")
    path.write_bytes(data)
    return path
