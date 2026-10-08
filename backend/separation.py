"""Stem separation with Demucs (htdemucs_6s)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import numpy as np
import torch

log = logging.getLogger("stemscope.separation")

MODEL_NAME = "htdemucs_6s"
SAMPLE_RATE = 44100

_model = None
_lock = threading.Lock()


def devices() -> list[str]:
    """Devices to try in order. Apple-silicon GPU (MPS) is ~3x faster than CPU here."""
    forced = os.environ.get("STEMSCOPE_DEVICE")
    if forced:
        return [forced]
    return ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]


def load_model():
    global _model
    with _lock:
        if _model is None:
            from demucs.pretrained import get_model

            _model = get_model(MODEL_NAME)  # downloads to ~/.cache/torch on first use
            _model.eval()
        return _model


def separate(
    wav: np.ndarray, on_progress: Callable[[float], None] | None = None
) -> tuple[dict[str, np.ndarray], str]:
    """Split a (2, samples) float32 44.1 kHz mix into stems.

    Returns ({source_name: (2, samples) array}, device_used).
    """
    from demucs.apply import apply_model

    model = load_model()
    mix = torch.from_numpy(wav)
    # Same normalisation the demucs CLI applies before inference.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (mix - mean) / std

    length = mix.shape[-1]
    sub = model.models[0] if hasattr(model, "models") else model
    segment = int(float(sub.segment) * model.samplerate)
    progress = [0.0]

    def callback(info: dict) -> None:
        if info.get("state") != "end" or on_progress is None:
            return
        frac = min(1.0, (info["segment_offset"] + segment) / length)
        overall = (info["model_idx_in_bag"] + frac) / info["models"]
        if overall > progress[0]:
            progress[0] = overall
            on_progress(overall)

    candidates = devices()
    for device in candidates:
        try:
            with torch.no_grad():
                out = apply_model(
                    model, mix[None], device=device, shifts=1, split=True,
                    overlap=0.25, progress=False, callback=callback,
                )[0]
            break
        except Exception:
            if device == candidates[-1]:
                raise
            log.warning("Demucs failed on %s, retrying on next device", device, exc_info=True)
            progress[0] = 0.0
    out = out * std + mean
    stems = {name: out[i].numpy().astype(np.float32) for i, name in enumerate(model.sources)}
    return stems, device
