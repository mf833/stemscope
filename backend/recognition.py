"""Instrument recognition with PANNs (Cnn14_DecisionLevelMax, trained on AudioSet).

The sound-event-detection variant of Cnn14 outputs a probability for each of
AudioSet's 527 classes for every 10 ms frame, which is what lets us say *when*
an instrument is playing rather than just *whether* it is.
"""
from __future__ import annotations

import csv
import http.client
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# panns_inference imports matplotlib.pyplot; keep it off the macOS GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")

log = logging.getLogger("stemscope.recognition")

SAMPLE_RATE = 32000
HOP = 320  # samples per model frame -> 100 frames per second
MODEL_NAME = "PANNs Cnn14_DecisionLevelMax (AudioSet, 527 classes)"

# panns_inference reads its label file from this folder at import time.
PANNS_DIR = Path.home() / "panns_data"
LABELS_PATH = PANNS_DIR / "class_labels_indices.csv"
CHECKPOINT_PATH = PANNS_DIR / "Cnn14_DecisionLevelMax.pth"
LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
CHECKPOINT_MIN_BYTES = 300_000_000

CHUNK_S = 30.0  # audio is processed in chunks to bound memory use
CONTEXT_S = 2.0  # extra audio on each side of a chunk, discarded after inference

_model = None
_device = None
_labels: list[str] | None = None
_lock = threading.Lock()


def _download(url: str, dest: Path, on_progress: Callable[[float], None] | None, parts: int = 8) -> None:
    """Download `url` to `dest`, using parallel byte ranges when the server allows it.

    Zenodo throttles each connection to ~100 KB/s, so 8 ranges make the
    one-time 330 MB checkpoint download several times faster.
    """
    tmp = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.part")  # unique per process; renamed when complete
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)

    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=60) as resp:
        final_url = resp.geturl()  # follow redirects once
        content_range = resp.headers.get("Content-Range", "")
        total = int(content_range.rpartition("/")[2]) if resp.status == 206 and "/" in content_range else 0

    lock = threading.Lock()
    done = [0]

    def advance(n: int) -> None:
        with lock:
            done[0] += n
            if on_progress and total:
                on_progress(min(1.0, done[0] / total))

    def fetch_range(start: int, end: int) -> None:
        pos, failures = start, 0
        while pos <= end:
            try:
                req = urllib.request.Request(final_url, headers={"Range": f"bytes={pos}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "r+b") as f:
                    if r.status != 206:
                        raise OSError(f"server ignored range request (HTTP {r.status})")
                    f.seek(pos)
                    while chunk := r.read(1 << 18):
                        f.write(chunk)
                        pos += len(chunk)
                        advance(len(chunk))
            except (OSError, http.client.HTTPException):
                failures += 1
                if failures > 5:
                    raise
                log.warning("Range %d-%d interrupted at %d, retrying", start, end, pos, exc_info=True)
                time.sleep(2 * failures)

    try:
        if total > 0:
            with open(tmp, "wb") as f:
                f.truncate(total)
            size = -(-total // parts)
            with ThreadPoolExecutor(max_workers=parts) as pool:
                futures = [pool.submit(fetch_range, s, min(s + size, total) - 1) for s in range(0, total, size)]
                for fut in futures:
                    fut.result()
        else:  # no range support: plain single stream
            with urllib.request.urlopen(final_url, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    advance(len(chunk))
        if total and tmp.stat().st_size != total:
            raise OSError(f"incomplete download of {dest.name}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_files(on_progress: Callable[[float], None] | None = None) -> None:
    """Fetch the label list and checkpoint (~330 MB) on first use."""
    if not LABELS_PATH.exists():
        _download(LABELS_URL, LABELS_PATH, None)
    if not CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.stat().st_size < CHECKPOINT_MIN_BYTES:
        _download(CHECKPOINT_URL, CHECKPOINT_PATH, on_progress)


def needs_download() -> bool:
    return not CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.stat().st_size < CHECKPOINT_MIN_BYTES


def labels() -> list[str]:
    global _labels
    if _labels is None:
        with open(LABELS_PATH, newline="") as f:
            rows = list(csv.reader(f))
        _labels = [row[2] for row in rows[1:]]
    return _labels


def _pick_device() -> str:
    forced = os.environ.get("STEMSCOPE_DEVICE")
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(on_download_progress: Callable[[float], None] | None = None):
    global _model, _device
    with _lock:
        if _model is None:
            ensure_files(on_download_progress)
            from panns_inference.models import Cnn14_DecisionLevelMax

            model = Cnn14_DecisionLevelMax(
                sample_rate=SAMPLE_RATE, window_size=1024, hop_size=HOP, mel_bins=64,
                fmin=50, fmax=14000, classes_num=len(labels()), interpolate_mode="nearest",
            )
            # Official checkpoint from the PANNs Zenodo record; it stores a plain dict.
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            _device = _pick_device()
            _model = model.to(_device)
            log.info("PANNs loaded on %s", _device)
        return _model


def _infer(model, audio: np.ndarray) -> np.ndarray:
    global _model, _device
    x = torch.from_numpy(audio[None].astype(np.float32))
    try:
        with torch.no_grad():
            out = model(x.to(_device), None)["framewise_output"][0]
    except RuntimeError:
        if _device == "cpu":
            raise
        log.warning("PANNs failed on %s, falling back to CPU", _device, exc_info=True)
        _device = "cpu"
        _model = model = model.to("cpu")
        with torch.no_grad():
            out = model(x, None)["framewise_output"][0]
    return out.float().cpu().numpy()


def detect(mono: np.ndarray, on_progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Run sound event detection on mono 32 kHz audio.

    Returns framewise class probabilities, shape (frames, 527), 100 frames/s.
    """
    model = load_model()
    n = len(mono)
    chunk = int(CHUNK_S * SAMPLE_RATE)
    ctx = int(CONTEXT_S * SAMPLE_RATE)
    outputs = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        a, b = max(0, start - ctx), min(n, end + ctx)
        seg = mono[a:b]
        if len(seg) < SAMPLE_RATE:  # the CNN needs at least ~1 s of input
            seg = np.pad(seg, (0, SAMPLE_RATE - len(seg)))
        frames = _infer(model, seg)
        f0 = (start - a) // HOP
        f1 = f0 + int(np.ceil((end - start) / HOP))
        outputs.append(frames[f0:f1])
        if on_progress:
            on_progress(end / n)
    return np.concatenate(outputs, axis=0)
