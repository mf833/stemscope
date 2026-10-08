"""Song-level measurements for the stats header: key, tempo + beats, loudness, stem presence.

All of these use the separated stems where that helps: key is estimated from
everything except drums, and beats from the drum stem, which is far cleaner
than analysing the full mix.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import resample

log = logging.getLogger("stemscope.stats")

STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
SR = 22050
# Conventional tonic spellings (D♭ major but C♯ minor, etc.).
MAJOR_TONICS = ["C", "D♭", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]
MINOR_TONICS = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "G♯", "A", "B♭", "B"]
LETTERS = "CDEFGAB"
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Krumhansl-Kessler key profiles (major / minor), tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]


def spell_scale(tonic: str, steps: list[int]) -> list[str]:
    """Spell a diatonic scale using each letter once (F♯ minor -> F♯ G♯ A B C♯ D E)."""
    start = LETTERS.index(tonic[0])
    tonic_pc = (NATURAL_PC[tonic[0]] + {"♯": 1, "♭": -1}.get(tonic[1:], 0)) % 12
    notes = []
    for i, step in enumerate(steps):
        letter = LETTERS[(start + i) % 7]
        diff = (tonic_pc + step - NATURAL_PC[letter] + 6) % 12 - 6
        notes.append(letter + {0: "", 1: "♯", 2: "𝄪", -1: "♭", -2: "𝄫"}[diff])
    return notes


def _mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr, SR)


def estimate_key(y: np.ndarray) -> dict:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=4096)
    energy = librosa.feature.rms(y=y, frame_length=8192, hop_length=4096)[0][: chroma.shape[1]]
    profile = (chroma[:, : len(energy)] * energy).sum(axis=1)
    scores = []
    for mode, template in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            scores.append((float(np.corrcoef(profile, np.roll(template, tonic))[0, 1]), tonic, mode))
    corr = np.array([s[0] for s in scores])
    probs = np.exp((corr - corr.max()) / 0.06)
    probs /= probs.sum()
    best = int(np.argmax(corr))
    _, tonic, mode = scores[best]
    if mode == "major":
        name, relative = MAJOR_TONICS[tonic], f"{MINOR_TONICS[(tonic - 3) % 12]} minor"
    else:
        name, relative = MINOR_TONICS[tonic], f"{MAJOR_TONICS[(tonic + 3) % 12]} major"
    return {
        "key": f"{name} {'maj' if mode == 'major' else 'min'}",
        "tonic": name,
        "mode": mode,
        "confidence": round(float(probs[best]), 3),
        "relative": relative,
        "scale": "Major" if mode == "major" else "Natural minor",
        "scale_notes": spell_scale(name, MAJOR_STEPS if mode == "major" else MINOR_STEPS),
    }


def estimate_tempo(drums: np.ndarray, low: np.ndarray, fallback: np.ndarray) -> dict:
    import librosa

    source = drums if np.sqrt(np.mean(drums**2)) > 0.01 else fallback
    hop = 512
    onset = librosa.onset.onset_strength(y=source, sr=SR, hop_length=hop)
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=hop, units="time")
    if len(beats) < 8:
        return {"bpm": None, "beats": [], "downbeat": 0, "stability": None}
    ibi = np.diff(beats)
    median = float(np.median(ibi))
    stability = float(np.mean(np.abs(ibi - median) <= 0.05 * median))
    # Downbeat guess: the beat phase (mod 4) where kick + bass onsets are strongest.
    low_onset = librosa.onset.onset_strength(y=low, sr=SR, hop_length=hop, fmax=200, n_mels=32)
    frames = np.clip(np.round(beats * SR / hop).astype(int), 0, len(low_onset) - 1)
    strength = [float(low_onset[frames[p::4]].mean()) for p in range(4)]
    return {
        "bpm": round(60.0 / median, 1),
        "beats": [round(float(b), 3) for b in beats],
        "downbeat": int(np.argmax(strength)),
        "stability": round(stability, 3),
    }


def measure_loudness(path: Path) -> dict:
    import pyloudnorm as pyln

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    meter = pyln.Meter(sr)  # ITU-R BS.1770-4
    integrated = float(meter.integrated_loudness(x))
    peak = float(20 * np.log10(np.abs(x).max() + 1e-9))
    # Loudness range (EBU R128 style): spread of 3 s short-term loudness, 10th-95th percentile.
    win, hop = 3 * sr, sr
    short = np.array([meter.integrated_loudness(x[i:i + win]) for i in range(0, max(1, len(x) - win), hop)])
    short = short[np.isfinite(short) & (short > -70) & (short > integrated - 20)]
    lra = float(np.percentile(short, 95) - np.percentile(short, 10)) if len(short) > 2 else 0.0
    return {"lufs": round(integrated, 1), "peak_db": round(peak, 1), "lra": round(lra, 1)}


def stem_presence(levels_path: Path) -> dict:
    """Share of the (non-silent) song where each stem is clearly audible.

    A stem counts as playing when it is above -50 dBFS and within 24 dB of the
    whole mix at that moment, which ignores the faint bleed separation leaves.
    """
    levels = np.load(levels_path)
    db = {n: levels[n].astype(np.float64) for n in STEMS}
    mix_db = 10 * np.log10(sum(10 ** (v / 10) for v in db.values()) + 1e-12)
    audible = mix_db > -50
    if not audible.any():
        return {n: 0.0 for n in STEMS}
    return {n: round(float(np.mean(((db[n] > -50) & (db[n] > mix_db - 24))[audible])), 3) for n in STEMS}


def compute(job_dir: Path) -> dict:
    stems = {n: _mono(job_dir / "stems" / f"{n}.flac") for n in STEMS}
    harmonic = sum(stems[n] for n in STEMS if n != "drums")
    return {
        **estimate_key(harmonic),
        **estimate_tempo(stems["drums"], stems["drums"] + stems["bass"], harmonic),
        **measure_loudness(job_dir / "mix.flac"),
        "presence": stem_presence(job_dir / "levels.npz"),
    }
