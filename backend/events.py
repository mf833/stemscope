"""Turn PANNs probabilities + stem loudness into an instrument timeline and events.

Everything here is cheap numpy on cached arrays, so the report is rebuilt on
every request; that is what lets the UI's sensitivity slider update live.

Presence uses hysteresis: a family is "playing" over any stretch where its
confidence stays above `off`, provided the stretch peaks above `on`. Short gaps
are bridged and very short blips dropped. "Gets louder/quieter" comes from the
matching Demucs stem (PANNs confidence is not a loudness measure): within a
presence segment we compare the stem's level over the few seconds before and
after each moment and keep the biggest jumps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .instruments import FAMILIES, GENERIC, Family, short_name

SMOOTH_S = 0.75  # moving-average window over PANNs output
MIN_SEGMENT_S = 2.0  # ignore detections shorter than this
MAX_GAP_S = 2.5  # bridge dropouts shorter than this
LEVEL_WINDOW_S = 4.0  # compare loudness over this long before vs after
LEVEL_SPACING_S = 8.0  # minimum distance between two level-change events
AUDIBLE_DB = -45.0  # ignore level changes in a stem that's barely there
SHOW_FAMILY_PEAK = 0.08  # families below this everywhere get no lane
SHOW_CLASS_PEAK = 0.05

PLURAL = {"enters": "enter", "plays": "play", "comes": "come", "gets": "get", "drops": "drop"}


def thresholds(sensitivity: float) -> dict:
    """Sensitivity 0..1 -> detection thresholds.

    Instruments inside a dense mix typically score ~0.15-0.5 with PANNs, so the
    default (0.5) triggers at 20% confidence; the curve is exponential so the
    slider stays useful at both ends (50% at 0, 8% at 1).
    """
    s = float(np.clip(sensitivity, 0.0, 1.0))
    on = 0.5 * 0.16**s
    return {"on": on, "off": on * 0.5, "db": 9.0 - 6.0 * s}


def _say(fam: Family, phrase: str) -> str:
    """Conjugate the leading verb for plural family names ("Drums enter")."""
    verb, _, rest = phrase.partition(" ")
    if fam.plural:
        verb = PLURAL.get(verb, verb)
    return f"{verb} {rest}".strip()


def _smooth(a: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average along axis 0 (edges average over fewer samples)."""
    if width <= 1:
        return a
    pad = width // 2
    csum = np.concatenate([np.zeros((1, *a.shape[1:])), np.cumsum(a, axis=0)], axis=0)
    idx = np.arange(len(a))
    lo, hi = np.clip(idx - pad, 0, len(a)), np.clip(idx + pad + 1, 0, len(a))
    return (csum[hi] - csum[lo]) / (hi - lo).reshape((-1,) + (1,) * (a.ndim - 1))


def segments(conf: np.ndarray, on: float, off: float, hop: float) -> list[tuple[int, int]]:
    """[start, end) bin ranges where the family is considered present."""
    above = np.concatenate([[0], (conf >= off).astype(int), [0]])
    edges = np.flatnonzero(np.diff(above))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if conf[a:b].max() >= on]
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * hop < MAX_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged if (b - a) * hop >= MIN_SEGMENT_S]


def level_changes(db: np.ndarray, a: int, b: int, hop: float, threshold_db: float) -> list[tuple[int, float]]:
    """Step changes in loudness inside [a, b): [(bin, change_in_dB)]."""
    w = int(round(LEVEL_WINDOW_S / hop))
    if b - a < 2 * w + 1:
        return []
    power = 10 ** (db.astype(np.float64) / 10)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    idx = np.arange(a + w, b - w + 1)
    before = 10 * np.log10((csum[idx] - csum[idx - w]) / w + 1e-10)
    after = 10 * np.log10((csum[idx + w] - csum[idx]) / w + 1e-10)
    diff = after - before
    ok = (np.abs(diff) >= threshold_db) & (np.maximum(before, after) > AUDIBLE_DB)
    spacing = int(round(LEVEL_SPACING_S / hop))
    picked: list[tuple[int, float]] = []
    for j in np.argsort(-np.abs(diff)):
        if np.abs(diff[j]) < threshold_db:
            break
        if ok[j] and all(abs(idx[j] - p) >= spacing for p, _ in picked):
            picked.append((int(idx[j]), float(diff[j])))
    return sorted(picked)


def build_report(job_dir: Path, duration: float, sensitivity: float = 0.5) -> dict:
    data = np.load(job_dir / "analysis.npz")
    hop = float(data["hop"])
    label_index = {str(label): i for i, label in enumerate(data["labels"])}
    probs = _smooth(data["probs"].astype(np.float32), max(1, int(round(SMOOTH_S / hop))))
    levels_path = job_dir / "levels.npz"
    levels = dict(np.load(levels_path)) if levels_path.exists() else {}

    th = thresholds(sensitivity)
    end_bin = len(probs) - int(round(1.0 / hop))  # reaching the last second = "plays to the end"
    pct = lambda a: np.clip(np.round(a * 100), 0, 100).astype(int).tolist()  # noqa: E731

    lanes, events, summary = [], [], []
    for fam in FAMILIES:
        labels = [c for c in fam.classes if c in label_index]
        if not labels:
            continue
        member = probs[:, [label_index[c] for c in labels]]
        conf = member.max(axis=1)
        if conf.max() < SHOW_FAMILY_PEAK:
            continue
        segs = segments(conf, th["on"], th["off"], hop)
        lanes.append({
            "family": fam.name,
            "color": fam.color,
            "stem": fam.stem,
            "peak": round(float(conf.max()), 3),
            "values": pct(conf),
            "segments": [[a * hop, min(b * hop, duration)] for a, b in segs],
            "classes": [
                {"label": label, "name": short_name(label), "peak": round(float(member[:, k].max()), 3),
                 "values": pct(member[:, k])}
                for k, label in enumerate(labels) if member[:, k].max() >= SHOW_CLASS_PEAK
            ],
        })

        def emit(kind: str, t: float, verb_phrase: str, detail: str = "", **extra) -> None:
            events.append({"time": t, "family": fam.name, "color": fam.color, "kind": kind,
                           "text": f"{fam.name} {_say(fam, verb_phrase)}", "detail": detail, **extra})

        parts: list[dict] = []  # pieces of the one-line summary, e.g. "enters at" 0:12
        for si, (a, b) in enumerate(segs):
            seg_conf = conf[a:b]
            detail = f"{seg_conf.max():.0%} peak · {seg_conf.mean():.0%} avg confidence"
            specific = {lab: float(member[a:b, k].mean()) for k, lab in enumerate(labels) if lab not in GENERIC}
            if len(specific) > 1:
                kind = max(specific, key=specific.get)
                if specific[kind] >= 0.05:
                    detail = f"mostly {short_name(kind)} · {detail}"

            if si > 0:
                emit("enter", a * hop, "comes back in", detail, confidence=round(float(seg_conf.max()), 3))
                parts.append({"text": _say(fam, "comes back in at"), "time": a * hop})
            elif a == 0:
                emit("enter", 0.0, "plays from the start", detail, confidence=round(float(seg_conf.max()), 3))
                parts.append({"text": _say(fam, "plays from the start"), "time": None})
            else:
                emit("enter", a * hop, "enters", detail, confidence=round(float(seg_conf.max()), 3))
                parts.append({"text": _say(fam, "enters at"), "time": a * hop})

            if fam.stem in levels:
                for i, change in level_changes(levels[fam.stem], a, b, hop, th["db"]):
                    word = "louder" if change > 0 else "quieter"
                    emit(word, i * hop, f"gets {word}", f"{change:+.0f} dB in the {fam.stem} stem",
                         change_db=round(change, 1))
                    parts.append({"text": _say(fam, f"gets {word} at"), "time": i * hop})

            if b < end_bin:
                emit("exit", b * hop, "drops out")
                parts.append({"text": _say(fam, "drops out at"), "time": b * hop})
            elif len(parts) == 1 and a == 0:
                parts = [{"text": _say(fam, "plays throughout"), "time": None}]
            else:
                parts.append({"text": _say(fam, "plays to the end"), "time": None})

        if parts:
            summary.append({"family": fam.name, "color": fam.color, "first": segs[0][0] * hop,
                            "parts": parts})

    events.sort(key=lambda e: (e["time"], e["kind"] != "exit"))
    summary.sort(key=lambda s: s["first"])
    return {
        "model": "PANNs Cnn14_DecisionLevelMax · AudioSet",
        "hop": hop,
        "duration": duration,
        "sensitivity": sensitivity,
        "thresholds": {k: round(v, 3) for k, v in th.items()},
        "lanes": lanes,
        "events": events,
        "summary": summary,
    }
