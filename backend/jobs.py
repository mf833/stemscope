"""Job store and the background processing pipeline.

Each upload becomes a job folder under data/jobs/<id>/ where <id> is a hash of
the file contents, so uploading the same song twice reuses earlier results:

    source.mp3        the original upload
    meta.json         job state (what the UI polls)
    mix.flac          decoded original, 44.1 kHz stereo (sample-aligned with stems)
    stems/*.flac      Demucs output
    peaks.json        waveform overview for every track
    levels.npz        per-stem loudness curves (used for "gets louder" events)
    analysis.npz      PANNs class probabilities over time
    stats.json        key, tempo + beats, loudness, stem presence
    sections.json     user-defined song sections
    cover.jpg|png     embedded album art, when the file has any
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, events, recognition, separation
from . import stats as song_stats

log = logging.getLogger("stemscope.jobs")

STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_LABELS = {
    "mix": "Original mix", "vocals": "Vocals", "drums": "Drums", "bass": "Bass",
    "guitar": "Guitar", "piano": "Piano", "other": "Other",
}
PEAKS_PER_SECOND = 100
HOP_S = 0.25  # time resolution of loudness curves and instrument timelines
MAX_DURATION_S = 15 * 60  # 7 decoded tracks of a longer file strain browser memory

# Share of the overall progress bar given to each stage.
P_DECODE, P_SEPARATE, P_WRITE, P_RECOGNIZE = 0.03, 0.72, 0.77, 0.90


@dataclass
class Job:
    id: str
    filename: str
    created: float
    status: str = "queued"  # queued | processing | done | error
    stage: str = "Queued"
    progress: float = 0.0
    error: str | None = None
    duration: float | None = None
    device: str | None = None
    stems_ready: bool = False
    analysis_ready: bool = False
    analysis_error: str | None = None
    stats_ready: bool = False
    stats_error: str | None = None
    timings: dict = field(default_factory=dict)
    # Library metadata
    favorite: bool = False
    has_cover: bool = False
    key: str | None = None
    bpm: float | None = None
    presence: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d["format"] = Path(self.filename).suffix.lstrip(".").upper() or "AUDIO"
        d["tracks"] = (
            [{"name": n, "label": STEM_LABELS[n], "url": f"/api/jobs/{self.id}/audio/{n}.flac"}
             for n in ["mix", *STEMS]]
            if self.stems_ready else []
        )
        return d


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        # One job at a time: separation already uses every core / the GPU.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._load_existing()
        # Songs processed before stats existed (or whose measuring failed) get them in the background.
        for job in self.list():
            if job.status == "done" and job.stems_ready and not job.stats_ready:
                job.stats_error = None
                self.executor.submit(self._backfill, job)

    # ---------------------------------------------------------------- storage
    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _load_existing(self) -> None:
        for meta in self.root.glob("*/meta.json"):
            try:
                job = Job(**json.loads(meta.read_text()))
            except Exception:
                log.warning("Skipping unreadable %s", meta)
                continue
            if job.status in ("queued", "processing"):
                job.status, job.stage = "error", "Interrupted"
                job.error = "Processing was interrupted when the server stopped. Click Retry."
            self.jobs[job.id] = job

    def _save(self, job: Job) -> None:
        path = self.dir(job.id) / "meta.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), indent=2))
        tmp.replace(path)

    def update(self, job: Job, save: bool = True, **changes) -> None:
        with self.lock:
            for k, v in changes.items():
                setattr(job, k, v)
            if save:
                self._save(job)

    # ------------------------------------------------------------------- API
    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def add_upload(self, job_id: str, tmp_file: Path, filename: str) -> Job:
        existing = self.jobs.get(job_id)
        if existing:
            tmp_file.unlink(missing_ok=True)
            if existing.status == "error":
                self.retry(existing)
            return existing
        d = self.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".mp3"
        shutil.move(str(tmp_file), d / f"source{suffix}")
        job = Job(id=job_id, filename=filename, created=time.time())
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        self.executor.submit(self._run, job)
        return job

    def retry(self, job: Job) -> None:
        if job.status in ("queued", "processing"):
            return
        self.update(job, status="queued", stage="Queued", error=None, analysis_error=None,
                    stats_error=None, progress=0.0)
        self.executor.submit(self._run, job)

    def set_favorite(self, job: Job, favorite: bool) -> None:
        self.update(job, favorite=favorite)

    def sections(self, job: Job) -> list[dict]:
        path = self.dir(job.id) / "sections.json"
        return json.loads(path.read_text()) if path.exists() else []

    def save_sections(self, job: Job, sections: list[dict]) -> list[dict]:
        clean = sorted(
            ({"name": str(s.get("name", "")).strip()[:40] or "Section",
              "start": round(min(max(float(s["start"]), 0.0), job.duration or 0.0), 3)}
             for s in sections),
            key=lambda s: s["start"],
        )
        (self.dir(job.id) / "sections.json").write_text(json.dumps(clean, indent=2))
        return clean

    def cover_path(self, job: Job) -> Path | None:
        return next(self.dir(job.id).glob("cover.*"), None)

    def delete(self, job: Job) -> None:
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(self.dir(job.id), ignore_errors=True)

    # -------------------------------------------------------------- pipeline
    def _run(self, job: Job) -> None:
        if job.id not in self.jobs:  # deleted while queued
            return
        self.update(job, status="processing")
        try:
            if not job.stems_ready:
                self._separate(job)
            if not job.analysis_ready:
                self._analyze(job)
            if not job.stats_ready:
                self._measure(job)
            self.update(job, status="done", stage="Done", progress=1.0)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            log.exception("Job %s failed", job.id)
            self.update(job, status="error", stage="Failed", error=f"{type(e).__name__}: {e}")

    def _separate(self, job: Job) -> None:
        d = self.dir(job.id)
        source = next(d.glob("source.*"))

        t0 = time.time()
        self.update(job, stage="Decoding audio", progress=0.01)
        self.update(job, has_cover=audio.extract_cover(source, d) is not None)
        raw, sr = audio.decode(source)
        duration = raw.shape[-1] / sr
        if duration > MAX_DURATION_S:
            raise ValueError(f"Audio is {duration / 60:.0f} min long; the limit is {MAX_DURATION_S // 60} min.")
        wav = audio.to_stereo(audio.resample(raw, sr, separation.SAMPLE_RATE))
        del raw
        self.update(job, duration=wav.shape[-1] / separation.SAMPLE_RATE, progress=P_DECODE)

        t1 = time.time()
        self.update(job, stage="Loading Demucs model", progress=P_DECODE)
        separation.load_model()

        def on_progress(f: float) -> None:
            self.update(job, save=False, stage=f"Separating stems · {f:.0%}",
                        progress=P_DECODE + (P_SEPARATE - P_DECODE) * f)

        stems, device = separation.separate(wav, on_progress)
        t2 = time.time()

        self.update(job, stage="Writing stems", progress=P_SEPARATE, device=device)
        sr = separation.SAMPLE_RATE
        (d / "stems").mkdir(exist_ok=True)
        audio.write_flac(d / "mix.flac", wav, sr)
        for name in STEMS:
            audio.write_flac(d / "stems" / f"{name}.flac", stems[name], sr)

        tracks = {"mix": wav, **stems}
        raw_peaks = {name: audio.peaks(x, sr, PEAKS_PER_SECOND) for name, x in tracks.items()}
        norm = max(float(p.max()) for p, _ in raw_peaks.values()) or 1.0
        # Square-root amplitude: quiet stems stay readable without flattening dynamics.
        to_bytes = lambda a: np.clip(np.round(np.sqrt(a / norm) * 255), 0, 255).astype(int).tolist()  # noqa: E731
        (d / "peaks.json").write_text(json.dumps({
            "per_second": PEAKS_PER_SECOND,
            "tracks": {n: {"peak": to_bytes(p), "rms": to_bytes(r)} for n, (p, r) in raw_peaks.items()},
        }, separators=(",", ":")))

        n_frames = int(np.ceil(job.duration / HOP_S))
        np.savez_compressed(
            d / "levels.npz", hop=HOP_S,
            **{name: audio.loudness_db(stems[name], sr, HOP_S, n_frames) for name in STEMS},
        )
        timings = {"decode_s": round(t1 - t0, 1), "separate_s": round(t2 - t1, 1),
                   "write_s": round(time.time() - t2, 1)}
        self.update(job, stems_ready=True, progress=P_WRITE, timings={**job.timings, **timings})

    def _analyze(self, job: Job) -> None:
        d = self.dir(job.id)
        t0 = time.time()
        try:
            if recognition.needs_download():
                def on_download(f: float) -> None:
                    self.update(job, save=False, stage=f"Downloading PANNs model (first run only) · {f:.0%}")
                self.update(job, stage="Downloading PANNs model (first run only)")
                recognition.ensure_files(on_download)
                job.timings["download_s"] = round(time.time() - t0, 1)
                t0 = time.time()
            self.update(job, stage="Loading PANNs model")
            recognition.load_model()

            mix, sr = audio.decode(d / "mix.flac")
            mono = audio.resample(mix.mean(axis=0), sr, recognition.SAMPLE_RATE)

            def on_progress(f: float) -> None:
                self.update(job, save=False, stage=f"Recognizing instruments · {f:.0%}",
                            progress=P_WRITE + (P_RECOGNIZE - P_WRITE) * f)

            frames = recognition.detect(mono, on_progress)
            # Average the 10 ms frames into HOP_S bins.
            per_bin = int(round(HOP_S * recognition.SAMPLE_RATE / recognition.HOP))
            n_bins = int(np.ceil(job.duration / HOP_S))
            padded = np.zeros((n_bins * per_bin, frames.shape[1]), dtype=np.float32)
            used = min(len(frames), len(padded))
            padded[:used] = frames[:used]
            counts = np.clip(np.arange(n_bins) * per_bin + per_bin, 0, used) - np.arange(n_bins) * per_bin
            probs = padded.reshape(n_bins, per_bin, -1).sum(axis=1) / np.maximum(counts, 1)[:, None]
            np.savez_compressed(d / "analysis.npz", hop=HOP_S, probs=probs.astype(np.float16),
                                labels=np.array(recognition.labels()))
            self.update(job, analysis_ready=True, analysis_error=None,
                        timings={**job.timings, "recognize_s": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001 - stems stay usable even if this fails
            log.exception("Instrument recognition failed for %s", job.id)
            self.update(job, analysis_error=f"{type(e).__name__}: {e}")

    def _measure(self, job: Job, quiet: bool = False) -> None:
        """Key, tempo, loudness and stem presence; failure leaves the rest usable."""
        d = self.dir(job.id)
        t0 = time.time()
        try:
            if not quiet:
                self.update(job, stage="Measuring key, tempo & loudness", progress=P_RECOGNIZE)
            result = song_stats.compute(d)
            (d / "stats.json").write_text(json.dumps(result))
            tags = []
            if job.analysis_ready:
                report = events.build_report(d, job.duration or 0.0, 0.5)
                tags = [s["family"].lower() for s in report["summary"]]
            tags.append(f"{result['tonic']} {result['mode']}")
            self.update(job, stats_ready=True, stats_error=None, key=result["key"], bpm=result["bpm"],
                        presence=result["presence"], tags=tags,
                        timings={**job.timings, "measure_s": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            log.exception("Measuring failed for %s", job.id)
            self.update(job, stats_error=f"{type(e).__name__}: {e}")

    def _backfill(self, job: Job) -> None:
        if job.id not in self.jobs:
            return
        source = next(self.dir(job.id).glob("source.*"), None)
        if source and not job.has_cover:
            self.update(job, has_cover=audio.extract_cover(source, self.dir(job.id)) is not None)
        self._measure(job, quiet=True)
