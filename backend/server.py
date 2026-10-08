"""HTTP API + static frontend. Run with:  python -m uvicorn backend.server:app --port 8765"""
from __future__ import annotations

import hashlib
import logging
import tempfile
import zipfile
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import events
from .jobs import STEMS, Job, JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ALLOWED_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}

app = FastAPI(title="Stemscope")
store = JobStore(DATA / "jobs")


def _job_or_404(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such song")
    return job


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix or '?'}'. Upload an MP3 (or WAV/FLAC/M4A).")
    (DATA / "uploads").mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=DATA / "uploads", delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(1 << 20):
            digest.update(chunk)
            tmp.write(chunk)
    job = store.add_upload(digest.hexdigest()[:16], Path(tmp.name), file.filename or f"upload{suffix}")
    return job.public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public() for j in store.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_or_404(job_id).public()


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    store.retry(job)
    return job.public()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job.status in ("queued", "processing"):
        raise HTTPException(409, "This song is still processing")
    store.delete(job)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/peaks")
def get_peaks(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if not job.stems_ready:
        raise HTTPException(409, "Stems are not ready yet")
    return FileResponse(store.dir(job_id) / "peaks.json", media_type="application/json")


def _title(job: Job) -> str:
    return Path(job.filename).stem


@app.get("/api/jobs/{job_id}/audio/{track}.flac")
def get_audio(job_id: str, track: str, download: bool = False) -> FileResponse:
    job = _job_or_404(job_id)
    if not job.stems_ready or track not in ("mix", *STEMS):
        raise HTTPException(404, "No such track")
    path = store.dir(job_id) / ("mix.flac" if track == "mix" else f"stems/{track}.flac")
    # Job ids are content hashes, so a given URL always serves the same audio.
    return FileResponse(
        path, media_type="audio/flac", headers={"Cache-Control": "max-age=86400"},
        filename=f"{_title(job)} - {track}.flac" if download else None,
    )


@app.get("/api/jobs/{job_id}/stems.zip")
def get_stems_zip(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if not job.stems_ready:
        raise HTTPException(409, "Stems are not ready yet")
    d = store.dir(job_id)
    path = d / "stems.zip"
    if not path.exists():
        tmp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # FLAC is already compressed
            for name in STEMS:
                z.write(d / "stems" / f"{name}.flac", f"{_title(job)} - {name}.flac")
        tmp.replace(path)
    return FileResponse(path, media_type="application/zip", filename=f"{_title(job)} - stems.zip")


@app.get("/api/jobs/{job_id}/stats")
def get_stats(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if not job.stats_ready:
        raise HTTPException(409, job.stats_error or "Stats are not ready yet")
    return FileResponse(store.dir(job_id) / "stats.json", media_type="application/json")


@app.get("/api/jobs/{job_id}/cover")
def get_cover(job_id: str) -> FileResponse:
    path = store.cover_path(_job_or_404(job_id))
    if path is None:
        raise HTTPException(404, "No cover art")
    return FileResponse(path)


@app.post("/api/jobs/{job_id}/favorite")
def set_favorite(job_id: str, favorite: bool = Body(..., embed=True)) -> dict:
    job = _job_or_404(job_id)
    store.set_favorite(job, favorite)
    return job.public()


@app.get("/api/jobs/{job_id}/sections")
def get_sections(job_id: str) -> list[dict]:
    return store.sections(_job_or_404(job_id))


@app.put("/api/jobs/{job_id}/sections")
def put_sections(job_id: str, sections: list[dict] = Body(...)) -> list[dict]:
    job = _job_or_404(job_id)
    try:
        return store.save_sections(job, sections)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid sections: {e}") from e


@app.get("/api/jobs/{job_id}/analysis")
def get_analysis(job_id: str, sensitivity: float = 0.5) -> JSONResponse:
    job = _job_or_404(job_id)
    if not job.analysis_ready:
        raise HTTPException(409, job.analysis_error or "Instrument recognition is not ready yet")
    return JSONResponse(events.build_report(store.dir(job_id), job.duration or 0.0, sensitivity))


class Frontend(StaticFiles):
    """Static files that the browser revalidates on every load, so edits show up after a refresh."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", Frontend(directory=ROOT / "frontend", html=True), name="frontend")
