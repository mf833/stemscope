# Stemscope

A local web app for taking a song apart. Upload an MP3 and it:

1. **Separates stems** with Demucs `htdemucs_6s`: vocals, drums, bass, guitar, piano, other.
2. **Shows a multitrack player**: every stem (plus the original mix, muted by default for A/B)
   is a waveform lane on a shared, zoomable timeline with play, mute, solo and volume per track.
3. **Recognizes instruments** by running PANNs (`Cnn14_DecisionLevelMax`, trained on AudioSet)
   over the original song. The output is a confidence heatmap per instrument family, aligned to the same timeline.
4. **Lists events**, e.g. *"Guitar enters at 0:12, gets louder at 1:05, drops out at 2:40."*

It also measures each song (key and scale, BPM and tempo stability, LUFS and loudness range, and how much
of the song each stem plays), and adds sections, A–B looping, a click track on the detected beats,
an Original/Stems A/B switch, mix export, favorites and tags.

Everything runs on your Mac; nothing is uploaded anywhere.

## Run it

```bash
./run.sh
```

That starts the server on <http://127.0.0.1:8765> and opens it in your browser. The first run of
`run.sh` creates `.venv` and installs the requirements. The first song you process also downloads
the model weights: Demucs (~55 MB, to `~/.cache`) and PANNs (~330 MB, to `~/panns_data`).

Typical speed on Apple silicon: a 4-minute song takes about 30 s to separate on the GPU (MPS) and
a few seconds for instrument recognition.

## Using it

- **Upload**: drag a file anywhere onto the page, or use *Upload song*. MP3, WAV, FLAC, OGG, M4A and AIFF work.
  Re-uploading the same file reopens the cached result instantly. *Library* lists past songs.
- **Transport**: `Space` play/pause, `←`/`→` skip 5 s (`Shift` for 1 s), `Home` back to start.
  Click or drag on any lane to seek. Zoom with the slider, `+`/`−`, or pinch / ⌘-scroll on the timeline.
- **Per track**: drag the fader (dB), `▶` plays that stem on its own, `M` mutes, `S` solos (⌥-click to solo
  only that track), `⤓` downloads the stem as FLAC. Double-click a fader to reset it to 0 dB.
  *Show* chips in the top bar hide stems you don't need (hidden stems are silent).
- **Sections**: `+ Add` marks a section at the playhead (Intro, Verse, Chorus…). Click one to jump there,
  ⌥-click to loop it, double-click to rename, right-click to delete. Sections are saved with the song.
- **Loop**: drag across the time ruler, or press `L` (loops the current section, or four bars).
- **Listen**: *Original* plays the untouched file for comparison; *Stems* plays your mix.
- **Click track**: a metronome on the beats detected in the drum stem, with optional four-beat count-in,
  bar/beat grid, and /2 · ×2 for half or double time. The accented "beat 1" is a guess.
- **Export**: *Current mix* renders exactly what you hear (mutes, solos, faders) to a WAV in your Downloads;
  *All stems* downloads a ZIP of the six FLACs.
- **Library**: search (`⌘K`), favorites (♡ under the player), and tags built from detected instruments and key.
- **Detected instruments**: each row is an instrument family. Brightness is PANNs confidence; the bar
  underneath marks where it counts as *playing*. Click a row to expand the individual AudioSet classes.
  Hover for exact values.
- **Instrument events**: a one-line story per instrument plus a chronological log; click any time to jump there.
  The *Detection sensitivity* slider re-thresholds instantly, with no reprocessing.

## How events are derived

- **Enters / drops out** come from PANNs only. Frame probabilities (100 per second) are averaged into 0.25 s bins,
  smoothed over 0.75 s, then the max is taken over each family's classes (for example Guitar = guitar, electric,
  acoustic, slide, strum). A family is *playing* once its confidence reaches the trigger level (20% at the default
  sensitivity) and until it falls below half of that. Gaps under 2.5 s are bridged and blips under 2 s are ignored.
- **Gets louder / quieter** comes from the matching Demucs stem, because PANNs confidence is not loudness. Within each
  playing segment, the stem's level over the 4 s before each moment is compared with the 4 s after. Steps of 6 dB
  or more (at default sensitivity) are reported, at least 8 s apart. Only families with their own stem
  (vocals, drums, bass, guitar, piano) get these events.
- PANNs confidences are uncalibrated and run lower in dense mixes than on solo instruments. If an instrument you
  can hear isn't detected, raise the sensitivity.

## How the song stats are measured

- **Key / scale**: chroma of every stem except drums, matched against the Krumhansl–Kessler key profiles.
  The % badge is how strongly the best key wins over the other 23.
- **BPM / tempo stability**: beat tracking on the drum stem (librosa). Stability is the share of beats within
  5% of the main tempo. Half- or double-time readings are common; the click track's /2 and ×2 fix that.
- **LUFS / dynamic range**: ITU-R BS.1770 integrated loudness and EBU-style loudness range (LU).
- **Presence**: the share of the song where a stem is above −50 dBFS and within 24 dB of the full mix,
  which ignores the faint bleed that separation leaves behind.

## Layout

```
backend/
  server.py        FastAPI routes + static frontend
  jobs.py          job store and the processing pipeline (one job at a time, background thread)
  separation.py    Demucs wrapper (MPS with CPU fallback, progress callback)
  recognition.py   PANNs wrapper (parallel checkpoint download, chunked inference)
  instruments.py   AudioSet classes -> instrument families, colors, stem mapping
  events.py        presence segments, loudness steps, summary sentences
  stats.py         key, tempo + beats, loudness, stem presence
  audio.py         decode / resample / waveform peaks / loudness curves
frontend/          index.html, style.css, app.js (no build step; Web Audio + canvas)
tests/             python -m unittest discover tests
data/jobs/<id>/    per-song results: source file, mix.flac, stems/*.flac, peaks.json, *.npz
```

Delete a song from the *Library* menu (×) to free its disk space (~25 MB per minute of audio).

## Configuration

- `PORT=9000 ./run.sh` uses a different port.
- `STEMSCOPE_DEVICE=cpu ./run.sh` forces CPU (by default it uses the Apple GPU when available).
- Songs longer than 15 minutes are rejected, because the browser would need several GB to hold all the tracks.
