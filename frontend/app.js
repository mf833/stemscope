"use strict";

/* Stemscope frontend: song library, stats header, a Web Audio multitrack mixer for
 * the Demucs stems, the PANNs instrument timeline, sections, loop, click track and
 * export, all drawn on one shared, zoomable time axis. */

const STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"];
const COLORS = {
  mix: "#c9ced8", vocals: "#ec5f74", drums: "#f08a3e", bass: "#e9b949",
  guitar: "#62c972", piano: "#a98bfa", other: "#8fa5cc",
};
const LABELS = { mix: "Original", vocals: "Vocals", drums: "Drums", bass: "Bass", guitar: "Guitar", piano: "Piano", other: "Other" };
const PRESENCE_LABELS = {
  vocals: "Vocal presence", drums: "Drum presence", bass: "Bass presence",
  guitar: "Guitar presence", piano: "Piano presence", other: "Other",
};
const EVENT_ICONS = { enter: "↗", exit: "↘", louder: "▲", quieter: "▼" };
const SECTION_SUGGESTIONS = ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"];
const SECTION_COLORS = {
  intro: "#8b909c", outro: "#8b909c", verse: "#6ea8fe", chorus: "#f5b31d", "pre-chorus": "#62c972",
  bridge: "#a98bfa", solo: "#ec5f74", drop: "#f08a3e", hook: "#f5b31d", breakdown: "#8fa5cc",
};
const POLL_MS = 700;
const MAX_PPS = 100; // pixels per second at full zoom; matches the server's peak resolution
const MIN_DB = -40; // fader floor, treated as silence
const SANS = '-apple-system, BlinkMacSystemFont, "Inter", sans-serif';
const MONO = '"SF Mono", ui-monospace, Menlo, monospace';

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const mod = (n, m) => ((n % m) + m) % m;
const pct = (v) => `${Math.round(v * 100)}%`;
const pad2 = (n) => String(n).padStart(2, "0");
const DPR = () => window.devicePixelRatio || 1;

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c != null && c !== false) node.append(c instanceof Node ? c : String(c));
  }
  return node;
}

function icon(name, cls = "i") {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", cls);
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

const fmt = (t) => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${pad2(Math.floor(t % 60))}`; };
const mmss = (t) => { t = Math.max(0, t || 0); return `${pad2(Math.floor(t / 60))}:${pad2(Math.floor(t % 60))}`; };
const mmssff = (t) => `${mmss(t)}.${pad2(Math.floor((Math.max(0, t) % 1) * 100))}`;
const dbToGain = (db) => (db <= MIN_DB ? 0 : 10 ** (db / 20));
const titleOf = (job) => job.filename.replace(/\.[^.]+$/, "");

function lowerBound(arr, t, key = (x) => x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (key(arr[m]) < t) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function paintRange(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  input.style.setProperty("--fill", `${((Number(input.value) - min) / (max - min)) * 100}%`);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

let toastTimer;
function toast(msg, ok = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = ok ? "toast ok" : "toast";
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ok ? 2500 : 6000);
}

function downloadUrl(url, filename) {
  const a = h("a", { href: url, download: filename || "" });
  document.body.append(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------------- audio engine
/* All tracks are AudioBuffers started on one AudioContext clock, so stems stay
 * sample-locked. Mute/solo/faders are gain ramps; seeking restarts the sources.
 * A loop uses the sources' native looping, so it is seamless. */
class Engine {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.tracks = [];
    this.playing = false;
    this.offset = 0; // song position when paused, or when playback last started
    this.startAt = 0; // ctx.currentTime at which `offset` plays
    this.duration = 0;
    this.mode = "stems"; // or "original" for A/B against the source file
    this.loop = null; // {a, b} in seconds
    this.loopOn = false;
    this.activeLoop = null; // loop the current sources were started with
    this.onStart = null;
  }
  ensureCtx() {
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      this.master = this.ctx.createGain();
      this.master.gain.value = Number($("masterVol").value);
      this.master.connect(this.ctx.destination);
    }
    return this.ctx;
  }
  reset() {
    this.stopSources();
    for (const t of this.tracks) t.gain?.disconnect();
    Object.assign(this, { tracks: [], playing: false, offset: 0, loop: null, loopOn: false, activeLoop: null, mode: "stems" });
  }
  addTrack(t) {
    const ctx = this.ensureCtx();
    t.gain = ctx.createGain();
    t.gain.connect(this.master);
    this.tracks.push(t);
  }
  get ready() { return this.tracks.length > 0 && this.tracks.every((t) => t.buffer); }
  get position() {
    if (!this.playing) return this.offset;
    let p = this.offset + Math.max(0, this.ctx.currentTime - this.startAt);
    const L = this.activeLoop;
    if (L && p >= L.b) p = L.a + ((p - L.b) % (L.b - L.a));
    return clamp(p, 0, this.duration);
  }
  async play(delay = 0) {
    if (!this.ready || this.playing) return;
    await this.ctx.resume();
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    if (this.loopOn && this.loop && this.offset >= this.loop.b) this.offset = this.loop.a;
    this.startSources(delay);
    this.playing = true;
  }
  pause() {
    if (!this.playing) return;
    this.offset = this.position;
    this.stopSources();
    this.playing = false;
  }
  seek(t) {
    t = clamp(t, 0, this.duration);
    if (this.playing) {
      this.stopSources();
      this.offset = t;
      this.startSources();
    } else {
      this.offset = t;
    }
  }
  restart() { if (this.playing) this.seek(this.position); }
  startSources(delay = 0) {
    const when = this.ctx.currentTime + 0.03 + delay;
    const L = this.loopOn && this.loop && this.offset < this.loop.b ? { ...this.loop } : null;
    this.activeLoop = L;
    for (const t of this.tracks) {
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      if (L) {
        src.loop = true;
        src.loopStart = L.a;
        src.loopEnd = L.b;
      }
      src.connect(t.gain);
      src.start(when, this.offset);
      t.source = src;
    }
    this.startAt = when;
    this.onStart?.();
  }
  stopSources() {
    for (const t of this.tracks) {
      if (!t.source) continue;
      try { t.source.stop(); } catch { /* already stopped */ }
      t.source.disconnect();
      t.source = null;
    }
  }
  applyGains(instant = false) {
    const anySolo = this.tracks.some((t) => t.name !== "mix" && t.soloed && t.visible);
    for (const t of this.tracks) {
      t.audible = t.name === "mix"
        ? this.mode === "original"
        : this.mode === "stems" && t.visible && !t.muted && (!anySolo || t.soloed);
      const v = t.audible ? (t.name === "mix" ? 1 : dbToGain(t.db)) : 0;
      if (instant) t.gain.gain.value = v;
      else t.gain.gain.setTargetAtTime(v, this.ctx.currentTime, 0.015);
    }
  }
}

// ---------------------------------------------------------------- click track
/* Metronome on the detected beats, scheduled slightly ahead on the audio clock. */
const Click = {
  on: false,
  countIn: false,
  rate: 1,
  vol: 0.6,
  list: [], // [{t, accent}]
  idx: -1,
  lastPos: 0,
  rebuild() {
    this.list = [];
    this.idx = -1;
    const st = S.stats;
    if (!st?.beats?.length) return;
    const B = st.beats;
    const d = st.downbeat || 0;
    for (let i = 0; i < B.length; i++) {
      const k = mod(i - d, 4);
      if (this.rate === 0.5 && k % 2) continue;
      this.list.push({ t: B[i], accent: k === 0 });
      if (this.rate === 2 && i + 1 < B.length) this.list.push({ t: (B[i] + B[i + 1]) / 2, accent: false });
    }
  },
  blip(when, accent) {
    const ctx = engine.ctx;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.frequency.value = accent ? 1760 : 1180;
    const peak = Math.max(0.0002, this.vol * (accent ? 0.9 : 0.55));
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(peak, when + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, when + 0.06);
    osc.connect(g);
    g.connect(engine.master);
    osc.start(when);
    osc.stop(when + 0.08);
  },
  tick() {
    if (!this.on || !engine.playing || !this.list.length) return;
    const pos = engine.position;
    // Before playback actually starts (count-in), song time `offset` maps to `startAt`.
    const base = Math.max(engine.ctx.currentTime, engine.startAt);
    if (this.idx < 0 || pos < this.lastPos - 0.05) this.idx = lowerBound(this.list, pos - 0.01, (c) => c.t);
    this.lastPos = pos;
    const loop = engine.activeLoop;
    while (this.idx < this.list.length && this.list[this.idx].t < pos + 0.12) {
      const c = this.list[this.idx];
      if (loop && c.t >= loop.b) break; // resumes when the position wraps
      this.idx++;
      if (c.t >= pos - 0.02) this.blip(base + Math.max(0, c.t - pos), c.accent);
    }
  },
};

// ---------------------------------------------------------------- state
const engine = new Engine();
engine.onStart = () => { Click.idx = -1; };
const scroller = $("scroller");
const S = {
  token: 0, // bumped whenever the open song changes; stale async work checks it
  jobId: null,
  job: null,
  peaks: null,
  analysis: null,
  stats: null,
  sections: [],
  analysisSeq: 0,
  duration: 0,
  zoom: 1, maxZoom: 1, pps: 1, vw: 0, hw: 304,
  tracks: [],
  laneRows: [], // rows that own a canvas: {kind, canvas, lane, draw, ...}
  expanded: new Set(),
  sensitivity: 0.5,
  visible: new Set(STEMS),
  gridOn: false,
  pollTimer: null,
  libTimer: null,
  drag: null,
  eventEls: [],
  currentEvent: -1,
  library: [],
  libView: "all",
  search: "",
  tagFilter: null,
};

// ---------------------------------------------------------------- library sidebar
function thumb(job, cls = "thumb") {
  if (job.has_cover) return h("div", { class: `${cls} img` }, h("img", { src: `/api/jobs/${job.id}/cover`, alt: "" }));
  if (job.presence && Object.keys(job.presence).length) {
    return h("div", { class: cls, title: "Stem mix: how much each stem plays" },
      STEMS.map((s) => h("span", { style: `--c:${COLORS[s]};height:${Math.max(8, Math.round((job.presence[s] || 0) * 100))}%` })));
  }
  return h("div", { class: `${cls} letter` }, (job.filename[0] || "?").toUpperCase());
}

function songSub(job) {
  if (job.status === "error" && !job.stems_ready) return "failed";
  if (!job.stems_ready) return job.stage || "processing…";
  const bits = [mmss(job.duration), "6 stems"];
  if (job.key) bits.push(job.key);
  if (job.bpm) bits.push(`${Math.round(job.bpm)} BPM`);
  return bits.join(" · ");
}

async function refreshLibrary() {
  try { S.library = await api("/api/jobs"); } catch { return; }
  renderLibrary();
  // Keep refreshing while anything is still processing or measuring in the background.
  clearTimeout(S.libTimer);
  const pending = S.library.some((j) => ["queued", "processing"].includes(j.status)
    || (j.status === "done" && j.stems_ready && !j.stats_ready && !j.stats_error));
  if (pending) S.libTimer = setTimeout(refreshLibrary, 4000);
}

function renderLibrary() {
  const q = S.search.trim().toLowerCase();
  let list = S.library;
  if (S.libView === "favorites") list = list.filter((j) => j.favorite);
  if (S.tagFilter) list = list.filter((j) => (j.tags || []).includes(S.tagFilter));
  if (q) list = list.filter((j) => j.filename.toLowerCase().includes(q) || (j.tags || []).some((t) => t.includes(q)));

  $("listTitle").textContent = S.libView === "favorites" ? "Favorites" : "Recent";
  $("listCount").textContent = list.length ? String(list.length) : "";
  const empty = !S.library.length ? "No songs yet. Upload one to get started."
    : S.libView === "favorites" && !q && !S.tagFilter ? "No favorites yet. Use ♡ under the player."
      : "Nothing matches.";
  $("songList").replaceChildren(...(list.length ? list.map((job) => h("div", {
    class: `song-item${job.id === S.jobId ? " active" : ""}`,
    onclick: () => { $("app").classList.remove("side-open"); if (job.id !== S.jobId) openJob(job.id); },
  },
  thumb(job),
  h("div", { style: "min-width:0" }, h("div", { class: "si-title", title: job.filename }, titleOf(job)),
    h("div", { class: "si-sub" }, songSub(job))),
  h("div", { class: "si-side" },
    job.favorite ? icon("heart", "i si-fav") : null,
    h("span", { class: `si-dot ${job.status}` }),
    h("button", {
      class: "si-del", title: "Delete song and stems",
      onclick: (e) => { e.stopPropagation(); deleteJob(job); },
    }, icon("x"))))) : [h("div", { class: "empty" }, empty)]));

  const counts = new Map();
  for (const j of S.library) for (const t of j.tags || []) counts.set(t, (counts.get(t) || 0) + 1);
  const tags = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  $("tagList").replaceChildren(...(tags.length ? tags.map(([t, n]) => h("button", {
    class: `tag${t === S.tagFilter ? " on" : ""}`,
    onclick: () => { S.tagFilter = S.tagFilter === t ? null : t; renderLibrary(); },
  }, t, h("b", {}, n))) : [h("div", { class: "empty" }, "Detected instruments and keys appear here.")]));
  $("clearTag").hidden = !S.tagFilter;
}

async function deleteJob(job) {
  if (!confirm(`Delete “${titleOf(job)}” and its stems?`)) return;
  try {
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
  } catch (e) {
    toast(e.message);
    return;
  }
  if (job.id === S.jobId) goHome();
  refreshLibrary();
}

function setLibView(view) {
  S.libView = view;
  for (const b of document.querySelectorAll("[data-libview]")) b.classList.toggle("on", b.dataset.libview === view);
  showSidebar();
  renderLibrary();
}

function narrow() { return window.innerWidth <= 1180; }
function showSidebar() {
  if (narrow()) $("app").classList.add("side-open");
  else $("app").classList.remove("side-closed");
}
function toggleSidebar() {
  if (narrow()) $("app").classList.toggle("side-open");
  else $("app").classList.toggle("side-closed");
}

// ---------------------------------------------------------------- views
function showView(name) {
  for (const v of ["welcome", "processing", "workspace"]) $(`${v}View`).hidden = v !== name;
  $("transport").hidden = name !== "workspace";
  for (const c of document.querySelectorAll(".chip")) c.disabled = name !== "workspace";
}

function setSongTitle(job) {
  $("fileFieldText").textContent = job ? job.filename : "Drop an MP3 here, or click to choose a song…";
  document.title = job ? `${titleOf(job)} · Stemscope` : "Stemscope";
}

function leaveSong() {
  S.token++;
  clearTimeout(S.pollTimer);
  engine.reset();
  Click.on = false;
  Object.assign(S, {
    jobId: null, job: null, peaks: null, analysis: null, stats: null, sections: [], tracks: [], laneRows: [],
    duration: 0, loadingWorkspace: false, statsLoading: false,
  });
  updatePlayUI();
  renderTransportControls();
}

function goHome() {
  leaveSong();
  history.replaceState(null, "", location.pathname);
  setSongTitle(null);
  showView("welcome");
  renderLibrary();
}

// ---------------------------------------------------------------- upload, open, poll
async function uploadFile(file) {
  if (!file) return;
  leaveSong();
  setSongTitle({ filename: file.name });
  showView("processing");
  $("pcFile").textContent = file.name;
  $("pcStage").textContent = "Uploading…";
  $("pcBar").style.width = "0%";
  $("pcMeta").textContent = "";
  $("pcError").hidden = true;
  setSteps(null);
  const form = new FormData();
  form.append("file", file);
  try {
    const job = await api("/api/upload", { method: "POST", body: form });
    refreshLibrary();
    openJob(job.id);
  } catch (e) {
    showProcessingError(e.message, null);
  }
}

async function openJob(id) {
  leaveSong();
  const token = S.token;
  S.jobId = id;
  if (location.hash.slice(1) !== id) history.replaceState(null, "", `#${id}`);
  renderLibrary();
  let job;
  try {
    job = await api(`/api/jobs/${id}`);
  } catch (e) {
    toast(e.message);
    goHome();
    return;
  }
  if (token === S.token) onJob(job, token);
}

function onJob(job, token) {
  const prev = S.job;
  S.job = job;
  setSongTitle(job);
  if (!job.stems_ready) {
    showView("processing");
    renderProgress(job);
  } else if (!S.peaks) {
    if (!S.loadingWorkspace) loadWorkspace(job, token);
  } else {
    if (!S.analysis) {
      if (job.analysis_ready) loadAnalysis(token);
      else { buildDetectionRows(); renderAnalysisPanel(); }
    }
    if (!S.stats) {
      if (job.stats_ready) loadStats(token);
      else renderStats();
    }
    renderNowPlaying();
  }
  const busy = ["queued", "processing"].includes(job.status)
    || (job.status === "done" && job.stems_ready && !job.stats_ready && !job.stats_error);
  if (busy) {
    clearTimeout(S.pollTimer);
    S.pollTimer = setTimeout(() => poll(token), POLL_MS);
  }
  if (prev && (prev.status !== job.status || prev.stats_ready !== job.stats_ready)) refreshLibrary();
}

async function poll(token) {
  if (token !== S.token) return;
  try {
    const job = await api(`/api/jobs/${S.jobId}`);
    if (token === S.token) onJob(job, token);
  } catch {
    if (token === S.token) S.pollTimer = setTimeout(() => poll(token), POLL_MS * 3);
  }
}

async function retryJob() {
  try {
    const job = await api(`/api/jobs/${S.jobId}/retry`, { method: "POST" });
    onJob(job, S.token);
  } catch (e) {
    toast(e.message);
  }
}

const STEP_OF = [
  [/^Decoding/, "decode"], [/Demucs|^Separating/, "separate"], [/^Writing/, "write"],
  [/PANNs|^Recognizing/, "recognize"], [/^Measuring/, "measure"],
];
function setSteps(stage, failed) {
  const order = ["decode", "separate", "write", "recognize", "measure"];
  const cur = stage ? (STEP_OF.find(([re]) => re.test(stage)) || [])[1] : null;
  const ci = cur ? order.indexOf(cur) : -1;
  for (const li of $("pcSteps").children) {
    const i = order.indexOf(li.dataset.step);
    li.className = i < ci ? "done" : i === ci && !failed ? "active" : "";
  }
}

function renderProgress(job) {
  $("pcFile").textContent = job.filename;
  $("pcBar").style.width = `${Math.round(job.progress * 100)}%`;
  if (job.status === "error") {
    showProcessingError(job.error, job);
    return;
  }
  $("pcError").hidden = true;
  $("pcStage").textContent = job.stage;
  setSteps(job.stage);
  const bits = [`${fmt(Math.max(0, Date.now() / 1000 - job.created))} elapsed`];
  if (job.duration) bits.push(`${fmt(job.duration)} of audio`);
  $("pcMeta").textContent = bits.join(" · ");
}

function showProcessingError(message, job) {
  $("pcStage").textContent = "Something went wrong";
  setSteps(job?.stage, true);
  const box = $("pcError");
  box.hidden = false;
  box.replaceChildren(h("div", {}, message || "Unknown error"));
  if (job) box.append(h("button", { class: "pill", onclick: retryJob }, "Retry"));
}

// ---------------------------------------------------------------- workspace
async function loadWorkspace(job, token) {
  S.loadingWorkspace = true;
  try {
    S.peaks = await api(`/api/jobs/${job.id}/peaks`);
  } catch (e) {
    toast(e.message);
    return;
  } finally {
    S.loadingWorkspace = false;
  }
  if (token !== S.token) return;
  showView("workspace");
  S.duration = job.duration;
  S.zoom = 1;
  S.visible = new Set(STEMS);
  $("totTime").textContent = `/ ${mmss(job.duration)}`;
  engine.reset();
  engine.duration = job.duration;
  S.tracks = job.tracks.map((t) => ({
    ...t, label: LABELS[t.name], color: COLORS[t.name], muted: false, soloed: false, db: 0, visible: true, buffer: null,
  }));
  for (const t of S.tracks) engine.addTrack(t);
  engine.applyGains(true);
  S.laneRows = [];
  buildHeadRows();
  buildTrackRows();
  buildDetectionRows();
  renderChips();
  renderStats();
  renderAnalysisPanel();
  renderNowPlaying();
  renderTransportControls();
  layout();
  scroller.scrollLeft = 0;
  $("main").scrollTop = 0;
  if (job.analysis_ready) loadAnalysis(token);
  if (job.stats_ready) loadStats(token);
  loadSections(token);
  loadAudio(token);
}

async function loadAudio(token) {
  const ctx = engine.ensureCtx();
  let done = 0;
  const total = S.tracks.length;
  const status = () => setStatus(`Loading audio ${done}/${total}…`);
  status();
  try {
    await Promise.all(S.tracks.map(async (t) => {
      const res = await fetch(t.url);
      if (!res.ok) throw new Error(`${t.label}: HTTP ${res.status}`);
      const buffer = await ctx.decodeAudioData(await res.arrayBuffer());
      if (token !== S.token) return;
      t.buffer = buffer;
      if (t.subEl) t.subEl.textContent = "";
      done++;
      status();
    }));
  } catch (e) {
    if (token === S.token) toast(`Could not load audio. ${e.message}`);
    return;
  }
  if (token !== S.token) return;
  updatePlayUI();
  updateStatus();
}

async function loadStats(token) {
  if (S.statsLoading) return;
  S.statsLoading = true;
  try {
    const st = await api(`/api/jobs/${S.jobId}/stats`);
    if (token !== S.token) return;
    S.stats = st;
  } catch {
    return;
  } finally {
    S.statsLoading = false;
  }
  Click.rebuild();
  renderStats();
  renderNowPlaying();
  renderTransportControls();
  updateStatus();
  drawAll();
}

async function loadSections(token) {
  try {
    const sections = await api(`/api/jobs/${S.jobId}/sections`);
    if (token !== S.token) return;
    S.sections = sections;
  } catch {
    S.sections = [];
  }
  setSaved(S.sections.length ? "saved" : "");
  drawAll();
}

function makeRow(parent, cls, headChildren, { canvas = true, style = "" } = {}) {
  const head = h("div", { class: "row-head" }, headChildren);
  const lane = h("div", { class: "lane" });
  const row = h("div", { class: `row ${cls}`, style }, head, lane);
  const r = { row, head, lane, canvas: null };
  if (canvas) {
    r.canvas = h("canvas");
    lane.append(r.canvas);
  }
  row._r = r;
  parent.append(row);
  return r;
}

// ---------------------------------------------------------------- stats header
function renderStats() {
  const job = S.job;
  const st = S.stats;
  if (!job) return;
  const wait = job.stats_error ? "unavailable" : "measuring…";
  const card = (label, value, sub, opts = {}) => h("div", {
    class: `stat${opts.loading ? " loading" : ""}`, style: opts.color ? `--c:${opts.color}` : null, title: opts.title,
  }, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value), h("div", { class: "stat-sub" }, sub));

  const drLabel = (lu) => (lu < 5 ? "Compressed" : lu < 9 ? "Moderate" : lu < 15 ? "Wide" : "Very wide");
  const stabLabel = (s) => (s >= 0.95 ? "Very stable" : s >= 0.85 ? "Stable" : s >= 0.7 ? "Some drift" : "Loose");
  const stabColor = (s) => (s >= 0.9 ? "var(--green)" : s >= 0.75 ? "var(--amber)" : "var(--red)");

  $("statsTop").replaceChildren(
    card("Key", st ? [st.key, h("span", {
      class: "badge-conf", title: "How strongly the harmony matches this key compared with the other 23",
    }, pct(st.confidence))] : "—", st ? `relative ${st.relative}` : wait, { loading: !st }),
    card("BPM", st?.bpm ? String(Math.round(st.bpm)) : "—",
      st?.bpm ? `${st.bpm.toFixed(1)} · ${st.beats.length} beats` : st ? "no steady beat" : wait,
      { loading: !st, title: "Tempo from the beats detected in the drum stem" }),
    card("LUFS", st ? st.lufs.toFixed(1) : "—", st ? `Peak ${st.peak_db.toFixed(1)} dBFS` : wait,
      { loading: !st, title: "Integrated loudness (ITU-R BS.1770)" }),
    card("Duration", mmss(job.duration), `${job.format} · 6 stems`),
    card("Scale", st ? st.scale : "—", st ? st.scale_notes.join(" ") : wait, { loading: !st }),
    card("Dynamic range", st ? [String(Math.round(st.lra)), h("small", {}, "LU")] : "—", st ? drLabel(st.lra) : wait,
      { loading: !st, title: "Loudness range: spread between quiet and loud passages (EBU R128 LRA)" }),
    card("Tempo stability", st?.stability != null ? pct(st.stability) : "—",
      st?.stability != null ? stabLabel(st.stability) : st ? "no steady beat" : wait,
      { loading: !st, color: st?.stability != null ? stabColor(st.stability) : null,
        title: "Share of beats within 5% of the main tempo" }),
  );
  $("statsStems").replaceChildren(...STEMS.map((s) => {
    const v = st?.presence?.[s];
    return card(PRESENCE_LABELS[s], v != null ? pct(v) : "—", "of the song", {
      loading: v == null, color: COLORS[s],
      title: `Share of the song where the ${s} stem is clearly audible (above −50 dBFS and within 24 dB of the full mix)`,
    });
  }));
}

// ---------------------------------------------------------------- timeline rows
function setSaved(state) {
  if (!S.saveEl) return;
  S.saveEl.className = state === "saving" ? "saved busy" : "saved";
  S.saveEl.replaceChildren(...(state === "saved" ? [icon("check"), "Saved"] : state === "saving" ? ["Saving…"] : []));
}

function buildHeadRows() {
  const box = $("headRows");
  box.replaceChildren();
  S.saveEl = h("span", { class: "saved" });
  const sec = makeRow(box, "sections", [
    h("span", { class: "rh-title" }, "Sections"), S.saveEl,
    h("button", { class: "mini-pill", title: "Mark a section starting at the playhead", onclick: addSection }, icon("plus"), "Add"),
  ]);
  Object.assign(sec, { kind: "sections", draw: drawSections });
  sec.lane.addEventListener("click", onSectionClick);
  sec.lane.addEventListener("dblclick", onSectionRename);
  sec.lane.addEventListener("contextmenu", onSectionDelete);
  const ruler = makeRow(box, "ruler", [h("span", { class: "rh-title" }, "Mixer"), h("span", { class: "rh-hint" }, "Drag fader · M/S")]);
  Object.assign(ruler, { kind: "ruler", draw: drawRuler });
  S.laneRows.push(sec, ruler);
}

const dbText = (db) => (db <= MIN_DB ? "−∞" : `${db > 0 ? "+" : ""}${db.toFixed(1)}`);

function buildTrackRows() {
  const box = $("trackRows");
  box.replaceChildren();
  for (const t of S.tracks) {
    if (t.name === "mix") continue;
    t.subEl = h("small", {}, "loading…");
    t.dbEl = h("span", { class: "db" }, "0.0");
    const fader = h("input", {
      type: "range", class: "fader", min: String(MIN_DB), max: "6", step: "0.5", value: "0",
      style: `--c:${t.color}`, title: "Level in dB (double-click to reset)", "aria-label": `${t.label} level`,
    });
    paintRange(fader);
    t.btnPlay = h("button", { class: "sq audition", title: `Play ${t.label.toLowerCase()} on its own` }, icon("play"));
    t.btnMute = h("button", { class: "sq mute", title: "Mute" }, "M");
    t.btnSolo = h("button", { class: "sq solo", title: "Solo (⌥-click: solo only this)" }, "S");
    const dl = h("a", {
      class: "sq dl", href: `${t.url}?download=1`, download: "", title: `Download the ${t.label.toLowerCase()} stem (FLAC)`,
    }, icon("download"));
    const r = makeRow(box, `track ${t.name}`, [
      h("span", { class: "track-name" }, t.label, t.subEl), fader, t.dbEl,
      h("div", { class: "sqs" }, t.btnPlay, t.btnMute, t.btnSolo, dl),
    ], { style: `--c:${t.color}` });
    Object.assign(r, { kind: "track", track: t, draw: drawWave });
    t.rowEl = r.row;
    S.laneRows.push(r);

    fader.oninput = () => {
      t.db = Number(fader.value);
      t.dbEl.textContent = dbText(t.db);
      paintRange(fader);
      engine.applyGains();
      drawAll();
    };
    fader.ondblclick = () => { fader.value = "0"; fader.oninput(); };
    t.btnMute.onclick = () => { t.muted = !t.muted; refreshTracks(); };
    t.btnSolo.onclick = (e) => {
      const exclusive = e.altKey || e.metaKey;
      const next = exclusive ? !isAuditioning(t) : !t.soloed;
      if (exclusive) for (const o of S.tracks) o.soloed = false;
      t.soloed = next;
      refreshTracks();
    };
    t.btnPlay.onclick = () => audition(t);
  }
  refreshTracks();
}

const stemTracks = () => S.tracks.filter((t) => t.name !== "mix");
const isAuditioning = (t) => t.soloed && stemTracks().every((o) => o === t || !o.soloed);

/* Per-track ▶: hear just this stem (exclusive solo) from the playhead. */
async function audition(t) {
  if (engine.playing && isAuditioning(t) && engine.mode === "stems") {
    engine.pause();
  } else {
    for (const o of stemTracks()) o.soloed = o === t;
    t.muted = false;
    if (!S.visible.has(t.name)) { S.visible.add(t.name); applyVisibility(); }
    setMode("stems");
    refreshTracks();
    await engine.play();
  }
  updatePlayUI();
}

function refreshTracks() {
  engine.applyGains();
  for (const t of stemTracks()) {
    t.btnMute.classList.toggle("on", t.muted);
    t.btnSolo.classList.toggle("on", t.soloed);
    t.rowEl.classList.toggle("silent", !t.audible);
    t.rowEl.hidden = !t.visible;
  }
  updatePlayUI();
  drawAll();
}

function updatePlayUI() {
  $("playBtn").disabled = !engine.ready;
  $("playBtn").classList.toggle("playing", engine.playing);
  for (const t of stemTracks()) t.btnPlay?.classList.toggle("on", engine.playing && engine.mode === "stems" && isAuditioning(t));
}

async function togglePlay() {
  if (engine.playing) {
    engine.pause();
  } else if (engine.ready) {
    await engine.ctx.resume();
    if (Click.countIn && S.stats?.bpm) {
      const beat = 60 / S.stats.bpm;
      const first = engine.ctx.currentTime + 0.05;
      for (let k = 0; k < 4; k++) Click.blip(first + k * beat, k === 0);
      await engine.play(0.02 + 4 * beat);
    } else {
      await engine.play();
    }
  }
  updatePlayUI();
}

// Top-bar stem chips: show only some stems (hidden ones are silent).
function renderChips() {
  const box = $("stemChips");
  const all = S.visible.size === STEMS.length;
  box.replaceChildren(
    h("span", { class: "chips-label" }, "Show"),
    h("button", { class: `chip all${all ? " on" : ""}`, onclick: () => { S.visible = new Set(STEMS); applyVisibility(); } }, "All"),
    ...STEMS.map((s) => h("button", {
      class: `chip${!all && S.visible.has(s) ? " on" : ""}`, style: `--c:${COLORS[s]}`,
      onclick: () => {
        if (all) S.visible = new Set([s]);
        else if (S.visible.has(s)) S.visible.delete(s);
        else S.visible.add(s);
        if (!S.visible.size) S.visible = new Set(STEMS);
        applyVisibility();
      },
    }, h("i"), LABELS[s])),
  );
  for (const c of box.querySelectorAll(".chip")) c.disabled = $("workspaceView").hidden;
}

function applyVisibility() {
  for (const t of stemTracks()) t.visible = S.visible.has(t.name);
  renderChips();
  refreshTracks();
  layout();
}

function setMode(mode) {
  engine.mode = mode;
  for (const b of $("abSeg").children) b.classList.toggle("on", b.dataset.mode === mode);
  if (S.tracks.length) refreshTracks();
  updateStatus();
}

// ---------------------------------------------------------------- instrument lanes
function buildDetectionRows() {
  const box = $("detectRows");
  box.replaceChildren();
  S.laneRows = S.laneRows.filter((r) => r.kind !== "det");
  const job = S.job;
  const a = S.analysis;
  if (!job) return;
  const label = makeRow(box, "section-label", [h("span", {}, "Detected instruments")], { canvas: false });
  const note = h("div", { class: "section-note" });
  label.lane.append(note);

  if (!a) {
    if (job.analysis_error || job.status === "error") {
      note.append(h("span", { class: "analysis-error" }, `Recognition failed: ${job.analysis_error || job.error}`),
        h("button", { class: "mini-pill", style: "margin:0", onclick: retryJob }, "Retry"));
    } else {
      note.append(h("span", { class: "spinner" }), h("span", {}, job.status === "done" ? "Loading…" : job.stage || "Waiting…"));
    }
    return;
  }
  note.append(
    h("span", { class: "legend" }, "confidence", h("span", { class: "ramp" })),
    h("span", { class: "legend" }, h("span", { class: "segkey" }), "detected as playing"),
    h("span", {}, "· PANNs on the original mix · click a row for individual AudioSet classes"),
  );
  if (!a.lanes.length) {
    makeRow(box, "cls empty-note", [h("span", {}, "Nothing detected")], { canvas: false });
    return;
  }
  for (const lane of a.lanes) {
    const open = S.expanded.has(lane.family);
    const r = makeRow(box, `family${open ? " open" : ""}`, [
      h("span", { class: "chev" }, "▶"),
      h("span", { class: "fam-dot" }),
      h("span", { class: "fam-name" }, lane.family),
      h("span", { class: "fam-peak", title: "Peak confidence" }, pct(lane.peak)),
    ], { style: `--c:${lane.color}` });
    Object.assign(r, {
      kind: "det", draw: drawHeat, values: lane.values, color: lane.color, segments: lane.segments, label: lane.family,
    });
    S.laneRows.push(r);
    r.head.onclick = () => {
      if (S.expanded.has(lane.family)) S.expanded.delete(lane.family);
      else S.expanded.add(lane.family);
      buildDetectionRows();
      layout();
    };
    if (!open) continue;
    for (const cls of lane.classes) {
      const c = makeRow(box, "cls", [h("span", {}, cls.name), h("span", { class: "fam-peak" }, pct(cls.peak))]);
      Object.assign(c, {
        kind: "det", draw: drawHeat, values: cls.values, color: lane.color, segments: null, label: `${cls.label} (AudioSet)`,
      });
      S.laneRows.push(c);
    }
  }
}

async function loadAnalysis(token) {
  const seq = ++S.analysisSeq;
  try {
    const a = await api(`/api/jobs/${S.jobId}/analysis?sensitivity=${S.sensitivity}`);
    if (token !== S.token || seq !== S.analysisSeq) return;
    S.analysis = a;
  } catch (e) {
    if (token === S.token) toast(`Could not load instrument analysis: ${e.message}`);
    return;
  }
  buildDetectionRows();
  layout();
  renderAnalysisPanel();
}

// Event times are labelled to the nearest second; clicking seeks to the exact onset.
function chip(t) {
  return h("span", { class: "tchip", title: "Jump here", onclick: () => seekTo(t, true) }, fmt(Math.round(t)));
}

function renderAnalysisPanel() {
  const a = S.analysis;
  const job = S.job;
  S.eventEls = [];
  S.currentEvent = -1;
  if (!a) {
    const failure = job?.analysis_error || (job?.status === "error" ? job.error : null);
    $("summary").replaceChildren(h("div", { class: "placeholder" }, ...(failure
      ? [h("span", { class: "analysis-error" }, failure)]
      : [h("span", { class: "spinner" }), h("span", {}, "Waiting for instrument recognition…")])));
    $("events").replaceChildren();
    $("method").textContent = "";
    return;
  }
  $("summary").replaceChildren(...(a.summary.length ? a.summary.map((s) => {
    const p = h("p", {}, h("span", { class: "who", style: `--c:${s.color}` }, s.family), " ");
    s.parts.forEach((part, i) => {
      p.append(part.text);
      if (part.time != null) p.append(" ", chip(part.time));
      p.append(i < s.parts.length - 1 ? ", " : ".");
    });
    return p;
  }) : [h("div", { class: "placeholder" }, "No instrument was confidently detected. Try raising the sensitivity.")]));
  $("events").replaceChildren(...(a.events.length ? a.events.map((ev) => {
    const row = h("div", { class: "ev", style: `--c:${ev.color}`, onclick: () => seekTo(ev.time, true) },
      h("span", { class: "time" }, fmt(Math.round(ev.time))),
      h("span", { class: "icon" }, EVENT_ICONS[ev.kind] || "•"),
      h("span", { class: "what" }, ev.text, ev.detail ? h("span", { class: "detail" }, ev.detail) : null));
    S.eventEls.push({ time: ev.time, el: row });
    return row;
  }) : [h("div", { class: "placeholder" }, "No events.")]));
  const th = a.thresholds;
  $("method").textContent =
    `Playing = PANNs confidence reaches ${pct(th.on)} and stays above ${pct(th.off)} ` +
    `(gaps under 2.5 s bridged, blips under 2 s ignored). Louder / quieter = a ${th.db.toFixed(0)} dB+ step ` +
    "in the matching Demucs stem, comparing the 4 s before and after. Confidence is the model's per-frame " +
    "probability averaged over 0.25 s, not a calibrated likelihood.";
  drawAll();
}

// ---------------------------------------------------------------- sections
function sectionAt(t) {
  const i = lowerBound(S.sections, t + 1e-6, (s) => s.start) - 1;
  if (i < 0) return null;
  return { i, s: S.sections[i], end: S.sections[i + 1]?.start ?? S.duration };
}

function sectionColor(name) {
  const key = name.toLowerCase().replace(/\s*\d+$/, "");
  if (SECTION_COLORS[key]) return SECTION_COLORS[key];
  const palette = ["#6ea8fe", "#62c972", "#a98bfa", "#ec5f74", "#f08a3e", "#8fa5cc"];
  let hash = 0;
  for (const ch of key) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return palette[hash % palette.length];
}

async function saveSections() {
  setSaved("saving");
  try {
    S.sections = await api(`/api/jobs/${S.jobId}/sections`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S.sections),
    });
    setSaved("saved");
  } catch (e) {
    setSaved("");
    toast(`Could not save sections: ${e.message}`);
  }
  drawAll();
}

function addSection() {
  if (!S.duration) return;
  const t = engine.position;
  if (S.sections.some((s) => Math.abs(s.start - t) < 0.5)) {
    toast("A section already starts here. Move the playhead first.");
    return;
  }
  const suggestion = SECTION_SUGGESTIONS[Math.min(lowerBound(S.sections, t, (s) => s.start), SECTION_SUGGESTIONS.length - 1)];
  const name = prompt(`Name the section starting at ${mmssff(t)}`, suggestion);
  if (name == null) return;
  S.sections.push({ name: name.trim() || "Section", start: Math.round(t * 1000) / 1000 });
  S.sections.sort((a, b) => a.start - b.start);
  saveSections();
}

function onSectionClick(e) {
  const hit = sectionAt(timeAt(e.clientX));
  if (!hit) return;
  if (e.altKey) {
    setLoop(hit.s.start, hit.end, true);
  } else {
    seekTo(hit.s.start, true);
  }
}

function onSectionRename(e) {
  const hit = sectionAt(timeAt(e.clientX));
  if (!hit) return;
  const name = prompt("Rename section", hit.s.name);
  if (name == null || !name.trim()) return;
  hit.s.name = name.trim();
  saveSections();
}

function onSectionDelete(e) {
  const hit = sectionAt(timeAt(e.clientX));
  if (!hit) return;
  e.preventDefault();
  if (!confirm(`Delete section “${hit.s.name}”?`)) return;
  S.sections.splice(hit.i, 1);
  saveSections();
}

// ---------------------------------------------------------------- loop
function setLoop(a, b, enable) {
  a = clamp(a, 0, S.duration);
  b = clamp(b, 0, S.duration);
  if (b - a < 0.25) return;
  engine.loop = { a, b };
  if (enable) engine.loopOn = true;
  if (engine.playing && engine.loopOn && (engine.position < a || engine.position >= b)) engine.seek(a);
  else engine.restart();
  renderLoopUI();
}

function toggleLoop() {
  if (!S.duration) return;
  if (!engine.loop) {
    const pos = engine.position;
    const sec = sectionAt(pos);
    const beats = S.stats?.beats || [];
    if (sec) {
      engine.loop = { a: sec.s.start, b: sec.end };
    } else if (beats.length > 16) {
      const i = clamp(lowerBound(beats, pos), 0, beats.length - 17);
      engine.loop = { a: beats[i], b: beats[i + 16] }; // four bars from the playhead
    } else {
      engine.loop = { a: pos, b: Math.min(S.duration, pos + 8) };
    }
  }
  engine.loopOn = !engine.loopOn;
  if (engine.loopOn && engine.playing && (engine.position < engine.loop.a || engine.position >= engine.loop.b)) {
    engine.seek(engine.loop.a);
  } else {
    engine.restart();
  }
  renderLoopUI();
}

function renderLoopUI() {
  $("loopBtn").classList.toggle("on", engine.loopOn);
  const L = engine.loop;
  $("loopStart").textContent = L ? mmssff(L.a) : "--:--.--";
  $("loopEnd").textContent = L ? mmssff(L.b) : "--:--.--";
  $("loopStart").classList.toggle("set", !!L);
  $("loopEnd").classList.toggle("set", !!L);
  updateStatus();
  drawAll();
}

// ---------------------------------------------------------------- transport bar
function renderNowPlaying() {
  const job = S.job;
  if (!job) return;
  $("npCover").replaceWith(Object.assign(thumb(job, "cover thumb"), { id: "npCover" }));
  $("npTitle").textContent = titleOf(job);
  $("npTitle").title = job.filename;
  $("npMeta").replaceChildren(
    h("span", {}, mmss(job.duration)), h("span", {}, h("span", { class: "dot" }), " 6 stems"),
    h("span", {}, job.format), job.key ? h("span", {}, job.key) : null,
  );
  $("npDate").textContent = `Processed ${new Date(job.created * 1000).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  })}`;
  $("favBtn").classList.toggle("on", !!job.favorite);
  $("favBtn").title = job.favorite ? "Remove from favorites" : "Add to favorites";
}

function renderTransportControls() {
  const hasBeats = (S.stats?.beats?.length || 0) >= 8;
  for (const id of ["clickBtn", "countBtn", "gridBtn"]) $(id).disabled = !hasBeats;
  for (const b of $("clickRate").children) b.disabled = !hasBeats;
  $("clickBtn").classList.toggle("on", Click.on);
  $("clickBtn").querySelector("span").textContent = Click.on ? "On" : "Off";
  $("countBtn").classList.toggle("on", Click.countIn);
  $("gridBtn").classList.toggle("on", S.gridOn);
  for (const b of $("clickRate").children) b.classList.toggle("on", Number(b.dataset.rate) === Click.rate);
  renderLoopUI();
}

let statusOverride = null;
function setStatus(text) {
  statusOverride = text;
  $("tpStatus").textContent = text;
}

function updateStatus() {
  if (!engine.ready && statusOverride) return;
  statusOverride = null;
  const st = S.stats;
  const b = (t) => h("b", {}, t);
  let parts;
  if (Click.on && st?.bpm) {
    parts = ["Click at ", b(`${st.bpm} BPM`), ` on ${st.beats.length} beats detected in the drum stem · the accented beat 1 is a guess · use /2 or ×2 if it feels half or double time`];
  } else if (engine.loopOn && engine.loop) {
    parts = ["Looping ", b(mmssff(engine.loop.a)), " – ", b(mmssff(engine.loop.b)), " · drag across the time ruler to change it · L to stop"];
  } else if (engine.mode === "original") {
    parts = ["Playing the ", b("original file"), " · switch back to Stems to use the mixer"];
  } else {
    parts = ["Drag across the time ruler to set a loop · ⌥-click a section to loop it · ⌘-scroll to zoom · Space to play"];
  }
  $("tpStatus").replaceChildren(...parts);
}

async function toggleFavorite() {
  if (!S.job) return;
  try {
    const job = await api(`/api/jobs/${S.jobId}/favorite`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ favorite: !S.job.favorite }),
    });
    S.job = job;
    renderNowPlaying();
    refreshLibrary();
  } catch (e) {
    toast(e.message);
  }
}

function mixDescription() {
  if (engine.mode === "original") return "original";
  const on = stemTracks().filter((t) => t.audible).map((t) => t.name);
  if (on.length === STEMS.length) return "full mix";
  if (on.length >= 4) return `no ${STEMS.filter((s) => !on.includes(s)).join(" + ")}`;
  return on.join(" + ");
}

function encodeWav(buf) {
  const ch = buf.numberOfChannels;
  const len = buf.length;
  const bytes = len * ch * 2;
  const out = new ArrayBuffer(44 + bytes);
  const view = new DataView(out);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); view.setUint32(4, 36 + bytes, true); str(8, "WAVE"); str(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, ch, true);
  view.setUint32(24, buf.sampleRate, true); view.setUint32(28, buf.sampleRate * ch * 2, true);
  view.setUint16(32, ch * 2, true); view.setUint16(34, 16, true); str(36, "data"); view.setUint32(40, bytes, true);
  const pcm = new Int16Array(out, 44); // WAV is little-endian, as are Macs
  const chans = Array.from({ length: ch }, (_, i) => buf.getChannelData(i));
  for (let i = 0, o = 0; i < len; i++) {
    for (let c = 0; c < ch; c++, o++) {
      const s = Math.max(-1, Math.min(1, chans[c][i]));
      pcm[o] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }
  return new Blob([out], { type: "audio/wav" });
}

async function exportMix() {
  if (!engine.ready) { toast("Audio is still loading"); return; }
  const first = S.tracks[0].buffer;
  const off = new OfflineAudioContext(2, first.length, first.sampleRate);
  let any = false;
  for (const t of S.tracks) {
    const v = t.audible ? (t.name === "mix" ? 1 : dbToGain(t.db)) : 0;
    if (!v) continue;
    any = true;
    const src = off.createBufferSource();
    src.buffer = t.buffer;
    const g = off.createGain();
    g.gain.value = v;
    src.connect(g).connect(off.destination);
    src.start(0);
  }
  if (!any) { toast("Nothing is audible. Unmute a stem first."); return; }
  const desc = mixDescription();
  toast(`Rendering “${desc}”…`, true);
  const url = URL.createObjectURL(encodeWav(await off.startRendering()));
  downloadUrl(url, `${titleOf(S.job)} - ${desc}.wav`);
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  toast(`Exported “${desc}” to your Downloads`, true);
}

// ---------------------------------------------------------------- layout & drawing
function layout() {
  if (!S.duration || $("workspaceView").hidden) return;
  S.hw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--head-w")) || 304;
  S.vw = Math.max(50, scroller.clientWidth - S.hw);
  const fitPps = S.vw / S.duration;
  S.maxZoom = Math.max(1, MAX_PPS / fitPps);
  S.zoom = clamp(S.zoom, 1, S.maxZoom);
  S.pps = fitPps * S.zoom;
  $("rows").style.setProperty("--lane-w", `${Math.max(S.vw, Math.ceil(S.duration * S.pps))}px`);
  const dpr = DPR();
  for (const r of S.laneRows) {
    if (!r.canvas) continue;
    r.canvas.width = Math.round(S.vw * dpr);
    r.canvas.height = Math.round(r.lane.clientHeight * dpr);
    r.canvas.style.width = `${S.vw}px`;
  }
  const bar = `${scroller.offsetHeight - scroller.clientHeight}px`;
  $("playhead").style.bottom = bar;
  $("loopShade").style.bottom = bar;
  const z = $("zoomSlider");
  z.value = S.maxZoom > 1 ? Math.round((100 * Math.log(S.zoom)) / Math.log(S.maxZoom)) : 0;
  paintRange(z);
  buildOverviewCache();
  drawAll();
}

let drawQueued = false;
function drawAll() {
  for (const r of S.laneRows) if (!r.row.hidden) r.draw(r);
}
function queueDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; drawAll(); });
}

function setZoom(z, anchorClientX) {
  if (!S.duration) return;
  const left = scroller.getBoundingClientRect().left + S.hw;
  let ax;
  if (anchorClientX != null) {
    ax = clamp(anchorClientX - left, 0, S.vw);
  } else {
    const px = engine.position * S.pps - scroller.scrollLeft;
    ax = px >= 0 && px <= S.vw ? px : S.vw / 2;
  }
  const t = (scroller.scrollLeft + ax) / S.pps;
  S.zoom = clamp(z, 1, S.maxZoom);
  layout();
  scroller.scrollLeft = t * S.pps - ax;
  drawAll();
}

function rulerStep(pps) {
  const steps = [[1, 4], [2, 4], [5, 5], [10, 5], [15, 3], [30, 3], [60, 4], [120, 4], [300, 5], [600, 5]];
  return steps.find(([s]) => s * pps >= 70) || steps[steps.length - 1];
}

function drawGrid(ctx, W, H) {
  const dpr = DPR();
  const sl = scroller.scrollLeft;
  const pps = S.pps;
  const [step] = rulerStep(pps);
  ctx.fillStyle = "#15181e";
  for (let k = Math.floor(sl / pps / step); (k * step * pps - sl) * dpr <= W; k++) {
    ctx.fillRect(Math.round((k * step * pps - sl) * dpr), 0, dpr, H);
  }
  const st = S.stats;
  if (!S.gridOn || !st?.beats?.length) return;
  const B = st.beats;
  const d = st.downbeat || 0;
  const beatPx = (60 / (st.bpm || 120)) * pps;
  const barEvery = beatPx * 4 >= 10 ? 4 : 16; // too dense? draw every 4th bar instead
  for (let i = lowerBound(B, sl / pps); i < B.length && (B[i] * pps - sl) * dpr <= W; i++) {
    const k = mod(i - d, barEvery);
    if (k !== 0 && beatPx < 8) continue;
    ctx.fillStyle = k === 0 ? "rgba(245,179,29,0.22)" : "rgba(255,255,255,0.05)";
    ctx.fillRect(Math.round((B[i] * pps - sl) * dpr), 0, dpr, H);
  }
}

function drawRuler(r) {
  const c = r.canvas;
  const ctx = c.getContext("2d");
  const dpr = DPR();
  const W = c.width;
  const H = c.height;
  ctx.clearRect(0, 0, W, H);
  const pps = S.pps;
  const sl = scroller.scrollLeft;
  const [step, div] = rulerStep(pps);
  const minor = step / div;
  ctx.font = `500 ${10.5 * dpr}px ${MONO}`;
  ctx.textBaseline = "middle";
  ctx.strokeStyle = "#323743";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  for (let k = Math.floor(sl / pps / minor); (k * minor * pps - sl) * dpr <= W + 40 * dpr; k++) {
    const t = k * minor;
    const x = Math.round((t * pps - sl) * dpr) + 0.5;
    const major = k % div === 0;
    ctx.moveTo(x, H);
    ctx.lineTo(x, H - (major ? 9 : 4) * dpr);
    if (major) {
      ctx.fillStyle = "#8b909c";
      ctx.fillText(fmt(t), x + 5 * dpr, H / 2 - 2 * dpr);
    }
  }
  ctx.stroke();
  const L = engine.loop;
  if (L) {
    ctx.fillStyle = engine.loopOn ? "#f5b31d" : "#5a5f6b";
    const x0 = (L.a * pps - sl) * dpr;
    ctx.fillRect(x0, H - 3 * dpr, (L.b - L.a) * pps * dpr, 3 * dpr);
  }
  for (const ev of S.analysis?.events || []) {
    const x = (ev.time * pps - sl) * dpr;
    if (x < -5 || x > W + 5) continue;
    ctx.fillStyle = ev.color;
    ctx.beginPath();
    ctx.moveTo(x - 3.5 * dpr, H - 9 * dpr);
    ctx.lineTo(x + 3.5 * dpr, H - 9 * dpr);
    ctx.lineTo(x, H - 3.5 * dpr);
    ctx.fill();
  }
}

function drawSections(r) {
  const c = r.canvas;
  const ctx = c.getContext("2d");
  const dpr = DPR();
  const W = c.width;
  const H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.textBaseline = "middle";
  if (!S.sections.length) {
    ctx.fillStyle = "#4a4f5a";
    ctx.font = `${11.5 * dpr}px ${SANS}`;
    ctx.fillText("No sections yet. Move the playhead and press + Add to mark the intro, verses, chorus…", 12 * dpr, H / 2);
    return;
  }
  const sl = scroller.scrollLeft;
  const pps = S.pps;
  ctx.font = `650 ${11 * dpr}px ${SANS}`;
  S.sections.forEach((s, i) => {
    const end = S.sections[i + 1]?.start ?? S.duration;
    const x0 = (s.start * pps - sl) * dpr;
    const x1 = (end * pps - sl) * dpr;
    if (x1 < 0 || x0 > W) return;
    const color = sectionColor(s.name);
    const top = 5 * dpr;
    const hh = H - 10 * dpr;
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x0 + dpr, top, Math.max(2 * dpr, x1 - x0 - 2 * dpr), hh, 4 * dpr);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillRect(x0 + dpr, top, 2 * dpr, hh);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, 0, x1 - x0 - 4 * dpr, H);
    ctx.clip();
    ctx.fillText(s.name, Math.max(x0, 0) + 9 * dpr, H / 2);
    ctx.restore();
  });
}

function drawWave(r) {
  const t = r.track;
  const c = r.canvas;
  const ctx = c.getContext("2d");
  const dpr = DPR();
  const W = c.width;
  const H = c.height;
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, W, H);
  const data = S.peaks?.tracks[t.name];
  if (!data) return;
  const per = S.peaks.per_second;
  const n = data.rms.length;
  const sl = scroller.scrollLeft;
  const pps = S.pps;
  const slot = 3; // css px per bar: 2 px bar + 1 px gap
  const mid = Math.round(H / 2);
  const maxH = H / 2 - 5 * dpr;
  const dim = !t.audible;
  ctx.fillStyle = t.color;
  // Bars are anchored to absolute lane positions so they don't shimmer while scrolling.
  for (let s = Math.floor(sl / slot); ; s++) {
    const xCss = s * slot - sl;
    if (xCss * dpr > W) break;
    const i0 = Math.floor(((s * slot) / pps) * per);
    if (i0 >= n) break;
    const i1 = Math.max(i0 + 1, Math.floor((((s + 1) * slot) / pps) * per));
    let v = 0;
    for (let i = i0; i < i1 && i < n; i++) {
      const x = data.rms[i] * 0.75 + data.peak[i] * 0.25;
      if (x > v) v = x;
    }
    const hgt = Math.min(maxH, (v / 255) * maxH * 1.1);
    const x = Math.round(xCss * dpr);
    if (hgt < 1.5 * dpr) {
      if (s % 2 === 0) {
        ctx.globalAlpha = dim ? 0.2 : 0.55;
        ctx.fillRect(x, mid - dpr / 2, dpr, dpr);
      }
      continue;
    }
    ctx.globalAlpha = dim ? 0.18 : 1;
    ctx.fillRect(x, mid - hgt, 2 * dpr, hgt * 2);
  }
  ctx.globalAlpha = 1;
}

function drawHeat(r) {
  const c = r.canvas;
  const ctx = c.getContext("2d");
  const dpr = DPR();
  const W = c.width;
  const H = c.height;
  ctx.clearRect(0, 0, W, H);
  const vals = r.values;
  const n = vals.length;
  const hop = S.analysis.hop;
  const sl = scroller.scrollLeft;
  const pps = S.pps;
  const segH = r.segments ? 5 * dpr : 0;
  const top = 3 * dpr;
  const hh = H - top - Math.max(top, segH);
  ctx.fillStyle = r.color;
  // Max confidence per device-pixel column, drawn as runs of equal value.
  let runStart = 0;
  let runVal = -1;
  for (let x = 0; x <= W; x++) {
    let v = -1;
    if (x < W) {
      const i0 = Math.floor((sl + x / dpr) / pps / hop);
      const i1 = Math.max(i0, Math.ceil((sl + (x + 1) / dpr) / pps / hop) - 1);
      if (i0 < n) {
        v = 0;
        for (let i = i0; i <= i1 && i < n; i++) if (vals[i] > v) v = vals[i];
      }
    }
    if (v !== runVal) {
      if (runVal >= 2) {
        ctx.globalAlpha = Math.min(1, (runVal / 70) ** 0.8);
        ctx.fillRect(runStart, top, x - runStart, hh);
      }
      runStart = x;
      runVal = v;
    }
  }
  ctx.globalAlpha = 1;
  if (!r.segments) return;
  for (const [s, e] of r.segments) {
    const x0 = (s * pps - sl) * dpr;
    const x1 = (e * pps - sl) * dpr;
    if (x1 < 0 || x0 > W) continue;
    ctx.beginPath();
    ctx.roundRect(x0, H - segH + dpr, Math.max(2 * dpr, x1 - x0), 3 * dpr, 1.5 * dpr);
    ctx.fill();
  }
}

// Whole-song overview under the transport: cached waveform plus live overlays.
let overviewCache = null;
function buildOverviewCache() {
  const c = $("overview");
  const dpr = DPR();
  const W = Math.round(c.clientWidth * dpr);
  const H = Math.round(c.clientHeight * dpr);
  overviewCache = null;
  if (!W || !H || !S.peaks || !S.duration) return;
  c.width = W;
  c.height = H;
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const ctx = off.getContext("2d");
  const pps = W / dpr / S.duration;
  const [step] = rulerStep(pps);
  ctx.font = `500 ${9 * dpr}px ${MONO}`;
  ctx.textBaseline = "top";
  for (let t = 0; t <= S.duration; t += step) {
    const x = Math.round(t * pps * dpr);
    ctx.fillStyle = "#1e2229";
    ctx.fillRect(x, 0, dpr, H);
    ctx.fillStyle = "#4a4f5a";
    ctx.fillText(fmt(t), x + 4 * dpr, 3 * dpr);
  }
  const data = S.peaks.tracks.mix;
  const per = S.peaks.per_second;
  const mid = H * 0.62;
  const maxH = H * 0.32;
  ctx.fillStyle = "#3a3f4a";
  for (let x = 0; x < W; x += 2 * dpr) {
    const i0 = Math.floor((x / W) * S.duration * per);
    const i1 = Math.max(i0 + 1, Math.floor(((x + 2 * dpr) / W) * S.duration * per));
    let v = 0;
    for (let i = i0; i < i1 && i < data.rms.length; i++) v = Math.max(v, data.rms[i]);
    const hh = Math.max(dpr / 2, (v / 255) * maxH);
    ctx.fillRect(x, mid - hh, dpr, hh * 2);
  }
  overviewCache = off;
}

function drawOverview(pos) {
  if (!overviewCache) return;
  const c = $("overview");
  const ctx = c.getContext("2d");
  const W = c.width;
  const H = c.height;
  const dpr = DPR();
  const X = (t) => (t / S.duration) * W;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(overviewCache, 0, 0);
  ctx.fillStyle = "rgba(245,179,29,0.05)";
  ctx.fillRect(0, 0, X(pos), H);
  S.sections.forEach((s, i) => {
    const end = S.sections[i + 1]?.start ?? S.duration;
    ctx.fillStyle = sectionColor(s.name);
    ctx.fillRect(X(s.start), H - 3 * dpr, Math.max(dpr, X(end) - X(s.start) - dpr), 2 * dpr);
  });
  if (engine.loop) {
    ctx.fillStyle = engine.loopOn ? "rgba(245,179,29,0.16)" : "rgba(255,255,255,0.05)";
    ctx.fillRect(X(engine.loop.a), 0, X(engine.loop.b) - X(engine.loop.a), H);
  }
  if (S.zoom > 1.01) {
    const a = scroller.scrollLeft / S.pps;
    const b = a + S.vw / S.pps;
    ctx.fillStyle = "rgba(255,255,255,0.05)";
    ctx.fillRect(X(a), 0, X(b) - X(a), H);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = dpr;
    ctx.strokeRect(X(a) + dpr / 2, dpr / 2, X(b) - X(a) - dpr, H - dpr);
  }
  ctx.fillStyle = "#ff5d6c";
  ctx.fillRect(X(pos) - dpr / 2, 0, 1.5 * dpr, H);
}

// ---------------------------------------------------------------- animation loop
let lastClock = "";
function tick() {
  requestAnimationFrame(tick);
  if (!S.duration || $("workspaceView").hidden) return;
  let pos = engine.position;
  if (engine.playing && !engine.activeLoop && pos >= S.duration - 0.005) {
    engine.pause();
    engine.offset = S.duration;
    pos = S.duration;
    updatePlayUI();
  }
  if (engine.playing && !S.drag) {
    const px = pos * S.pps - scroller.scrollLeft;
    if (px > S.vw - 24 || px < 0) scroller.scrollLeft = pos * S.pps - S.vw * 0.1;
  }
  const sl = scroller.scrollLeft;
  const x = S.hw + pos * S.pps - sl;
  const ph = $("playhead");
  ph.style.transform = `translateX(${x}px)`;
  ph.hidden = x < S.hw - 1 || x > S.hw + S.vw + 1;

  const shade = $("loopShade");
  const L = engine.loop;
  if (L) {
    const x0 = Math.max(S.hw, S.hw + L.a * S.pps - sl);
    const x1 = Math.min(S.hw + S.vw, S.hw + L.b * S.pps - sl);
    shade.hidden = x1 <= x0;
    shade.style.transform = `translateX(${x0}px)`;
    shade.style.width = `${Math.max(0, x1 - x0)}px`;
    shade.classList.toggle("off", !engine.loopOn);
  } else {
    shade.hidden = true;
  }

  const clock = mmss(pos);
  if (clock !== lastClock) { $("curTime").textContent = clock; lastClock = clock; }

  let idx = -1;
  for (let i = 0; i < S.eventEls.length && S.eventEls[i].time <= pos + 0.05; i++) idx = i;
  if (idx !== S.currentEvent) {
    S.eventEls[S.currentEvent]?.el.classList.remove("current");
    S.eventEls[idx]?.el.classList.add("current");
    S.currentEvent = idx;
  }
  drawOverview(pos);
}

function seekTo(t, reveal) {
  if (engine.loopOn && engine.loop && (t < engine.loop.a || t >= engine.loop.b)) {
    engine.loopOn = false; // jumping outside the loop ends it
    renderLoopUI();
  }
  engine.seek(t);
  if (reveal) {
    const x = t * S.pps - scroller.scrollLeft;
    if (x < 0 || x > S.vw) scroller.scrollLeft = t * S.pps - S.vw * 0.25;
  }
  updatePlayUI();
}

// ---------------------------------------------------------------- interaction
function timeAt(clientX) {
  const left = scroller.getBoundingClientRect().left + S.hw;
  return clamp((clientX - left + scroller.scrollLeft) / S.pps, 0, S.duration);
}

scroller.addEventListener("scroll", queueDraw);

scroller.addEventListener("pointerdown", (e) => {
  const lane = e.target.closest(".lane");
  if (!lane || e.button !== 0) return;
  const row = lane.closest(".row");
  if (row.matches(".section-label, .empty-note, .sections")) return;
  e.preventDefault();
  const t = timeAt(e.clientX);
  S.drag = { ruler: row.matches(".ruler"), x: e.clientX, t0: t, moved: false, wasPlaying: engine.playing };
  if (!S.drag.ruler) {
    engine.pause();
    engine.offset = t;
  }
  scroller.setPointerCapture(e.pointerId);
});
scroller.addEventListener("pointermove", (e) => {
  const d = S.drag;
  if (d) {
    const t = timeAt(e.clientX);
    if (!d.ruler) {
      engine.offset = t;
    } else if (Math.abs(e.clientX - d.x) > 4) {
      d.moved = true;
      engine.loop = { a: Math.min(d.t0, t), b: Math.max(d.t0, t) };
      renderLoopUI();
    }
  }
  showTooltip(e);
});
const endDrag = async () => {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  if (d.ruler) {
    if (d.moved && engine.loop) setLoop(engine.loop.a, engine.loop.b, true);
    else seekTo(d.t0);
  } else if (d.wasPlaying) {
    if (engine.loopOn && engine.loop && (engine.offset < engine.loop.a || engine.offset >= engine.loop.b)) {
      engine.loopOn = false;
      renderLoopUI();
    }
    await engine.play();
  }
  updatePlayUI();
};
scroller.addEventListener("pointerup", endDrag);
scroller.addEventListener("pointercancel", endDrag);
scroller.addEventListener("pointerleave", () => { $("tooltip").hidden = true; });

scroller.addEventListener("wheel", (e) => {
  if (!(e.ctrlKey || e.metaKey) || !S.duration) return;
  e.preventDefault(); // pinch-to-zoom on a trackpad arrives as ctrl+wheel
  setZoom(S.zoom * Math.exp(-e.deltaY * 0.01), e.clientX);
}, { passive: false });

function showTooltip(e) {
  const tip = $("tooltip");
  const row = e.target.closest(".row");
  const r = row?._r;
  if (!r || !e.target.closest(".lane") || S.drag) {
    tip.hidden = true;
    return;
  }
  const t = timeAt(e.clientX);
  let content = null;
  if (r.kind === "det" && S.analysis) {
    const v = r.values[Math.min(r.values.length - 1, Math.floor(t / S.analysis.hop))] ?? 0;
    const playing = r.segments?.some(([s, en]) => t >= s && t < en);
    content = [h("b", {}, r.label), " ", h("span", { class: "t" }, `${fmt(t)} · ${v}%${playing ? " · playing" : ""}`)];
  } else if (r.kind === "sections") {
    const hit = sectionAt(t);
    if (hit) {
      content = [h("b", {}, hit.s.name), " ", h("span", { class: "t" },
        `${fmt(hit.s.start)}–${fmt(hit.end)} · click jump · ⌥-click loop · double-click rename · right-click delete`)];
    }
  } else if (r.kind === "ruler") {
    content = [h("span", { class: "t" }, `${mmssff(t)} · click to jump, drag to loop`)];
  }
  if (!content) {
    tip.hidden = true;
    return;
  }
  tip.replaceChildren(...content);
  const wrap = $("timelineWrap").getBoundingClientRect();
  const rr = row.getBoundingClientRect();
  tip.style.left = `${clamp(e.clientX - wrap.left, 140, wrap.width - 140)}px`;
  tip.style.top = `${rr.top - wrap.top - 4}px`;
  tip.hidden = false;
}

// Overview: click or drag to jump; the main view follows.
const overview = $("overview");
let overviewDrag = false;
function overviewSeek(e) {
  const rect = overview.getBoundingClientRect();
  const t = clamp(((e.clientX - rect.left) / rect.width) * S.duration, 0, S.duration);
  seekTo(t);
  if (S.zoom > 1.01) scroller.scrollLeft = t * S.pps - S.vw / 2;
}
overview.addEventListener("pointerdown", (e) => {
  if (!S.duration) return;
  overviewDrag = true;
  overview.setPointerCapture(e.pointerId);
  overviewSeek(e);
});
overview.addEventListener("pointermove", (e) => { if (overviewDrag) overviewSeek(e); });
overview.addEventListener("pointerup", () => { overviewDrag = false; });

// Transport controls
$("playBtn").onclick = togglePlay;
$("stopBtn").onclick = () => { engine.pause(); seekTo(engine.loopOn && engine.loop ? engine.loop.a : 0, true); };
$("loopBtn").onclick = toggleLoop;
$("zoomIn").onclick = () => setZoom(S.zoom * 1.6);
$("zoomOut").onclick = () => setZoom(S.zoom / 1.6);
$("zoomFit").onclick = () => setZoom(1);
$("zoomSlider").oninput = (e) => setZoom(S.maxZoom ** (Number(e.target.value) / 100));
$("masterVol").oninput = (e) => {
  paintRange(e.target);
  if (engine.master) engine.master.gain.setTargetAtTime(Number(e.target.value), engine.ctx.currentTime, 0.02);
};
for (const b of $("abSeg").children) b.onclick = () => setMode(b.dataset.mode);

function toggleClick() {
  if ($("clickBtn").disabled) return;
  Click.on = !Click.on;
  Click.idx = -1;
  renderTransportControls();
}
$("clickBtn").onclick = toggleClick;
$("countBtn").onclick = () => { Click.countIn = !Click.countIn; renderTransportControls(); };
$("gridBtn").onclick = () => { S.gridOn = !S.gridOn; renderTransportControls(); };
for (const b of $("clickRate").children) {
  b.onclick = () => { Click.rate = Number(b.dataset.rate); Click.rebuild(); renderTransportControls(); };
}
$("clickVol").oninput = (e) => {
  Click.vol = Number(e.target.value);
  $("clickVolOut").textContent = pct(Click.vol);
  paintRange(e.target);
};
setInterval(() => Click.tick(), 25);

$("favBtn").onclick = toggleFavorite;
$("exportBtn").onclick = (e) => { e.stopPropagation(); $("exportMenu").hidden = !$("exportMenu").hidden; };
for (const b of $("exportMenu").querySelectorAll("button")) {
  b.onclick = () => {
    $("exportMenu").hidden = true;
    if (b.dataset.export === "mix") exportMix();
    else downloadUrl(`/api/jobs/${S.jobId}/stems.zip`);
  };
}

let sensTimer;
$("sensitivity").oninput = (e) => {
  $("sensOut").value = e.target.value;
  paintRange(e.target);
  S.sensitivity = Number(e.target.value) / 100;
  clearTimeout(sensTimer);
  sensTimer = setTimeout(() => { if (S.analysis) loadAnalysis(S.token); }, 120);
};

// Sidebar
$("sidebarToggle").onclick = () => { toggleSidebar(); requestAnimationFrame(layout); };
for (const b of document.querySelectorAll("[data-libview]")) b.onclick = () => setLibView(b.dataset.libview);
$("search").oninput = (e) => { S.search = e.target.value; renderLibrary(); };
$("clearTag").onclick = () => { S.tagFilter = null; renderLibrary(); };
$("helpBtn").onclick = (e) => { e.stopPropagation(); $("helpPop").hidden = !$("helpPop").hidden; };
$("homeLink").onclick = (e) => { e.preventDefault(); goHome(); };

document.addEventListener("click", (e) => {
  if (!e.target.closest(".export")) $("exportMenu").hidden = true;
  if (!e.target.closest("#helpPop, #helpBtn")) $("helpPop").hidden = true;
  if (narrow() && !e.target.closest(".sidebar, #sidebarToggle, [data-libview]")) $("app").classList.remove("side-open");
  // Keep Space from also "clicking" whichever button was clicked last.
  e.target.closest("button")?.blur();
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    showSidebar();
    $("search").focus();
    return;
  }
  if (e.key === "Escape") {
    $("exportMenu").hidden = true;
    $("helpPop").hidden = true;
    $("app").classList.remove("side-open");
    if (document.activeElement === $("search")) $("search").blur();
    return;
  }
  if (e.target.matches?.("input[type=search], input[type=text]")) return;
  if ($("workspaceView").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  const inRange = e.target.matches?.("input[type=range]");
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (!inRange && (e.code === "ArrowLeft" || e.code === "ArrowRight")) {
    e.preventDefault();
    seekTo(engine.position + (e.shiftKey ? 1 : 5) * (e.code === "ArrowLeft" ? -1 : 1), true);
  } else if (e.code === "Home") {
    seekTo(0, true);
  } else if (e.code === "KeyL") {
    toggleLoop();
  } else if (e.code === "KeyC") {
    toggleClick();
  }
});

for (const input of document.querySelectorAll(".file-input")) {
  input.onchange = () => { uploadFile(input.files[0]); input.value = ""; };
}

// Drag & drop anywhere on the page.
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  dragDepth++;
  $("dropOverlay").hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("dropOverlay").hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("dropOverlay").hidden = true;
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

// Re-layout whenever the timeline's width changes (window resize, sidebar toggle).
let resizeQueued = false;
new ResizeObserver(() => {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => { resizeQueued = false; layout(); });
}).observe($("main"));

window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (id && id !== S.jobId) openJob(id);
  else if (!id && S.jobId) goHome();
});

// ---------------------------------------------------------------- boot
for (const id of ["masterVol", "clickVol", "sensitivity", "zoomSlider"]) paintRange($(id));
renderChips();
refreshLibrary();
if (location.hash.length > 1) openJob(location.hash.slice(1));
else showView("welcome");
requestAnimationFrame(tick);
