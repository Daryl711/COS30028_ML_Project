// app.js - Plant disease identifier that runs fully in the browser.
//
// The models live in model/ as ONNX files, run with ONNX Runtime Web, so no
// server is needed and the photo never leaves the user's device.
// model/config.json lists the deployed models. While that list is empty the
// page still works (upload, samples, benchmarks, about) and says no model is
// deployed yet. Add ?demo to the URL to fill the results with random scores
// and check the layout.
//
// Flow: init() loads data/classes.json and model/config.json -> user picks an
//       image -> showImage() -> analyse() -> runModel() (preprocess + session.run)
//       -> scoreCompositions() -> similarities() -> renderPrediction()
//       -> renderGallery() / renderCompare().
//
// Optional model output (see model/README.md):
//   outputs.embedding  [1, D] image embedding + gallery_embeddings file   -> similarity ranking

// ---- Settings ---------------------------------------------------------------
const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const TOP_K = 5;
const GALLERY_MAX = 12;          // thumbnails per gallery tab
const ZOOM_MAX = 8;
const DEMO = new URLSearchParams(location.search).has("demo");
const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

let classes;            // data/classes.json: crops, diseases, compositions, examples
let galleryItems = [];  // every gallery thumbnail, flattened in composition order: { src, comp }
let models = [];        // model/config.json -> models
const entries = new Map();  // model id -> Promise<{ cfg, session, gallery }>
let current = null;     // entry of the selected model
let lastImg = null;     // last image element, re-analysed when the model changes
let lastTruth = null;   // { truth, split } of a sample image, or null for uploads
let imgToken = 0;       // increases with every new image, so stale results are dropped
let results = new Map();    // model id -> result for the current image
let primary = null;     // result of the selected model for the current image
let compareOn = false;
let galleryTab = null;  // last gallery tab the user chose
let bench = null;       // data/benchmarks.json, loaded on first open
let uploadUrl = null;   // object URL of the last upload, revoked on the next one

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const modelLabel = (cfg) => (cfg.role ? `${cfg.role} · ${cfg.name}` : cfg.name);

// ---- Start-up -----------------------------------------------------------------
async function init() {
  classes = await fetchJSON("data/classes.json");
  classes.compositions.forEach((c) => c.thumbs.forEach((src) => galleryItems.push({ src, comp: c.id })));
  buildExamples();

  try {
    models = (await fetchJSON("model/config.json")).models || [];
  } catch {
    models = [];
  }
  updateCompareBtn();

  if (!models.length) {
    renderModelInfo(null);
    if (DEMO) {
      $("demoBanner").hidden = false;
      setStatus("demo", "Demo mode (no model deployed)");
    } else {
      setStatus("idle", "No model deployed yet");
    }
    return;
  }

  $("modelSel").innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(modelLabel(m))}</option>`).join("");
  $("modelPick").hidden = false;
  $("modelSel").addEventListener("change", (e) => selectModel(e.target.value));

  await loadScript(ORT_BASE + "ort.min.js");
  ort.env.wasm.wasmPaths = ORT_BASE;
  await selectModel(models[0].id);
}

// Loads (once) the ONNX session of a model, runs a warm-up prediction so the
// first real image is not slowed by start-up work, and loads the optional
// gallery embeddings.
function loadEntry(id) {
  if (!entries.has(id)) {
    const cfg = models.find((m) => m.id === id);
    const p = (async () => {
      setStatus("loading", `Loading ${cfg.name}…`);
      const bytes = await fetchWithProgress(cfg.file, (f) => { $("progress").style.width = `${Math.round(f * 100)}%`; });
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      const s = cfg.input_size;
      await session.run({ [inputName(cfg, session)]: new ort.Tensor("float32", new Float32Array(3 * s * s), [1, 3, s, s]) });
      return { cfg, session, gallery: await loadGallery(cfg) };
    })();
    p.catch(() => entries.delete(id)); // allow a retry after a failed load
    entries.set(id, p);
  }
  return entries.get(id);
}

async function selectModel(id) {
  const cfg = models.find((m) => m.id === id);
  current = null;
  $("modelSel").value = id;
  try {
    const entry = await loadEntry(id);
    if ($("modelSel").value !== id) return; // the user switched again while this loaded
    current = entry;
    setStatus("ready", `${cfg.name} ready`);
    renderModelInfo(cfg);
    fillCompareSel();
    if (lastImg) analyse(lastImg);
  } catch (e) {
    setStatus("error", `Failed to load ${cfg.name}: ${e.message || e}`);
  }
}

// gallery_embeddings: raw little-endian float16 or float32 (gallery_dtype), one row
// per gallery thumbnail in galleryItems order
async function loadGallery(cfg) {
  if (!cfg.gallery_embeddings || !cfg.outputs?.embedding) return null;
  try {
    const r = await fetch(cfg.gallery_embeddings);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    const data = cfg.gallery_dtype === "float16" ? halfToFloat(new Uint16Array(buf)) : new Float32Array(buf);
    const n = galleryItems.length;
    if (!n || data.length % n) throw new Error(`${data.length} values do not split into ${n} rows`);
    const dim = data.length / n;
    for (let i = 0; i < n; i++) normalise(data.subarray(i * dim, (i + 1) * dim));
    return { data, dim };
  } catch (e) {
    console.warn(`gallery_embeddings for ${cfg.id} not used: ${e.message || e}`);
    return null;
  }
}

// IEEE 754 half -> float (no Float16Array needed)
function halfToFloat(h) {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], e = (x >> 10) & 31, m = x & 1023, sign = x >> 15 ? -1 : 1;
    out[i] = sign * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
  }
  return out;
}

function normalise(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= s;
  return v;
}

// ---- Preprocessing ----------------------------------------------------------
// Image -> Float32Array in NCHW order, normalised with the model's mean / std.
// Reproduces the torchvision evaluation transform on a PIL image, so the model
// sees the same pixels as in evaluation:
//   "squash"      = Resize((s, s))
//   "center_crop" = Resize(int(s / crop_pct)) (shorter side) + CenterCrop(s)
// The canvas's own scaling uses a different filter, which flipped about 1 in 10
// top-1 predictions, so resizing is done here with PIL's bilinear resampling.
// PIL also ignores the EXIF orientation tag (the training code never applied it),
// while browsers draw images upright, so the rotation is undone before resizing.
const MAX_SIDE = 5000; // only photos larger than this are pre-shrunk by the canvas (memory / speed)

function preprocess(img, cfg) {
  const s = cfg.input_size;
  const o = img.orientation ?? 1;
  const swap = o >= 5;                       // orientations 5-8 swap width and height
  let w = swap ? img.naturalHeight : img.naturalWidth, h = swap ? img.naturalWidth : img.naturalHeight;
  const k = Math.max(1, Math.ceil(Math.max(w, h) / MAX_SIDE));
  w = Math.round(w / k); h = Math.round(h / k);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  // Maps the upright (displayed) image back onto the stored pixel grid
  const T = { 1: [1, 0, 0, 1, 0, 0], 2: [-1, 0, 0, 1, w, 0], 3: [-1, 0, 0, -1, w, h], 4: [1, 0, 0, -1, 0, h],
              5: [0, 1, 1, 0, 0, 0], 6: [0, -1, 1, 0, 0, h], 7: [0, -1, -1, 0, w, h], 8: [0, 1, -1, 0, w, 0] }[o] ?? [1, 0, 0, 1, 0, 0];
  ctx.setTransform(...T);
  ctx.drawImage(img, 0, 0, swap ? h : w, swap ? w : h);
  let px = ctx.getImageData(0, 0, w, h).data;

  let x0 = 0, y0 = 0;
  if (cfg.resize === "center_crop") {
    const r = Math.floor(s / (cfg.crop_pct ?? 0.875));
    const [nw, nh] = w <= h ? [r, Math.floor(r * h / w)] : [Math.floor(r * w / h), r];
    px = resizeBilinear(px, w, h, nw, nh);
    w = nw; h = nh;
    x0 = roundHalfEven((w - s) / 2); y0 = roundHalfEven((h - s) / 2); // CenterCrop uses Python's round()
  } else {
    px = resizeBilinear(px, w, h, s, s);
    w = h = s;
  }

  const mean = cfg.mean ?? IMAGENET.mean, std = cfg.std ?? IMAGENET.std;
  const out = new Float32Array(3 * s * s), n = s * s;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x, j = ((y + y0) * w + (x + x0)) * 4;
      for (let c = 0; c < 3; c++) out[c * n + i] = (px[j + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

// Python's round(): halves go to the even neighbour (46.5 -> 46, 47.5 -> 48)
function roundHalfEven(x) {
  const f = Math.floor(x);
  return x - f !== 0.5 ? Math.round(x) : f % 2 === 0 ? f : f + 1;
}

// PIL's Image.resize(..., BILINEAR): a triangle filter widened by the scale
// factor when shrinking (antialiasing), applied horizontally then vertically,
// with each pass rounded to 8 bits. Input and output are RGBA byte arrays.
function resizeBilinear(src, sw, sh, dw, dh) {
  const coeffs = (inSize, outSize) => {
    const scale = inSize / outSize, fs = Math.max(scale, 1), support = fs, list = [];
    for (let o = 0; o < outSize; o++) {
      const centre = (o + 0.5) * scale;
      const lo = Math.max(Math.trunc(centre - support + 0.5), 0);
      const hi = Math.min(Math.trunc(centre + support + 0.5), inSize);
      const w = [];
      let sum = 0;
      for (let x = lo; x < hi; x++) {
        const t = Math.abs((x - centre + 0.5) / fs);
        const v = t < 1 ? 1 - t : 0;
        w.push(v); sum += v;
      }
      list.push({ lo, w: w.map((v) => (sum ? v / sum : 0)) });
    }
    return list;
  };
  const clip8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v));

  const cx = coeffs(sw, dw);
  const mid = new Uint8ClampedArray(dw * sh * 4);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      const { lo, w } = cx[x];
      let r = 0, g = 0, b = 0;
      for (let t = 0, p = (y * sw + lo) * 4; t < w.length; t++, p += 4) { r += src[p] * w[t]; g += src[p + 1] * w[t]; b += src[p + 2] * w[t]; }
      const q = (y * dw + x) * 4;
      mid[q] = clip8(r); mid[q + 1] = clip8(g); mid[q + 2] = clip8(b); mid[q + 3] = 255;
    }
  }
  const cy = coeffs(sh, dh);
  const out = new Uint8ClampedArray(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    const { lo, w } = cy[y];
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0;
      for (let t = 0, p = (lo * dw + x) * 4; t < w.length; t++, p += dw * 4) { r += mid[p] * w[t]; g += mid[p + 1] * w[t]; b += mid[p + 2] * w[t]; }
      const q = (y * dw + x) * 4;
      out[q] = clip8(r); out[q + 1] = clip8(g); out[q + 2] = clip8(b); out[q + 3] = 255;
    }
  }
  return out;
}

const inputName = (cfg, session) => cfg.input ?? session.inputNames[0];

// ---- Scoring ----------------------------------------------------------------
function logSoftmax(x) {
  const m = Math.max(...x);
  const lse = m + Math.log(x.reduce((a, v) => a + Math.exp(v - m), 0));
  return Array.from(x, (v) => v - lse);
}
const softmax = (x) => logSoftmax(x).map(Math.exp);

// Turns the model outputs into a probability for each of the 333 compositions.
// Two-head models: score = log p(crop) + log p(disease) (+ unseen_bias for
// unseen pairs), renormalised over the valid pairs. Models with a composition
// output: softmax over that output directly.
function scoreCompositions(out, cfg) {
  const comps = classes.compositions;
  let comp, crop, dis;
  if (out.composition) {
    comp = softmax(out.composition);
  } else {
    crop = logSoftmax(out.crop);
    dis = logSoftmax(out.disease);
    const bias = cfg.unseen_bias ?? 0;
    comp = softmax(comps.map((c) => crop[c.crop] + dis[c.disease] + (c.seen ? 0 : bias)));
    crop = crop.map(Math.exp);
    dis = dis.map(Math.exp);
  }
  // Composition-only models: crop and disease probabilities are the marginals
  if (!crop) {
    crop = new Array(classes.crops.length).fill(0);
    dis = new Array(classes.diseases.length).fill(0);
    comps.forEach((c, k) => { crop[c.crop] += comp[k]; dis[c.disease] += comp[k]; });
  }
  return { comp, crop, dis };
}

const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
const topK = (comp, k = TOP_K) => comp.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, k);

// Random scores with one clear winner, used only in demo mode
function demoScores() {
  const n = classes.compositions.length;
  const logits = Array.from({ length: n }, () => Math.random() * 3);
  logits[Math.floor(Math.random() * n)] += 6;
  const comp = softmax(logits);
  const crop = new Array(classes.crops.length).fill(0);
  const dis = new Array(classes.diseases.length).fill(0);
  classes.compositions.forEach((c, k) => { crop[c.crop] += comp[k]; dis[c.disease] += comp[k]; });
  return { comp, crop, dis };
}

// ---- Running a model ----------------------------------------------------------
// Returns the raw outputs of one forward pass plus its time in ms
async function forward(entry, img) {
  const { cfg, session } = entry;
  const t0 = performance.now();
  const s = cfg.input_size;
  const res = await session.run({ [inputName(cfg, session)]: new ort.Tensor("float32", preprocess(img, cfg), [1, 3, s, s]) });
  const names = cfg.outputs ?? {};
  const get = (key) => (names[key] && res[names[key]] ? res[names[key]] : null);
  const arr = (key) => (get(key) ? Array.from(get(key).data) : null);
  return {
    raw: { crop: arr("crop"), disease: arr("disease"), composition: arr("composition") },
    emb: get("embedding") ? normalise(Float32Array.from(get("embedding").data)) : null,
    ms: performance.now() - t0,
  };
}

// Cosine similarity of the query embedding to every gallery thumbnail
function similarities(emb, gallery) {
  if (!emb || !gallery) return null;
  if (emb.length !== gallery.dim) {
    console.warn(`embedding has ${emb.length} values, gallery rows have ${gallery.dim}: similarity skipped`);
    return null;
  }
  const n = galleryItems.length, d = gallery.dim, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0, o = i * d; j < d; j++) s += emb[j] * gallery.data[o + j];
    out[i] = s;
  }
  return out;
}

// Full result of one model on one image (used by the comparison panel)
async function runModel(entry, img) {
  const f = await forward(entry, img);
  const scores = scoreCompositions(f.raw, entry.cfg);
  return { model: entry.cfg, scores, sims: similarities(f.emb, entry.gallery) };
}

// ---- Analysis pipeline -------------------------------------------------------
const STEPS = [
  "Extracting visual features",
  "Combining crop & disease scores over 333 pairs",
  "Retrieving closest seen training samples",
];

const pipe = {
  start(labels) {
    this.t = [];
    $("pipeline").className = "pipeline";
    $("pipeline").innerHTML = labels.map((l) => `<li><span class="ic"></span><span>${esc(l)}…</span><span class="ms"></span></li>`).join("");
  },
  // Marks step i as running, yields a frame so it paints, then starts its clock
  async run(i) { $("pipeline").children[i].className = "run"; await nextFrame(); this.t0 = performance.now(); },
  done(i, ms = performance.now() - this.t0) {
    const li = $("pipeline").children[i];
    li.className = "done";
    li.querySelector(".ms").textContent = ms < 1 ? "<1 ms" : `${Math.round(ms)} ms`;
    li.children[1].textContent = li.children[1].textContent.replace(/…$/, "");
    this.t.push(ms);
  },
  finish() { $("pipeline").classList.add("summary"); },
  clear() { $("pipeline").innerHTML = ""; },
};

async function analyse(img) {
  const token = imgToken;
  lastImg = img;
  showWorkspace();

  if (!current && !DEMO) {
    // A model is listed but still loading or failed: keep the last state
    $("noModel").hidden = models.length > 0;
    $("preds").hidden = true;
    $("gallery").hidden = true;
    pipe.clear();
    renderTruth(null);
    if (models.length) setStatus("loading", "Model is still loading, the image will be analysed when it is ready");
    return;
  }

  const entry = current;
  pipe.start(DEMO && !entry ? ["Generating random demo scores", STEPS[1], STEPS[2]] : STEPS);

  await pipe.run(0);
  const f = entry ? await forward(entry, img) : null;
  pipe.done(0, f ? f.ms : undefined);

  await pipe.run(1);
  const scores = entry ? scoreCompositions(f.raw, entry.cfg) : demoScores();
  pipe.done(1);

  await pipe.run(2);
  const sims = entry ? similarities(f.emb, entry.gallery) : null;
  if (token !== imgToken || entry !== current) return; // a newer image or model took over
  primary = { model: entry ? entry.cfg : null, scores, sims };
  if (entry) results.set(entry.cfg.id, primary);
  renderPrediction(primary);
  pipe.done(2);
  pipe.finish();

  if (compareOn) renderCompare();
}

// ---- Prediction ---------------------------------------------------------------
function compName(k) {
  const c = classes.compositions[k];
  return `${classes.crops[c.crop]} · ${classes.diseases[c.disease]}`;
}

const ICON_SEEN = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ICON_UNSEEN = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l2.2 6.3L21 12l-6.8 2.7L12 21l-2.2-6.3L3 12l6.8-2.7z" fill="currentColor"/></svg>`;
const domainBadge = (seen) => (seen
  ? `${ICON_SEEN}In-distribution (seen)`
  : `${ICON_UNSEEN}Out-of-distribution (unseen, zero-shot)`);
const miniTag = (seen) => `<span class="mini ${seen ? "seen" : "unseen"}">${seen ? "Seen" : "Unseen"}</span>`;

function confLevel(p) {
  return p >= 0.7 ? ["high", "High confidence"] : p >= 0.4 ? ["mid", "Medium confidence"] : ["low", "Low confidence"];
}

function renderPrediction({ model, scores: { comp, crop, dis } }) {
  $("noModel").hidden = true;
  $("preds").hidden = false;

  const top = topK(comp);
  const [bestK, bestP] = top[0];
  const best = classes.compositions[bestK];

  $("predModel").textContent = model ? `· ${model.short_name ?? model.name}` : "· demo";
  $("domain").className = `domain ${best.seen ? "seen" : "unseen"}`;
  $("domain").innerHTML = domainBadge(best.seen);
  $("best").textContent = compName(bestK);
  $("bestTags").innerHTML =
    `<span class="pill crop"><small>Crop type</small><b>${esc(classes.crops[best.crop])}</b></span>` +
    `<span class="pill disease"><small>Disease type</small><b>${esc(classes.diseases[best.disease])}</b></span>` +
    `<span class="pill comp"><small>Composed class</small><b>${esc(compName(bestK))}</b></span>`;
  $("bestConf").textContent = pct(bestP);
  const [lvl, lvlText] = confLevel(bestP);
  $("badge").className = `badge ${lvl}`;
  $("badge").textContent = lvlText;

  const ci = argmax(crop), di = argmax(dis);
  $("heads").textContent =
    `Crop head on its own: ${classes.crops[ci]} (${pct(crop[ci])}) · ` +
    `Disease head on its own: ${classes.diseases[di]} (${pct(dis[di])})`;

  // Top-5 list, including the top match, so any of them can drive the gallery
  $("top5").innerHTML = top.map(([k, p], i) =>
    `<li><button type="button" data-k="${k}" aria-pressed="${i === 0}">` +
    `<span class="rank">#${i + 1}</span>` +
    `<span class="name">${esc(compName(k))}${miniTag(classes.compositions[k].seen)}</span>` +
    `<span class="pct">${pct(p)}</span>` +
    `<span class="bar"><i data-w="${p * 100}"></i></span></button></li>`).join("");
  $("top5").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("top5").querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    renderGallery(Number(b.dataset.k), b.dataset.k === String(bestK));
  }));

  // Bars start at 0% and grow to their value, so every prediction animates
  const bars = [$("bestBar"), ...$("top5").querySelectorAll(".bar>i")];
  bars.forEach((b) => { b.style.width = "0%"; });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("bestBar").style.width = `${bestP * 100}%`;
    $("top5").querySelectorAll(".bar>i").forEach((b) => { b.style.width = `${b.dataset.w}%`; });
  }));

  renderTruth(top);
  renderGallery(bestK, true);
}

// Ground-truth comparison banner for sample images.
// top: top-5 list of [composition, p], or null while there is no prediction.
function renderTruth(top) {
  const el = $("truth");
  if (!lastTruth) { el.hidden = true; return; }
  const t = classes.compositions[lastTruth.truth];
  const file = lastTruth.split === "unseen" ? "test_unseen.csv" : "test_seen.csv";
  let verdict = `<span class="verdict wait">No prediction</span>`, predRow = "—", parts = "";
  if (top) {
    const pk = top[0][0], p = classes.compositions[pk];
    const ok = pk === lastTruth.truth;
    const inTop5 = top.some(([k]) => k === lastTruth.truth);
    verdict = ok ? `<span class="verdict ok">✓ Match</span>` : `<span class="verdict bad">✗ Mismatch</span>`;
    predRow = esc(compName(pk));
    const part = (good, label) => `<span class="${good ? "y" : "n"}">${good ? "✓" : "✗"} ${label}</span>`;
    parts = `<div class="parts">${part(p.crop === t.crop, "Crop")}${part(p.disease === t.disease, "Disease")}${part(inTop5, "In top-5")}</div>`;
    el.className = `truth ${ok ? "ok" : "bad"}`;
  } else {
    el.className = "truth";
  }
  el.innerHTML =
    `<div class="t-head"><span>Ground truth comparison</span>${verdict}</div>` +
    `<dl><dt>Actual</dt><dd>${esc(compName(lastTruth.truth))} ${miniTag(t.seen)}<br><small>${file}</small></dd>` +
    `<dt>Top-1</dt><dd>${predRow}</dd></dl>${parts}`;
  el.hidden = false;
}

// ---- Closest training matches -----------------------------------------------
// Tabs: nearest overall (only with embeddings), same class (seen only), same
// crop, same disease on other crops. With embeddings each tab is ranked by
// cosine similarity; without, it shows a fixed sample of each class.
function renderGallery(k, isTop) {
  const c = classes.compositions[k];
  const sims = primary?.sims ?? null;
  const tabs = [];
  const ofComp = (pred) => galleryItems.map((it, i) => ({ ...it, i })).filter((it) => pred(classes.compositions[it.comp]));
  // Without similarities: one image per class, so a row shows many classes
  const pick = (items, perClass) => {
    if (sims) return items.sort((a, b) => sims[b.i] - sims[a.i]).slice(0, GALLERY_MAX);
    if (!perClass) return items.slice(0, GALLERY_MAX);
    const seen = new Set();
    return items.filter((it) => !seen.has(it.comp) && seen.add(it.comp)).slice(0, GALLERY_MAX);
  };

  const crop = classes.crops[c.crop], disease = classes.diseases[c.disease];
  if (sims) tabs.push({ key: "near", label: "Nearest overall", desc: "Most similar training images of any class.", items: pick(ofComp(() => true)) });
  if (c.seen) tabs.push({ key: "class", label: "Same class", desc: `Training images of ${compName(k)}.`, items: pick(ofComp((o) => o.id === k)), total: c.n_train });
  tabs.push({ key: "crop", label: "Crop matches", desc: `${crop} leaves with ${c.seen ? "other diseases or healthy" : "any disease, or healthy"}: the leaf structure the crop head learnt.`, items: pick(ofComp((o) => o.crop === c.crop && o.id !== k), true) });
  tabs.push({ key: "disease", label: "Disease symptom matches", desc: `${disease} on crops other than ${crop}: the symptom pattern the disease head learnt.`, items: pick(ofComp((o) => o.disease === c.disease && o.crop !== c.crop), true) });

  const active = tabs.find((t) => t.key === galleryTab) ?? tabs[0];
  $("galleryTitle").textContent = "Closest training matches (seen set)";
  $("galleryNote").innerHTML =
    `${isTop ? "Top-1 prediction" : "Selected candidate"}: <b>${esc(compName(k))}</b>. ` +
    (c.seen ? "" : "This combination never appears in training, so the model composes it from the crop and the disease it learnt separately. ") +
    (sims
      ? "Images are ranked by cosine similarity between the model's embedding of your photo and of each training image."
      : "These are a fixed sample of training images of each class (DPD-Small train.csv labels). The selected model exports no image embedding, so they are not ranked by similarity.");

  $("galleryTabs").innerHTML = tabs.map((t) =>
    `<button type="button" role="tab" data-key="${t.key}" aria-selected="${t === active}">${esc(t.label)}` +
    `<span class="count">${t.items.length}${t.total > t.items.length ? ` of ${t.total}` : ""}</span></button>`).join("");
  $("galleryTabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    galleryTab = b.dataset.key;
    renderGallery(k, isTop);
  }));

  $("galleryGrid").innerHTML = `<p class="hint-line tab-desc">${esc(active.desc)}</p>` + (active.items.length
    ? `<div class="grid">${active.items.map((it) => {
        const src = it.src.split("/").pop().split("_")[0];
        return `<figure><img src="${esc(it.src)}" alt="${esc(compName(it.comp))}" loading="lazy">` +
          (sims ? `<span class="sim" title="Cosine similarity of model embeddings">${Math.round(Math.max(0, sims[it.i]) * 100)}% similar</span>` : "") +
          `<figcaption><b>${esc(compName(it.comp))}</b>train · ${esc(src)}</figcaption></figure>`;
      }).join("")}</div>`
    : `<p class="note">No training images share this.</p>`);
  $("gallery").hidden = false;
}

// ---- Zoom / pan inspector -------------------------------------------------------
const view = { s: 1, x: 0, y: 0 };

// Fits the image inside the square viewport, keeping its aspect ratio
function fitStage() {
  const img = $("preview"), V = $("viewport").clientWidth;
  if (!img.naturalWidth || !V) return;
  const r = Math.min(V / img.naturalWidth, V / img.naturalHeight);
  const w = img.naturalWidth * r, h = img.naturalHeight * r;
  Object.assign($("stage").style, { width: `${w}px`, height: `${h}px`, left: `${(V - w) / 2}px`, top: `${(V - h) / 2}px` });
  applyView();
}

function applyView() {
  const V = $("viewport").clientWidth;
  view.x = Math.min(0, Math.max(V - V * view.s, view.x));
  view.y = Math.min(0, Math.max(V - V * view.s, view.y));
  $("pan").style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.s})`;
  $("zoomVal").textContent = `${Math.round(view.s * 100)}%`;
  $("viewport").classList.toggle("zoomed", view.s > 1);
}

function zoomAt(factor, cx, cy) {
  const ns = Math.min(ZOOM_MAX, Math.max(1, view.s * factor));
  view.x = cx - (cx - view.x) * (ns / view.s);
  view.y = cy - (cy - view.y) * (ns / view.s);
  view.s = ns;
  applyView();
}

function resetView() { view.s = 1; view.x = view.y = 0; applyView(); }

{
  const vp = $("viewport");
  const centre = () => vp.clientWidth / 2;
  const local = (e) => { const r = vp.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  $("zoomIn").addEventListener("click", () => zoomAt(1.5, centre(), centre()));
  $("zoomOut").addEventListener("click", () => zoomAt(1 / 1.5, centre(), centre()));
  $("zoomReset").addEventListener("click", resetView);
  vp.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(Math.exp(-e.deltaY * 0.0015), ...local(e)); }, { passive: false });
  vp.addEventListener("dblclick", (e) => (view.s > 1 ? resetView() : zoomAt(2, ...local(e))));
  let drag = null;
  vp.addEventListener("pointerdown", (e) => {
    if (view.s <= 1) return;
    drag = { x: e.clientX, y: e.clientY };
    vp.setPointerCapture(e.pointerId);
    vp.classList.add("dragging");
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    view.x += e.clientX - drag.x; view.y += e.clientY - drag.y;
    drag = { x: e.clientX, y: e.clientY };
    applyView();
  });
  const end = () => { drag = null; vp.classList.remove("dragging"); };
  vp.addEventListener("pointerup", end);
  vp.addEventListener("pointercancel", end);
  new ResizeObserver(fitStage).observe(vp);
  $("preview").addEventListener("load", () => { resetView(); fitStage(); });
}

// ---- Model comparison ----------------------------------------------------------
function updateCompareBtn() {
  const ok = models.length >= 2;
  $("compareBtn").disabled = !ok;
  $("compareBtn").title = ok ? "Show two models' predictions side by side" : "Needs at least two models in model/config.json";
}

function fillCompareSel() {
  const others = models.filter((m) => m.id !== current?.cfg.id);
  const keep = others.some((m) => m.id === $("compareSel").value) ? $("compareSel").value : others[0]?.id;
  $("compareSel").innerHTML = others.map((m) => `<option value="${esc(m.id)}">${esc(modelLabel(m))}</option>`).join("");
  if (keep) $("compareSel").value = keep;
}

$("compareBtn").addEventListener("click", () => {
  compareOn = !compareOn;
  $("compareBtn").setAttribute("aria-pressed", String(compareOn));
  $("compare").hidden = !compareOn;
  if (compareOn) {
    fillCompareSel();
    renderCompare();
    $("compare").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
$("compareSel").addEventListener("change", renderCompare);

function compareColumn(cfg, res) {
  const head = `<p class="role">${esc(cfg.role ?? "Model")}</p><h4>${esc(cfg.name)}</h4>`;
  if (!res) return `<div class="cmp">${head}<p class="loading">Running…</p></div>`;
  if (res.error) return `<div class="cmp">${head}<p class="loading">Failed: ${esc(res.error)}</p></div>`;
  const top = topK(res.scores.comp);
  const [k, p] = top[0];
  const truth = lastTruth?.truth;
  const verdict = truth === undefined ? "" : k === truth ? `<span class="verdict ok">✓ Match</span>` : `<span class="verdict bad">✗ Mismatch</span>`;
  return `<div class="cmp">${head}<p class="c-top">${esc(compName(k))}</p>` +
    `<div class="c-meta"><b class="num">${pct(p)}</b>${miniTag(classes.compositions[k].seen)}${verdict}</div>` +
    `<ol>${top.map(([kk, pp], i) =>
      `<li class="${kk === truth ? "hit" : ""}"><span>#${i + 1} ${esc(compName(kk))}${kk === truth ? " ✓" : ""}</span>` +
      `<span class="pct">${pct(pp)}</span><span class="bar"><i style="width:${pp * 100}%"></i></span></li>`).join("")}</ol></div>`;
}

async function renderCompare() {
  if (!compareOn) return;
  const otherId = $("compareSel").value;
  const other = models.find((m) => m.id === otherId);
  if (!current || !other) {
    $("compareNote").textContent = "Waiting for the selected model to load.";
    $("compareGrid").innerHTML = "";
    return;
  }
  if (!lastImg) {
    $("compareNote").textContent = "Choose an image or a sample to run both models on it.";
    $("compareGrid").innerHTML = "";
    return;
  }
  const token = imgToken, a = current.cfg;
  const draw = () => {
    const ra = results.get(a.id), rb = results.get(other.id);
    $("compareGrid").innerHTML = compareColumn(a, ra) + compareColumn(other, rb);
    if (ra?.scores && rb?.scores) {
      const ka = argmax(ra.scores.comp), kb = argmax(rb.scores.comp);
      let note = ka === kb ? `Both models agree on ${compName(ka)}.` : `The models disagree on the top-1 combination.`;
      if (lastTruth) {
        const ok = [ka, kb].map((x) => x === lastTruth.truth);
        note += ok[0] && ok[1] ? " Both are correct." : ok[0] ? ` Only ${a.short_name ?? a.name} is correct.` : ok[1] ? ` Only ${other.short_name ?? other.name} is correct.` : " Neither is correct.";
      }
      $("compareNote").textContent = note;
    } else {
      $("compareNote").textContent = "Running both models on the same image…";
    }
  };
  draw();
  if (!results.has(other.id)) {
    try {
      const entry = await loadEntry(other.id);
      setStatus("ready", `${a.name} ready`);
      const res = await runModel(entry, lastImg);
      if (token !== imgToken) return;
      results.set(other.id, res);
    } catch (e) {
      results.set(other.id, { error: e.message || String(e) });
    }
    if ($("compareSel").value === otherId) draw();
  }
}

// ---- Model details, metrics and status ------------------------------------------
function renderModelInfo(cfg) {
  const m = $("metrics");
  if (cfg && cfg.metrics && cfg.metrics.length) {
    m.innerHTML = `<div><b>${esc(cfg.short_name ?? cfg.name)}</b><span>deployed model</span></div>` +
      cfg.metrics.map((x) => `<div><b>${esc(x.value)}</b><span>${esc(x.label)}</span></div>`).join("");
  } else {
    m.innerHTML = `<p>${cfg ? "No test-set results listed for this model." : "Test-set results will appear here once a model is deployed. Open “View benchmarks” for every evaluated run."}</p>`;
  }
  $("aboutModelHead").textContent = cfg ? modelLabel(cfg) : "Selected model";
  $("aboutModel").innerHTML = cfg && cfg.about && cfg.about.length
    ? cfg.about.map((p) => `<p>${esc(p)}</p>`).join("")
    : `<p>${cfg ? "No description provided." : "No model deployed yet."}</p>`;
}

function setStatus(state, text) {
  $("status").textContent = text;
  $("dot").className = `dot ${state === "loading" ? "" : state}`;
  $("progressWrap").classList.toggle("done", state !== "loading");
}

// ---- Image input ------------------------------------------------------------
function showWorkspace() {
  $("result").hidden = false;
  document.body.classList.add("has-result");
}

// Shows the image and analyses a separate, unstyled copy at full resolution
function showImage(src, truth = null) {
  imgToken++;
  results = new Map();
  primary = null;
  lastTruth = truth;
  $("dropError").hidden = true;
  showWorkspace();
  renderTruth(null);
  $("preview").src = src;
  $("examples").querySelectorAll("button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.file === src)));
  const token = imgToken;
  loadImage(src)
    .then((raw) => { if (token === imgToken) return analyse(raw); })
    .catch((e) => (e.name === "EncodingError" || e.message === "decode"
      ? showDropError("That image could not be read. Try a JPG or PNG.")
      : setStatus("error", `Prediction failed: ${e.message || e}`)));
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// Decodes an image for the model and records its EXIF orientation (1 = none)
async function loadImage(src) {
  let orientation = 1;
  try { orientation = exifOrientation(await (await fetch(src)).arrayBuffer()); } catch { /* not a JPEG, or unreadable: assume upright */ }
  const img = new Image();
  img.src = src;
  try { await img.decode(); } catch { throw new Error("decode"); }
  img.orientation = orientation;
  return img;
}

// Reads the orientation tag (0x0112) from a JPEG's EXIF block
function exifOrientation(buf) {
  const v = new DataView(buf);
  if (v.byteLength < 4 || v.getUint16(0) !== 0xffd8) return 1;
  for (let off = 2; off + 4 <= v.byteLength;) {
    const marker = v.getUint16(off), len = v.getUint16(off + 2);
    if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) break;   // not a marker, or image data starts
    if (marker === 0xffe1 && v.getUint32(off + 4) === 0x45786966) { // "Exif"
      const tiff = off + 10, le = v.getUint16(tiff) === 0x4949;
      const ifd = tiff + v.getUint32(tiff + 4, le), n = v.getUint16(ifd, le);
      for (let i = 0; i < n; i++) {
        const e = ifd + 2 + i * 12;
        if (v.getUint16(e, le) === 0x0112) return v.getUint16(e + 8, le);
      }
      return 1;
    }
    off += 2 + len;
  }
  return 1;
}

function showDropError(msg) {
  $("dropError").textContent = msg;
  $("dropError").hidden = false;
}

function openFile(f) {
  if (!f) return;
  if (!f.type.startsWith("image/")) { showDropError(`“${f.name}” is not an image. Choose a JPG or PNG.`); return; }
  if (uploadUrl) URL.revokeObjectURL(uploadUrl);
  uploadUrl = URL.createObjectURL(f);
  showImage(uploadUrl);
}

$("file").addEventListener("change", (e) => openFile(e.target.files[0]));
const drop = $("drop");
$("dropTarget").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); }
});
let dragDepth = 0; // dragenter/leave fire for every child, so count them
drop.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; drop.classList.add("over"); });
drop.addEventListener("dragover", (e) => e.preventDefault());
drop.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; drop.classList.remove("over"); } });
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  drop.classList.remove("over");
  openFile(e.dataTransfer.files[0]);
});

$("uploadNew").addEventListener("click", () => $("file").click());
$("reset").addEventListener("click", () => {
  imgToken++;
  $("result").hidden = true;
  document.body.classList.remove("has-result");
  $("preview").removeAttribute("src");
  $("file").value = "";
  lastImg = lastTruth = primary = null;
  results = new Map();
  pipe.clear();
  $("examples").querySelectorAll("button").forEach((b) => b.setAttribute("aria-current", "false"));
  if (compareOn) renderCompare();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
$("aboutBtn").addEventListener("click", () => $("about").showModal());
for (const d of ["about", "bench"]) $(d).addEventListener("click", (e) => { if (e.target === $(d)) $(d).close(); });

// ---- Sample images --------------------------------------------------------------
function buildExamples() {
  $("examples").innerHTML = classes.examples.map((ex, i) => {
    const c = classes.compositions[ex.truth];
    const seen = ex.split !== "unseen";
    return `<button type="button" data-i="${i}" data-split="${seen ? "seen" : "unseen"}" data-file="${esc(ex.file)}" aria-current="false" title="${esc(compName(ex.truth))}">` +
      `<span class="thumb"><img src="${esc(ex.file)}" alt="${esc(compName(ex.truth))}" loading="lazy">${miniTag(seen)}` +
      `<span class="overlay">${esc(classes.crops[c.crop])}<br>${esc(classes.diseases[c.disease])}</span></span>` +
      `<span class="cap">${esc(compName(ex.truth))}</span></button>`;
  }).join("");
  $("examples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const ex = classes.examples[Number(b.dataset.i)];
    showImage(ex.file, { truth: ex.truth, split: ex.split });
  }));
  const unseen = classes.examples.filter((ex) => ex.split === "unseen");
  $("randomUnseen").hidden = !unseen.length;
}

$("sampleFilter").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  $("sampleFilter").querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  $("examples").querySelectorAll("button").forEach((x) => { x.hidden = b.dataset.f !== "all" && x.dataset.split !== b.dataset.f; });
});

$("randomUnseen").addEventListener("click", () => {
  const pool = classes.examples.filter((ex) => ex.split === "unseen");
  const others = pool.filter((ex) => ex.file !== $("preview").getAttribute("src"));
  const ex = (others.length ? others : pool)[Math.floor(Math.random() * (others.length || pool.length))];
  showImage(ex.file, { truth: ex.truth, split: ex.split });
});

// ---- Tooltip --------------------------------------------------------------------
// The tooltip moves into an open dialog so it renders above the modal layer
function showTip(html, x, y) {
  const tip = $("tooltip");
  const host = document.querySelector("dialog[open]") ?? document.body;
  if (tip.parentElement !== host) host.appendChild(tip);
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(window.innerWidth - r.width - 8, Math.max(8, x + 12))}px`;
  tip.style.top = `${y + 14 + r.height > window.innerHeight ? y - r.height - 10 : y + 14}px`;
}
const hideTip = () => { $("tooltip").hidden = true; };

// ---- Benchmarks dialog ----------------------------------------------------------
$("benchBtn").addEventListener("click", async () => {
  $("bench").showModal();
  if (bench) return;
  try {
    bench = await fetchJSON("data/benchmarks.json");
    renderBench();
  } catch (e) {
    $("benchBody").innerHTML = `<p class="note">Could not load benchmark results (${esc(e.message || e)}). Build them with <code>python UserInterface/tools/build_benchmarks.py</code>.</p>`;
  }
});
$("bench").addEventListener("close", hideTip);

function renderBench() {
  const runs = bench.runs;
  const deployed = new Set(models.map((m) => m.benchmark_id).filter(Boolean));
  const cols = [
    ["seen", "top1"], ["unseen", "top1"], ["hm", "top1"],
    ["seen", "top5"], ["unseen", "top5"], ["hm", "top5"],
    ["calibrated", "best_hm"], ["calibrated", "seen"], ["calibrated", "unseen"],
  ];
  const val = (r, [a, b]) => r[a]?.[b];
  const best = cols.map((c) => Math.max(...runs.map((r) => val(r, c) ?? -1)));

  const table =
    `<div class="table-wrap"><table class="bench"><thead>` +
    `<tr><th></th><th colspan="3">Top-1 accuracy</th><th colspan="3">Top-5 accuracy</th><th colspan="3">Calibrated (best HM)</th></tr>` +
    `<tr><th>Model</th><th>Seen</th><th>Unseen</th><th>HM</th><th>Seen</th><th>Unseen</th><th>HM</th><th>HM</th><th>Seen</th><th>Unseen</th></tr>` +
    `</thead><tbody>${runs.map((r) =>
      `<tr class="${deployed.has(r.id) ? "deployed" : ""}"><td><span class="role">${esc(r.role)}${deployed.has(r.id) ? " · deployed" : ""}</span>${esc(r.name)}</td>` +
      cols.map((c, i) => { const v = val(r, c); return `<td class="${v === best[i] ? "best" : ""}">${v == null ? "—" : (v * 100).toFixed(1)}</td>`; }).join("") +
      `</tr>`).join("")}</tbody></table></div>` +
    `<p class="hint-line">Values in %. ★ marks the best run in each column. Calibrated: a bias added to unseen-pair scores is swept on the test set; the columns give the best harmonic mean along that sweep and the seen / unseen accuracy at that point.</p>`;

  const charts =
    `<h3>Top-1 vs Top-5 composition accuracy</h3>` +
    `<div class="viz-legend"><span><i style="background:var(--viz-top1)"></i>Top-1</span><span><i style="background:var(--viz-top5)"></i>Top-5 (extends the Top-1 bar)</span></div>` +
    `<div class="viz viz-multiples">${[["seen", "Seen (test_seen)"], ["unseen", "Unseen (test_unseen)"], ["hm", "Harmonic mean"]]
      .map(([key, title]) => `<div><h4>${title}</h4>${barChart(runs, key)}</div>`).join("")}</div>`;

  const confRuns = runs.filter((r) => bench.confusion[r.id]);
  const conf = confRuns.length
    ? `<h3>Confusion on test_seen</h3>` +
      `<div class="bench-pick"><label>Run <select id="cmRun">${confRuns.map((r) => `<option value="${esc(r.id)}">${esc(`${r.role} · ${r.name}`)}</option>`).join("")}</select></label>` +
      `<label class="switch"><input type="checkbox" id="cmErrors" checked><span>Errors only (hide the diagonal)</span></label></div>` +
      `<div class="cm-layout" id="cmBody"></div>` +
      (confRuns.length < runs.length ? `<p class="hint-line">Runs not listed saved no confusion data.</p>` : "")
    : "";

  $("benchBody").innerHTML = table + charts + conf;

  $("benchBody").querySelectorAll(".viz g.row").forEach((g) => {
    const tip = (e) => showTip(g.dataset.tip, e.clientX, e.clientY);
    g.addEventListener("mousemove", tip);
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("focus", () => { const r = g.getBoundingClientRect(); showTip(g.dataset.tip, r.left, r.bottom - 10); });
    g.addEventListener("blur", hideTip);
  });
  if (confRuns.length) {
    $("cmRun").addEventListener("change", renderConfusion);
    $("cmErrors").addEventListener("change", renderConfusion);
    // Start on a run that has the crop matrix, if any
    const withMatrix = confRuns.find((r) => bench.confusion[r.id].crop_matrix);
    if (withMatrix) $("cmRun").value = withMatrix.id;
    renderConfusion();
  }
}

// Horizontal bars, one row per run: Top-1 solid, Top-5 a lighter step behind it.
// 4px rounded data end, square at the baseline; hairline grid; one 0-100 axis.
function barChart(runs, key) {
  const W = 320, L = 0, R = 64, rowH = 38, barH = 12, top = 6;
  const plotW = W - L - R, H = top + runs.length * rowH + 18;
  const x = (v) => L + (v || 0) * plotW;
  const grid = [0, 0.25, 0.5, 0.75, 1].map((t) =>
    `<line class="${t ? "grid-l" : "axis-l"}" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 16}"/>` +
    `<text x="${x(t)}" y="${H - 4}" text-anchor="${t === 0 ? "start" : t === 1 ? "end" : "middle"}">${t * 100}</text>`).join("");
  const rows = runs.map((r, i) => {
    const y = top + i * rowH, by = y + 17, v1 = r[key].top1, v5 = r[key].top5;
    const name = `${r.role}: ${r.name}`;
    return `<g class="row" tabindex="0" data-tip="${esc(`<b>${esc(name)}</b><br>Top-1 ${pct(v1)} · Top-5 ${pct(v5)}`)}">` +
      `<rect class="hit" x="0" y="${y}" width="${W}" height="${rowH}"/>` +
      `<text class="lbl" x="${L}" y="${y + 11}">${esc(r.short ?? r.role)}</text>` +
      `<path class="t5" d="${barPath(x(0), by, x(v5) - x(0), barH)}"/>` +
      `<path class="t1" d="${barPath(x(0), by, x(v1) - x(0), barH)}"/>` +
      `<text x="${W - R + 6}" y="${by + 10}" class="lbl">${(v1 * 100).toFixed(1)}<tspan> / ${(v5 * 100).toFixed(1)}</tspan></text></g>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Top-1 and top-5 accuracy per run">${grid}${rows}</svg>`;
}

function barPath(x, y, w, h, r = 4) {
  if (w <= 0) return "";
  r = Math.min(r, w, h / 2);
  return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`;
}

// Crop-level confusion heatmap (row-normalised) + the most confused pairs
const CM_LIGHT = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
const CM_DARK = ["#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"];

function renderConfusion() {
  const run = bench.runs.find((r) => r.id === $("cmRun").value);
  const data = bench.confusion[run.id];
  const errorsOnly = $("cmErrors").checked;
  $("cmErrors").parentElement.hidden = !data.crop_matrix;
  const topTable = data.top
    ? `<div><h4>Most confused combinations</h4><table class="mini-t"><thead><tr><th>Actual</th><th>Predicted as</th><th class="n">Images</th><th class="n">Rate</th></tr></thead>` +
      `<tbody>${data.top.map((t) => `<tr><td>${esc(t.true)}</td><td>${esc(t.pred)}</td><td class="n">${t.count}</td><td class="n">${pct(t.rate, 0)}</td></tr>`).join("")}</tbody></table>` +
      `<p class="hint-line">Rate = share of that actual class's test images.</p></div>`
    : `<div><p class="note">No list of confused combinations was saved for this run.</p></div>`;

  if (!data.crop_matrix) {
    $("cmBody").innerHTML = `<div><p class="note">This run saved no per-image predictions, so the crop confusion matrix is not available.</p></div>${topTable}`;
    return;
  }
  const { crops, matrix, n_images } = data.crop_matrix;
  const n = crops.length, cell = 8;
  const rates = matrix.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s); });
  let max = 0;
  rates.forEach((row, i) => row.forEach((v, j) => { if (!(errorsOnly && i === j)) max = Math.max(max, v); }));

  $("cmBody").innerHTML =
    `<div class="cm-wrap"><h4>Crop confusion (${n} crops, ${n_images} images)</h4>` +
    `<canvas id="cm" width="${n * cell}" height="${n * cell}" aria-label="Crop confusion matrix"></canvas>` +
    `<div class="cm-scale"><span>0%</span><i></i><span>${pct(max, 0)}</span></div>` +
    `<p class="hint-line">Rows: actual crop. Columns: predicted crop (A–Z). Shade: share of the row's images. Hover a cell for details.</p></div>${topTable}`;

  const cv = $("cm"), ctx = cv.getContext("2d");
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const ramp = (dark ? CM_DARK : CM_LIGHT).map(hexRgb);
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--card-2").trim() || "#f7faf6";
  ctx.fillRect(0, 0, cv.width, cv.height);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = rates[i][j];
    if (!v || (errorsOnly && i === j)) continue;
    const t = Math.min(1, v / (max || 1)) * (ramp.length - 1), a = ramp[Math.floor(t)], b = ramp[Math.min(ramp.length - 1, Math.floor(t) + 1)], f = t % 1;
    ctx.fillStyle = `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(",")})`;
    ctx.fillRect(j * cell, i * cell, cell, cell);
  }
  if (errorsOnly) {
    ctx.fillStyle = css.getPropertyValue("--line").trim() || "#dbe4dc";
    for (let i = 0; i < n; i++) ctx.fillRect(i * cell + 3, i * cell + 3, 2, 2); // mark the hidden diagonal
  }
  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left) / r.width * n), i = Math.floor((e.clientY - r.top) / r.height * n);
    if (i < 0 || j < 0 || i >= n || j >= n) return hideTip();
    const cnt = matrix[i][j], rowN = matrix[i].reduce((a, b) => a + b, 0);
    showTip(`Actual <b>${esc(crops[i])}</b> → predicted <b>${esc(crops[j])}</b><br>${cnt} of ${rowN} images (${pct(rates[i][j])})${i === j ? " · correct" : ""}`, e.clientX, e.clientY);
  });
  cv.addEventListener("mouseleave", hideTip);
}

const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// ---- Helpers ------------------------------------------------------------------
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// Downloads a file while reporting progress (0-1), for the model weights
async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got / total, 1)); // the size header can be off if the server compresses
  }
  return new Uint8Array(await new Blob(chunks).arrayBuffer());
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(s);
  });
}

init().catch((e) => setStatus("error", `Failed to start: ${e.message || e}`));
