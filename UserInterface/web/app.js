// app.js - Plant disease identifier that runs fully in the browser.
//
// The models live in model/ as ONNX files, run with ONNX Runtime Web, so no
// server is needed and the photo never leaves the user's device.
// model/config.json lists the deployed models. While that list is empty the
// page still works (upload, samples, about) and says no model is deployed yet.
// Add ?demo to the URL to fill the results with random scores and check the layout.
//
// Flow: init() loads data/classes.json and model/config.json -> user picks an
//       image -> showImage() -> classify() -> preprocess() -> session.run()
//       -> scoreCompositions() -> renderPrediction() -> renderGallery().

// ---- Settings ---------------------------------------------------------------
const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const TOP_K = 5;
const GALLERY_PER_ROW = 6;
const DEMO = new URLSearchParams(location.search).has("demo");
const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

let classes;            // data/classes.json: crops, diseases, compositions, examples
let models = [];        // model/config.json -> models
let current = null;     // { cfg, session } of the selected model
let lastImg = null;     // last image element, re-classified when the model changes
let lastTruth = null;   // ground-truth composition id of a sample image, or null
let lastScores = null;  // composition scores of the last prediction
const sessions = new Map();
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

// ---- Start-up -----------------------------------------------------------------
async function init() {
  classes = await fetchJSON("data/classes.json");
  buildExamples();

  try {
    models = (await fetchJSON("model/config.json")).models || [];
  } catch {
    models = [];
  }

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

  // Model picker, shown only when there is more than one model
  $("modelSel").innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("");
  $("modelPick").hidden = models.length < 2;
  $("modelSel").addEventListener("change", (e) => selectModel(e.target.value));

  await loadScript(ORT_BASE + "ort.min.js");
  ort.env.wasm.wasmPaths = ORT_BASE;
  await selectModel(models[0].id);
}

// Loads (or reuses) the ONNX session for a model, then runs a warm-up
// prediction so the first real image is not slowed down by start-up work.
async function selectModel(id) {
  const cfg = models.find((m) => m.id === id);
  current = null;
  setStatus("loading", `Loading ${cfg.name}…`);
  try {
    let session = sessions.get(id);
    if (!session) {
      const bytes = await fetchWithProgress(cfg.file, (f) => { $("progress").style.width = `${Math.round(f * 100)}%`; });
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      const s = cfg.input_size;
      await session.run({ [inputName(cfg, session)]: new ort.Tensor("float32", new Float32Array(3 * s * s), [1, 3, s, s]) });
      sessions.set(id, session);
    }
    current = { cfg, session };
    setStatus("ready", `${cfg.name} ready`);
    renderModelInfo(cfg);
    if (lastImg) classify(lastImg);
  } catch (e) {
    setStatus("error", `Failed to load ${cfg.name}: ${e.message || e}`);
  }
}

// ---- Preprocessing ----------------------------------------------------------
// Image -> Float32Array in NCHW order, normalised with the model's mean / std.
// "squash" matches torchvision Resize((s, s)). "center_crop" matches
// Resize(s / crop_pct) + CenterCrop(s): a centred square of side
// min(w, h) * crop_pct is cut out and scaled to s x s.
function preprocess(img, cfg) {
  const s = cfg.input_size;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = s;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  const w = img.naturalWidth, h = img.naturalHeight;
  if (cfg.resize === "center_crop") {
    const side = Math.min(w, h) * (cfg.crop_pct ?? 0.875);
    ctx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, s, s);
  } else {
    ctx.drawImage(img, 0, 0, w, h, 0, 0, s, s);
  }
  const px = ctx.getImageData(0, 0, s, s).data;
  const mean = cfg.mean ?? IMAGENET.mean, std = cfg.std ?? IMAGENET.std;
  const out = new Float32Array(3 * s * s);
  for (let i = 0, n = s * s; i < n; i++) {
    for (let c = 0; c < 3; c++) out[c * n + i] = (px[i * 4 + c] / 255 - mean[c]) / std[c];
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

// ---- Prediction ---------------------------------------------------------------
async function classify(img) {
  lastImg = img;
  $("result").hidden = false;
  renderTruth(null);

  if (!current && !DEMO) {
    // A model is listed but still loading or failed: keep the last state
    $("noModel").hidden = models.length > 0;
    $("preds").hidden = true;
    $("gallery").hidden = true;
    if (models.length) setStatus("loading", "Model is still loading, the image will be classified when it is ready");
    return;
  }

  let scores;
  if (current) {
    const { cfg, session } = current;
    const s = cfg.input_size;
    const feeds = { [inputName(cfg, session)]: new ort.Tensor("float32", preprocess(img, cfg), [1, 3, s, s]) };
    const res = await session.run(feeds);
    const names = cfg.outputs ?? {};
    const pick = (key) => (names[key] && res[names[key]] ? Array.from(res[names[key]].data) : null);
    scores = scoreCompositions({ crop: pick("crop"), disease: pick("disease"), composition: pick("composition") }, cfg);
  } else {
    scores = demoScores();
  }
  lastScores = scores;
  renderPrediction(scores);
}

function compName(k) {
  const c = classes.compositions[k];
  return `${classes.crops[c.crop]} · ${classes.diseases[c.disease]}`;
}

function renderPrediction({ comp, crop, dis }) {
  $("noModel").hidden = true;
  $("preds").hidden = false;

  const top = comp.map((p, k) => [k, p]).sort((a, b) => b[1] - a[1]).slice(0, TOP_K);
  const [bestK, bestP] = top[0];
  const best = classes.compositions[bestK];
  const pct = bestP * 100;

  $("best").textContent = compName(bestK);
  $("bestTags").innerHTML =
    `<span class="tag">Crop: ${esc(classes.crops[best.crop])}</span>` +
    `<span class="tag">Disease: ${esc(classes.diseases[best.disease])}</span>` +
    (best.seen
      ? `<span class="tag seen">Seen in training</span>`
      : `<span class="tag unseen">Unseen combination (zero-shot)</span>`);
  $("bestConf").textContent = `${pct.toFixed(1)}%`;
  const level = pct >= 70 ? ["high", "High confidence"] : pct >= 40 ? ["mid", "Medium confidence"] : ["low", "Low confidence"];
  $("badge").className = `badge ${level[0]}`;
  $("badge").textContent = level[1];

  const ci = argmax(crop), di = argmax(dis);
  $("heads").textContent =
    `Crop on its own: ${classes.crops[ci]} (${(crop[ci] * 100).toFixed(1)}%) · ` +
    `Disease on its own: ${classes.diseases[di]} (${(dis[di] * 100).toFixed(1)}%)`;

  // Candidate list, including the top match, so any of them can drive the gallery
  $("top5").innerHTML = top.slice(1).map(([k, p]) => {
    const tag = classes.compositions[k].seen ? "" : `<span class="mini unseen">unseen</span>`;
    return `<li><button type="button" data-k="${k}" aria-pressed="false">` +
      `<span class="name">${esc(compName(k))}${tag}</span><span class="pct">${(p * 100).toFixed(1)}%</span>` +
      `<span class="bar"><i data-w="${p * 100}"></i></span></button></li>`;
  }).join("");
  $("top5").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const on = b.getAttribute("aria-pressed") !== "true";
    $("top5").querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", String(on));
    renderGallery(on ? Number(b.dataset.k) : bestK, !on);
  }));

  // Bars start at 0% and grow to their value, so every prediction animates
  const bars = [$("bestBar"), ...$("top5").querySelectorAll(".bar>i")];
  bars.forEach((b) => { b.style.width = "0%"; });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("bestBar").style.width = `${pct}%`;
    $("top5").querySelectorAll(".bar>i").forEach((b) => { b.style.width = `${b.dataset.w}%`; });
  }));

  renderTruth(bestK);
  renderGallery(bestK, true);
}

// For sample images: shows the true label and whether the top match is right
function renderTruth(predK) {
  const el = $("truth");
  if (lastTruth === null) { el.hidden = true; return; }
  el.hidden = false;
  let html = `True label: <b>${esc(compName(lastTruth))}</b>`;
  if (predK !== null) html += predK === lastTruth ? ` · <b class="ok">top match correct</b>` : ` · <b class="bad">top match wrong</b>`;
  el.innerHTML = html;
}

// ---- Matching training images -------------------------------------------------
// Seen composition: its own training images. Unseen composition: it has none,
// so show training images that share its crop and ones that share its disease.
function renderGallery(k, isTop) {
  const c = classes.compositions[k];
  const name = compName(k);
  const rows = [];

  if (c.seen) {
    $("galleryTitle").textContent = `Training images of ${name}`;
    $("galleryNote").textContent =
      `${isTop ? "Top match" : "Selected candidate"}. ${c.n_train} training image${c.n_train === 1 ? "" : "s"} in this class` +
      (c.n_train > c.thumbs.length ? `, ${c.thumbs.length} shown.` : ".");
    rows.push({ title: "", items: c.thumbs.map((t) => ({ src: t, cap: name })) });
  } else {
    $("galleryTitle").textContent = `Related training images for ${name}`;
    $("galleryNote").textContent =
      "This combination never appears in the training images. The model predicts it by combining what it learnt " +
      "about the crop and about the disease, shown below.";
    const related = (field) => classes.compositions
      .filter((o) => o.seen && o[field] === c[field])
      .slice(0, GALLERY_PER_ROW)
      .map((o) => ({ src: o.thumbs[0], cap: compName(o.id) }));
    rows.push({ title: `Same crop: ${classes.crops[c.crop]}`, items: related("crop") });
    rows.push({ title: `Same disease: ${classes.diseases[c.disease]}`, items: related("disease") });
  }

  $("galleryRows").innerHTML = rows.map((r) =>
    `<div class="g-row">${r.title ? `<h4>${esc(r.title)}</h4>` : ""}` +
    (r.items.length
      ? `<div class="grid">${r.items.map((it) =>
          `<figure><img src="${esc(it.src)}" alt="${esc(it.cap)}" loading="lazy"><figcaption>${esc(it.cap)}</figcaption></figure>`).join("")}</div>`
      : `<p class="note">No training images share this.</p>`) +
    `</div>`).join("");
  $("gallery").hidden = false;
}

// ---- Model details, metrics and status ------------------------------------------
function renderModelInfo(cfg) {
  const m = $("metrics");
  if (cfg && cfg.metrics && cfg.metrics.length) {
    m.innerHTML = `<div><b>${esc(cfg.short_name ?? cfg.name)}</b><span>deployed model</span></div>` +
      cfg.metrics.map((x) => `<div><b>${esc(x.value)}</b><span>${esc(x.label)}</span></div>`).join("");
  } else {
    m.innerHTML = `<p>${cfg ? "No test-set results listed for this model." : "Test-set results will appear here once a model is deployed."}</p>`;
  }
  $("aboutModelHead").textContent = cfg ? cfg.name : "Selected model";
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
// Shows the image and classifies a separate, unstyled copy at full resolution
function showImage(src, truth = null) {
  lastTruth = truth;
  $("preview").src = src;
  const raw = new Image();
  raw.onload = () => classify(raw).catch((e) => setStatus("error", `Prediction failed: ${e.message || e}`));
  raw.src = src;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function openFile(f) {
  if (f && f.type.startsWith("image/")) showImage(URL.createObjectURL(f));
}

$("file").addEventListener("change", (e) => openFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); }
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  openFile(e.dataTransfer.files[0]);
});

$("uploadNew").addEventListener("click", () => $("file").click());
$("reset").addEventListener("click", () => {
  $("result").hidden = true;
  $("gallery").hidden = true;
  $("preview").removeAttribute("src");
  $("file").value = "";
  lastImg = lastTruth = lastScores = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
});
$("aboutBtn").addEventListener("click", () => $("about").showModal());
$("about").addEventListener("click", (e) => { if (e.target === $("about")) $("about").close(); });

function buildExamples() {
  $("examples").innerHTML = classes.examples.map((ex, i) =>
    `<button type="button" data-i="${i}"><img src="${esc(ex.file)}" alt="${esc(compName(ex.truth))}" loading="lazy">` +
    `<span>${ex.split === "unseen" ? "Unseen" : "Seen"} · ${esc(compName(ex.truth))}</span></button>`).join("");
  $("examples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const ex = classes.examples[Number(b.dataset.i)];
    showImage(ex.file, ex.truth);
  }));
}

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
