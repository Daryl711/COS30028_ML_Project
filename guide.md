# Experiment guide

Three stages, run in order. Every run uses the recipe in **`training_scheme.md`** (data split, augmentation,
AdamW wd 0.05, LR 1e-3 for new modules, batch 64, 30 epochs, 2 warm-up + cosine, AMP, grad clip 1.0,
label smoothing 0.1, early stopping patience 8, seed 42, same evaluation). Only the model and the
method-specific terms change between rows, so every row in `results.md` is a fair comparison.

### Decision: one LR (1e-3) for every new module

All new modules (heads, text classifiers, adapters, prompt tokens, discriminator) use LR 1e-3 in every run.
This overrides the 5e-4 in `PROPOSED_METHOD.md` (`training_scheme.md` §6.2).

- **Why:** 1e-3 is the head LR of the Stage 1 baseline, `stage1/stage1a/config.yaml`
  (`lr_head: 1.0e-3`, with `lr_backbone: 2.0e-4` for the fine-tuned EfficientNet). Every later stage
  builds on that baseline, so it keeps the same recipe. With the LR held fixed, a difference between
  rows comes from the module, not from per-run tuning, and no run gets an extra hyperparameter search
  the others don't.
- **Limit:** 1e-3 was carried over from the baseline, not chosen in an LR sweep (the EffB0 README lists
  it as already matching the scheme). It also comes from linear heads trained next to a fine-tuned
  backbone, while Stage 2 and 3 modules sit on a frozen one. With Adam, the step size is roughly the
  same for every parameter, so how large the step is relative to the parameter depends on that
  parameter's scale. The R4 prompt tokens have a per-dimension std of about 0.011, which makes 1e-3 a
  large step for them (R4 without λ_kg drifts to ‖T − T⁰‖² ≈ 1.1–1.4 by epoch 5). So for those modules
  1e-3 is a convention that keeps rows comparable, not a tuned value.
- **Rule:** a different LR for one module (for example 1e-4 on the R4 tokens) is a named sweep sub-row
  (`training_scheme.md` §6.1). It is reported next to the main row and does not replace it. If a sweep
  LR wins on the val selection score, write that in `results.md`. If it is adopted, it applies to that
  module in every run that contains it (R4, R4 + R5, All).

Run every command from the repo root.

```bash
# GPU (CUDA 12.6)
python -m pip install --force-reinstall --no-deps torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r stage3/stage3b/requirements.txt
```

| Stage | Question it answers | Experiments | Output names |
|---|---|---|---|
| 1 | How well does a plain CNN do, and does sharing a backbone help? | EffB0 multi-task, EffB0 single-task | `stage1a`, `stage1b_crop` + `stage1b_disease` |
| 2 | Does frozen BioCLIP beat the CNN? Do text classifiers help? Which text? Does logit adjustment help? | 2a image only, 2b taxonomy, 2c taxonomy + LLM, 2d = 2b + R8a, 2e = 2c + R8a | `stage2a`, `stage2b`, `stage2c`, `stage2d`, `stage2e` |
| 3 | Which DISCO modules help on the taxonomy-text base? | 3a R4, 3b R5, 3c R5+R6, 3d R4+R5, 3e R4+R5+R6 (All) | `stage3a`, `stage3b`, `stage3c`, `stage3d`, `stage3e` |

Method details and citations are in `PROPOSED_METHOD.md`: R3 §3.3, R4 §4.5, R5 §3.2, R6 §3.4 and §4.4,
R8a §3.7 and §4.1.

---

## Stage 1: EfficientNet-B0 baselines (no improvements)

Plain EfficientNet-B0 (ImageNet weights, fine-tuned), linear heads, no logit adjustment, no calibration
tuning, no hyperparameter search.

```
image (224) → EfficientNet-B0 → 1280-d ─┬─ Dropout 0.3 → Linear → 55 crop logits
                                         └─ Dropout 0.3 → Linear → 175 disease logits
```

| Run | Model |
|---|---|
| **1a Multi-task** | one backbone, two heads, loss = crop CE + disease CE |
| **1b Single-task** | two independent backbones, one head each; their log-probs are added at evaluation |

Config: `stage1/stage1a/config.yaml` (scheme values `weight_decay: 5.0e-2`, `grad_clip: 1.0`; backbone LR
2e-4 and `drop_path_rate` 0.2 are backbone-specific). 1b runs the same code through `stage1/stage1b/`.

```bash
# 1a multi-task
python stage1/stage1a/train.py
python stage1/stage1a/evaluate.py --checkpoints outputs/stage1a/best.pt --out outputs/stage1a/eval

# 1b single-task (independent)
python stage1/stage1b/train.py --mode crop       # -> outputs/stage1b_crop
python stage1/stage1b/train.py --mode disease    # -> outputs/stage1b_disease
python stage1/stage1b/evaluate.py                # -> outputs/stage1b/eval
```

What to record: seen/unseen composition top-1/top-5, crop and disease accuracy per split, HM, AUSUC.
Expected pattern (from earlier runs): seen ≈ 68 %, unseen composition ≈ 5–9 %, unseen disease ≈ 10 %.
Stage 1 is the reference that Stages 2 and 3 must beat.

---

## Stage 2: BioCLIP baselines (five runs)

Frozen BioCLIP (image and text encoders), **no adapters**. Only the heads train. Stage 2 asks three
things: does BioCLIP's image feature beat the CNN, do text classifiers help, and does logit adjustment
help.

| Run | Heads | Text (R3) | R8a logit adjustment |
|---|---|---|---|
| **2a Image only** (R1) | linear | none | no |
| **2b Taxonomy** (R2 + R3 taxonomy) | cosine, text-initialised | taxonomy | no |
| **2c Taxonomy + LLM** (R2 + R3 full) | cosine, text-initialised | taxonomy + LLM symptoms | no |
| **2d** = 2b + R8a | as 2b | taxonomy | yes |
| **2e** = 2c + R8a | as 2c | taxonomy + LLM symptoms | yes |

### 2a: image only (no text)

```
image → BioCLIP image encoder (frozen) → v (512) → N(v) ─┬─ Dropout 0.3 → Linear → 55 crop logits
                                                          └─ Dropout 0.3 → Linear → 175 disease logits
```

A linear probe on BioCLIP features. It isolates the effect of the backbone from the effect of text.

### 2b, 2c: text classifiers

```
image → BioCLIP image encoder (frozen) → v (512) → N(v) ─┬─ Dropout 0.3 → cos(·, T_c) · logit_scale → 55 crop logits
                                                          └─ Dropout 0.3 → cos(·, T_d) · logit_scale → 175 disease logits
T_c, T_d: initialised from BioCLIP text embeddings (text encoder frozen), then trained at LR 1e-3
```

The heads are the text embeddings, so a disease is a crop-agnostic point in text space. Epoch 0 is exactly
BioCLIP zero-shot with that text; log it as a sub-row.

The two runs differ **only in the text used to build T_c, T_d**:

| Run | Text per class | Source |
|---|---|---|
| **2b Taxonomy** | "a photo of a {crop} leaf, {taxonomy}" / "a photo of a leaf with {disease}, {taxonomy}" | DPD repo `text_description/plant_descriptions.csv`, `diseases_descriptions.csv` (save under `text/`) |
| **2c Taxonomy + LLM symptoms** | 2b embedding blended with the mean of K = 20 LLM descriptions: `T = N((1−η)·N(E_t(prompt)) + η·mean_j N(E_t(desc_j)))`, η = 0.3 | `text/descriptions.json`, generated once with the prompt in `PROPOSED_METHOD.md` §3.3.1 |

Rules for the LLM descriptions (2c):
- One call per crop and per disease, never per pair.
- Disease descriptions are purely visual (lesion shape, colour, margin, texture) and **name no crop**.
- Crop descriptions describe leaf morphology (shape, venation, margin, arrangement).
- Commit `text/descriptions.json` before training so every later run uses the same text.

### 2d, 2e: + R8a logit adjustment

Same as 2b / 2c, with the logit-adjusted CE of Menon et al. (τ_LA = 1) on both heads:

```
CE( s + τ_LA · log π , y )      π = class frequency in the training split (after the val split)
```

The prior is added **only in the loss**. Prediction and evaluation use the raw scores. Do not add a
class-balanced sampler as well.

### Decisions from Stage 2 (on val, never test)

- **Text for Stage 3: taxonomy (R3 taxonomy, as in 2b).** This is fixed; 2c is reported as a finding.
- **Logit adjustment for Stage 3:** use R8a in every Stage 3 run if **2d beats 2b** on the val selection
  score; otherwise leave it out of every Stage 3 run. Write the decision and both val scores in
  `results.md` before starting Stage 3.

What to record: everything from Stage 1, the epoch-0 zero-shot numbers for 2b and 2c, and for 2d/2e
accuracy by frequency bucket (≤ 5, 6–50, > 50 training images per composition).

---

## Stage 3: DISCO modules on the Stage 3 base

**Stage 3 base** = 2b (frozen BioCLIP, taxonomy text heads), plus R8a if Stage 2 chose it. Each run adds
module(s) to that base and changes nothing else. Stage 3 is where adapters first appear.

| Run | Adds | What it changes | Expected effect |
|---|---|---|---|
| **R4** | CoOp context (M = 4) + CSP primitive tokens, KgCoOp anchor λ_kg = 8 | T_c, T_d come from learnable prompts (initialised from the taxonomy text) through the frozen text encoder; `L += λ_kg · (‖T_c − T_c⁰‖² + ‖T_d − T_d⁰‖²)` | better seen fit without unseen collapse |
| **R5** | Separate adapters A_c, A_d | `f_c = N(v + A_c(v))` feeds the crop head, `f_d = N(v + A_d(v))` feeds the disease head (512 → 256 → 512, zero-init, as in `stage3/stage3b/model.py`) | crop and disease get separate subspaces |
| **R5 + R6** | R5 + conditional crop-invariance (GRL) | crop discriminator on f_d behind a gradient-reversal layer, softmax restricted to the crops that carry the image's disease, only for diseases on ≥ 2 crops; λ_max = 0.1 with the DANN ramp | crop leakage in f_d drops, unseen disease accuracy rises, seen drops slightly |
| **R4 + R5** | R4 + R5 | prompts + separate adapters | tests whether the two add up |
| **All (R4 + R5 + R6)** | all three | full Stage 3 model | best unseen / HM |

Notes:
- **R6 always needs R5.** The GRL acts on the disease adapter's output f_d; without A_d there is no
  trainable disease feature for it to change. So there is no "R6 alone" run.
- R5 starts exactly at the Stage 3 base: the zero-initialised adapters are the identity at epoch 0.
- R6 needs the multi-crop diseases (`|C(d)| ≥ 2`, 42 of 175). Compute the list from the training split only.
- New modules (adapters, prompt tokens, discriminator) all use LR 1e-3 and wd 0.05.
- R4 runs the frozen text encoder forward each step (gradients flow to the prompt tokens), recomputing
  T_c, T_d for all 230 classes once per step. Keep batch 64.

What to record, beyond the Stage 1 metrics:
- **Crop-leakage probe** (`PROPOSED_METHOD.md` §6.5) on the disease feature for the Stage 3 base, R5,
  R5 + R6 and All. For the base and R4 the probed feature is N(v).
- For R4, a sub-row without λ_kg (shows the prompt overfitting the anchor prevents).

---

## Results table (fill from `eval/metrics.json`)

| Stage | Run | Seen T1 | Unseen T1 | HM T1 | Seen T5 | Unseen T5 | Crop acc (U) | Dis acc (U) | Bal. unseen | AUSUC | Leak ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1a EffB0 multi-task | | | | | | | | | | |
| 1 | 1b EffB0 single-task | | | | | | | | | | |
| 2 | 2a BioCLIP image only | | | | | | | | | | |
| 2 | 2b BioCLIP + taxonomy | | | | | | | | | | |
| 2 | 2c BioCLIP + taxonomy + LLM | | | | | | | | | | |
| 2 | 2d = 2b + R8a | | | | | | | | | | |
| 2 | 2e = 2c + R8a | | | | | | | | | | |
| 3 | R4 | | | | | | | | | | |
| 3 | R5 | | | | | | | | | | |
| 3 | R5 + R6 | | | | | | | | | | |
| 3 | R4 + R5 | | | | | | | | | | |
| 3 | All (R4 + R5 + R6) | | | | | | | | | | |

## Evaluation output files

| File | What it shows |
|---|---|
| `eval/metrics.json` | all metrics above |
| `confusion_seen_crops.png` | all 55 crops, cells as percentages, row labels with test-image counts |
| `confusion_seen_crops_worst.png` | the 15 worst crops (≥ 5 test images), percentage and count per cell |
| `top_confusions_seen.png` | the 20 most common mistakes, e.g. "paddy \| healthy → blast: 20 of 32 images" |

## Report

- Results: fair comparison (same scheme for every row).
- Observations: what changed between rows, seen vs unseen, crop vs disease.
- Explanation: why, and what could be improved.
- Both quantitative (tables, curves) and qualitative (confusions, Grad-CAM, t-SNE).
