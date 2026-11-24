# Results: Compositional Zero-Shot Plant Disease Classification (DPD-Small)

This file compares the evaluated runs in `outputs/`. The main comparison covers these five:

| Run | Approach | Source |
|---|---|---|
| **S3 R5** | Frozen BioCLIP, taxonomy-text heads (Stage 2b base) plus separate crop/disease residual adapters A_c, A_d (Stage 3) | `outputs/s3_r5/eval/metrics.json` |
| **S3 R5 + R6** | S3 R5 plus a conditional crop discriminator on the disease feature behind a gradient-reversal layer (λ_max = 0.1) | `outputs/s3_r5r6/eval/metrics.json` |
| **EffB0 multi-task** | One EfficientNet-B0 backbone with two heads | `outputs/multitask_effb0/eval/metrics.json` |
| **EffB0 single-task** | Two independent EfficientNet-B0 models (crop, disease) | `outputs/SingleTask_effb0/eval/metrics.json` |
| **EffB0 single-task + logit tuning** | Same checkpoints as single-task, plus logit adjustment (per-head temperature/τ, pair prior, cross-fitted) | `outputs/FineTunedB0(Logit+Hyperparameter)/eval/metrics.json` |

Stage 3 R4 (learnable prompts, local evaluator) and its token-LR sweep are in [Stage 3 R4](#stage-3-r4-learnable-prompts-on-the-taxonomy-text-base), and the combined R4 + R5 and All (R4 + R5 + R6) runs are in [Stage 3 R4 + R5 and All](#stage-3-r4--r5-and-all-r4--r5--r6). The Stage 1 EffB0 re-runs and the zero-shot CLIP/BioCLIP runs are in [Stage 1 EffB0 re-runs](#stage-1-effb0-re-runs) and [Zero-shot CLIP and BioCLIP](#zero-shot-clip-and-bioclip).

All runs use the same test splits: `test_seen` has 2000 images and `test_unseen` has 1448. Compositions are scored as `log p(crop|x) + log p(disease|x)` over the 333 valid pairs (323 seen, 10 unseen).

## Main comparison

All values are percentages.

| Metric | **S3 R5** | **S3 R5 + R6** | EffB0 multi-task | EffB0 single-task | EffB0 single-task + logit tuning |
|---|---|---|---|---|---|
| **Seen** crop top-1 | 91.7 | **92.1** | 90.1 | 90.1 | 90.1 |
| Seen disease top-1 | 67.5 | 68.6 | **69.4** | 68.9 | 69.0 |
| Seen comp top-1 | 65.7 | 67.0 | 67.9 | 68.4 | **68.5** |
| Seen comp top-5 | 89.6 | **89.7** | 88.4 | 88.6 | 88.6 |
| **Unseen** crop top-1 | 70.2 | 69.8 | 66.9 | **72.1** | **72.1** |
| Unseen disease top-1 | 10.2 | 10.7 | **10.8** | 9.9 | 9.9 |
| Unseen comp top-1 | 5.2 | 5.5 | 4.9 | **8.5** | **8.5** |
| Unseen comp top-5 | 29.1 | 30.7 | 30.0 | **35.4** | **35.4** |
| Unseen class-balanced top-1 | 7.3 | 7.9 | — | — | — |
| HM top-1 (uncalibrated) | 9.7 | 10.2 | 9.1 | **15.1** | **15.1** |
| HM top-5 (uncalibrated) | 44.0 | 45.7 | 44.8 | **50.6** | **50.6** |
| Calibrated best HM (bias sweep) | 60.1 | 59.5 | 60.7 | 61.1 | **61.2** |
| Seen / unseen acc at best HM | 54.4 / 67.3 | 56.9 / 62.4 | 58.3 / 63.4 | 59.0 / 63.3 | 59.1 / 63.5 |
| Best unseen bias | 7.26 | 6.93 | 6.46 | 6.43 | 6.43 |
| AUSUC | **49.0** | 48.0 | 47.1 | 48.8 | 48.9 |

### Advanced calibration (EffB0 single-task + logit tuning only)

| Setting | Seen top-1 | Unseen top-1 | HM top-1 | Seen top-5 | Unseen top-5 | HM top-5 |
|---|---|---|---|---|---|---|
| Tuned on test (T_c=1.0, T_d=1.5, bias=4.83) | 59.5 | 71.3 | **64.9** | 86.5 | 91.2 | **88.8** |
| Cross-fitted (held-out) | 59.8 | 70.3 | 64.6 | 86.6 | 91.0 | 88.7 |

### Closed-world unseen composition (EffB0 runs only)

In the closed-world setting the model chooses only among the 10 unseen pairs.

| Run | Top-1 | Top-5 |
|---|---|---|
| EffB0 multi-task | 75.4 | 96.1 |
| EffB0 single-task | 77.3 | 95.4 |
| EffB0 single-task + logit tuning | 77.3 | 95.4 |

The S3 evaluator (R5 and R5 + R6) does not report closed-world numbers.

## Stage 1 EffB0 re-runs

All values are percentages and come from each run's `eval/metrics.json`. All runs are seed 42.

| Run | Approach | Source |
|---|---|---|
| S1 EffB0 MT | Stage 1 re-run of EffB0 multi-task (`weight_decay` 0.05, `grad_clip` 1.0) | `outputs/s1_mt_effb0/` |
| S1 EffB0 indep | Stage 1 re-run of independent EffB0 (`s1_crop_effb0` + `s1_disease_effb0`) | `outputs/s1_indep_effb0/` |

| Metric | S1 EffB0 MT | S1 EffB0 indep |
|---|---|---|
| **Seen** crop top-1 | 89.4 | 90.1 |
| Seen disease top-1 | 68.6 | 68.6 |
| Seen comp top-1 | 67.1 | 67.6 |
| Seen comp top-5 | 87.6 | 88.5 |
| **Unseen** crop top-1 | 68.5 | 72.5 |
| Unseen disease top-1 | 8.4 | 9.9 |
| Unseen comp top-1 | 4.1 | 8.6 |
| Unseen comp top-5 | 27.6 | 34.8 |
| HM top-1 (uncalibrated) | 7.8 | 15.2 |
| HM top-5 (uncalibrated) | 42.0 | 50.0 |
| Calibrated best HM (bias sweep) | 60.3 | 60.9 |
| Seen / unseen acc at best HM | 58.2 / 62.6 | 57.1 / 65.3 |
| Best unseen bias | 6.58 | 6.60 |
| AUSUC | 46.7 | 48.2 |
| Closed-world unseen top-1 / top-5 | 75.7 / 95.9 | 76.9 / 94.3 |

### What these runs show
- **The Stage 1 EffB0 re-runs reproduce the multi-task vs independent gap** (Finding 1): unseen comp top-1 4.1 vs 8.6, HM top-1 7.8 vs 15.2, AUSUC 46.7 vs 48.2. With the changed `weight_decay` and gradient clipping the numbers are within about 1 point of the original EffB0 runs.

## Zero-shot CLIP and BioCLIP

No training: composition scores come from the pretrained image–text similarity. Sources: `outputs/zeroshot_bioclip/eval/metrics.json` (`hf-hub:imageomics/bioclip`) and `outputs/zeroshot_clip/eval/metrics.json` (OpenAI `ViT-B-16-quickgelu`). Values are percentages.

| Metric | BioCLIP zero-shot | CLIP zero-shot |
|---|---|---|
| Seen crop / disease top-1 | 35.0 / 2.8 | 34.0 / 2.1 |
| Seen comp top-1 / top-5 | 6.2 / 25.7 | 7.2 / 22.0 |
| Unseen crop / disease top-1 | 32.5 / **25.7** | 24.8 / 2.4 |
| Unseen comp top-1 / top-5 | **27.4** / **48.3** | 3.3 / 17.6 |
| HM top-1 / top-5 (uncalibrated) | 10.0 / 33.5 | 4.5 / 19.6 |
| Calibrated best HM / AUSUC | 10.5 / 3.6 | 9.7 / 2.7 |
| Closed-world unseen top-1 / top-5 | 65.2 / 93.1 | 46.8 / 95.3 |

- **Zero-shot BioCLIP has the best unseen disease accuracy of any run.** Unseen disease top-1 is 25.7% and unseen comp top-1 27.4%, against 8.4–10.8% and 4.1–8.6% for every trained model. Seen disease top-1 is only 2.8%, though, so its AUSUC (3.6) is far below every trained model.
- **Generic CLIP has no usable disease knowledge** (2.1–2.4% disease top-1 on both splits), so BioCLIP's unseen disease result comes from its biology pretraining.
- These numbers differ from the Stage 2 "Zero-shot" column below (unseen T1 20.9 there vs 27.4 here) because that column uses the Colab notebook's evaluator and taxonomy prompts at epoch 0.

## Findings

### 1. With one backbone, EffB0 multi-task does worse than single-task (independent) EffB0
- Seen accuracy is about the same for both (composition top-1 67.9% vs 68.4%).
- On unseen data the multi-task model is clearly worse: crop top-1 is 66.9% vs 72.1%, composition top-1 is 4.9% vs 8.5%, and HM top-1 is 9.1 vs 15.1.
- A likely cause is that the shared features learn which crops and diseases appear together in training, which hurts on new pairings. Two separate models cannot learn those pairings.
- The Stage 1 EffB0 re-runs show the same gap (HM top-1 7.8 vs 15.2).

### 2. After calibration the gaps mostly close
- With the plain scalar-bias sweep, every model reaches a best HM of about 60–61.
- AUSUC is close for S3 R5 and single-task EffB0 (48.8–49.0) and lower for EffB0 multi-task (47.1; 46.7 for the Stage 1 re-run).
- The tuned EffB0 run reaches **64.9 HM top-1**, the best result overall. That number comes from tuning the calibration settings on the test sets. Its cross-fitted estimate, which tunes on one part of the test data and scores on the rest, is 64.6, so the gain is real. S3 R5 has not had this tuning yet, so this comparison is not like-for-like.

### 3. Disease recognition on unseen pairs is the bottleneck for every trained model
- Unseen disease top-1 is 8.4–10.8% for every trained model evaluated with the local scripts (6.1–10.2% for the Stage 2 Colab runs), while unseen crop top-1 is 67–73%.
- The disease head predicts the diseases it saw with each crop during training. None of these models solves that.
- Zero-shot BioCLIP, with no training, gets 25.7% unseen disease top-1, so training on the seen pairs destroys disease knowledge that transfers to unseen pairs.
- Closed-world unseen accuracy is about 75–77%. When the model only has to choose among the unseen pairs it does well; it fails because seen pairs outscore the correct unseen pair.

## Stage 3 R5: separate adapters on the taxonomy-text base

R5 adds two zero-initialised residual adapters (512 → 256 → 512) on the frozen BioCLIP embedding v: `f_c = N(v + A_c(v))` feeds the crop text head, `f_d = N(v + A_d(v))` feeds the disease text head. Best checkpoint is epoch 19 (early stop at epoch 27).

| | Value |
|---|---|
| Val crop / disease top-1 | 97.8 / 85.4 |
| Val selection score | **91.58** (Stage 2b base: 89.57, +2.0) |

### Crop-leakage probe on the disease feature

A 5-fold logistic-regression probe predicts the crop from the disease feature, on the 428 **validation** images whose disease occurs on ≥ 2 crops (30 such diseases in val; 42 multi-crop diseases in train). `leak = (probe_acc − chance) / (1 − chance)`, so 0 = no more crop information than the disease alone gives.

**Baseline (chance) = majority crop per disease**: for each disease, the crop it most often occurs on in the *training* split, scored on the 428 val images. This reaches **53.3%** with no crop evidence at all. The old uniform baseline `mean 1/|C(d)|` (20.5%) ignored that crops are unevenly split within a disease, so it overstated leakage; it is kept as `leak_uniform` in `metrics.json`.

| Feature | Probe acc | Leak ↓ (majority baseline) | Leak (old uniform baseline) |
|---|---|---|---|
| Base N(v) (frozen BioCLIP) | 74.6 ± 8.4 | 0.46 ± 0.18 | 0.68 ± 0.11 |
| R5 f_d | 52.4 ± 8.8 | −0.02 ± 0.19 | 0.40 ± 0.11 |
| R5 + R6 f_d | 48.2 ± 10.0 | **−0.11 ± 0.21** | 0.35 ± 0.13 |

### What R5 shows
- **Validation improves, test unseen does not.** R5 beats the 2b base on the val selection score (+2.0), but val only contains seen pairs. On test it has the best seen crop top-1 (91.7) and seen comp top-5 (89.6) of all runs, yet unseen comp top-1 is 5.2% and HM top-1 is 9.7, close to EffB0 multi-task (4.9 / 9.1) and below single-task EffB0 (8.5 / 15.1).
- **Strongest seen bias of any run.** It needs the largest calibration bias (7.26 vs about 6.4–6.5) and reaches the lowest calibrated best HM (60.1). AUSUC is nevertheless the highest (49.0, marginally above tuned EffB0 at 48.9), so the full seen/unseen trade-off curve is competitive; the uncalibrated operating point is what is poor.
- **The R5 disease adapter already removes all measurable crop leakage, yet unseen disease accuracy does not improve.** Against the majority baseline, the frozen BioCLIP feature leaks (0.46), but R5's f_d is at zero (−0.02 ± 0.19): a linear probe on f_d cannot beat guessing the disease's majority crop. Unseen disease top-1 still stays at 10.2%, the same 10–11% bottleneck every model shows (Finding 3). So, at least as this probe measures it, crop information inside f_d is not what drives the unseen failure. This weakens hypothesis H1 and leaves R6 (which targets exactly this leakage) with no measurable leakage to remove.
- **The R5 effect cannot be isolated yet.** There is no test-set eval for the 2b base, so it is unknown how much of R5's test behaviour comes from the text heads (2b) versus the separate adapters (R5). Evaluating 2b with `stage3/s3_r5/evaluate.py`'s protocol is needed for that comparison.

## Stage 3 R5 + R6: conditional crop-invariance on the disease feature

R5 + R6 keeps R5's model and adds a crop discriminator D (512 → 256 → 55) on f_d behind a gradient-reversal layer. D's softmax is restricted to C(d), the crops that carry the image's disease in train, and it only sees images whose disease occurs on ≥ 2 crops (42 diseases). Loss = crop CE + disease CE + L_inv, with λ ramping to λ_max = 0.1. Prediction and evaluation are identical to R5. Seed 42; ran all 30 epochs, best checkpoint at epoch 23.

| | R5 | R5 + R6 |
|---|---|---|
| Val crop / disease top-1 | 97.8 / 85.4 | 97.9 / 84.8 |
| Val selection score | **91.58** | 91.36 (−0.22) |
| Test HM top-1 / top-5 (uncalibrated) | 9.7 / 44.0 | **10.2 / 45.7** |
| Calibrated best HM / AUSUC | **60.1 / 49.0** | 59.5 / 48.0 |
| f_d leak (majority baseline) | −0.02 ± 0.19 | −0.11 ± 0.21 |

### What R5 + R6 shows
- **It loses on the selection rule.** Val selection score is 91.36 vs R5's 91.58, so under `guide.md` R5 stays the preferred checkpoint. The difference (0.2 points, one seed) is within noise.
- **Small uncalibrated test gains, all within noise.** Seen comp top-1 +1.3 (65.7 → 67.0), unseen disease top-1 +0.5 (10.2 → 10.7), unseen comp top-1 +0.3 (5.2 → 5.5, about 4 images), HM top-1 +0.5. The seen bias is a little smaller (best bias 6.93 vs 7.26).
- **The calibrated trade-off is slightly worse.** Best HM falls from 60.1 to 59.5 and AUSUC from 49.0 to 48.0, so across the full seen/unseen curve R6 does not help.
- **The leakage result confirms the R5 finding.** R6 pushes f_d's probe accuracy from 52.4% to 48.2%, below the 53.3% majority baseline (leak −0.11 ± 0.21, inside the noise around 0). Removing what little crop information remained did not move unseen disease top-1 out of the 10–11% band. Crop leakage in f_d, as this probe measures it, is not what limits unseen accuracy, so H1 is not supported.
- The λ sweep and the unconditional-DANN ablation are in the next section.

## Stage 3 R5 + R6 ablations: λ sweep and unconditional DANN

Sources: `outputs/s3_r5r6_lam0/`, `outputs/s3_r5r6_lam0.03/`, `outputs/s3_r5r6_lam0.3/`, `outputs/s3_r5r6_uncond/` (each `eval/metrics.json`), summarised in `outputs/s3_r5r6_sweep_summary.md`. All runs use seed 42, a 30-epoch budget and the same evaluator as R5 (333 valid pairs). The only setting that changes is `lambda_max`, plus `conditional: false` for the DANN run. With `conditional: false`, D's softmax covers all 55 crops instead of only C(d). Values are percentages, except the leak columns.

| Metric | R5 (no R6) | R6 cond λ 0 | R6 cond λ 0.03 | **R6 cond λ 0.1** (main) | R6 cond λ 0.3 | R6 uncond DANN λ 0.1 |
|---|---|---|---|---|---|---|
| Best epoch | 19 | 19 | 19 | 23 | 18 | 23 |
| Val crop / disease top-1 | 97.8 / 85.4 | 97.8 / 85.4 | 97.8 / 85.5 | 97.9 / 84.8 | 97.8 / 84.9 | 97.9 / 84.6 |
| Val selection score | 91.58 | 91.58 | **91.66** | 91.36 | 91.36 | 91.31 |
| Seen crop / disease top-1 | 91.7 / 67.5 | 91.6 / 67.5 | 91.6 / 67.5 | **92.1** / **68.6** | 91.6 / 66.8 | 92.0 / 68.5 |
| Seen comp top-1 / top-5 | 65.7 / 89.6 | 65.7 / 89.5 | 65.7 / 89.6 | **67.0** / **89.7** | 65.3 / 89.1 | 66.9 / 89.3 |
| Unseen crop top-1 | 70.2 | 70.2 | 70.2 | 69.8 | **70.5** | 69.5 |
| Unseen disease top-1 | 10.2 | 10.2 | 10.2 | 10.7 | **13.3** | 9.9 |
| Unseen comp top-1 / top-5 | 5.2 / 29.1 | 5.2 / 29.1 | 5.2 / 28.9 | 5.5 / 30.7 | **8.1** / 33.7 | 4.8 / **34.0** |
| Unseen class-balanced top-1 | 7.3 | 7.3 | 7.2 | 7.9 | **8.0** | 6.4 |
| HM top-1 / top-5 (uncalibrated) | 9.7 / 44.0 | 9.7 / 44.0 | 9.7 / 43.7 | 10.2 / 45.7 | **14.5** / 48.9 | 9.0 / **49.2** |
| Calibrated best HM | **60.1** | **60.1** | **60.1** | 59.5 | 59.4 | 58.8 |
| Seen / unseen acc at best HM | 54.4 / 67.3 | 54.4 / 67.3 | 54.2 / 67.3 | 56.9 / 62.4 | 56.0 / 63.2 | 57.5 / 60.2 |
| Best unseen bias | 7.26 | 7.26 | 7.25 | 6.93 | **6.44** | 6.53 |
| AUSUC | **49.0** | **49.0** | **49.0** | 48.0 | 46.7 | 46.7 |
| f_d probe acc (chance 53.3) | 52.4 ± 8.8 | 52.4 | 51.4 | 48.2 ± 10.0 | 43.7 ± 11.0 | 48.9 |
| f_d leak (majority baseline) ↓ | −0.02 ± 0.19 | −0.02 ± 0.19 | −0.04 ± 0.22 | −0.11 ± 0.21 | **−0.20 ± 0.24** | −0.09 ± 0.24 |

### What the ablations show
- **The λ = 0 run reproduces R5 exactly.** Every metric is the same, so adding the discriminator with no reversed gradient changes nothing. Any difference in the other rows therefore comes from L_inv and not from the code path.
- **λ = 0.03 is too weak to matter.** It ends at the same epoch as R5 with almost the same numbers, and it has the highest val score (91.66, +0.08 over R5, within noise). Under the `guide.md` selection rule it would be the chosen R6 setting, but it is in effect R5.
- **λ = 0.3 gives the largest uncalibrated unseen gain of any S3 run.** Against R5: unseen disease top-1 10.2 → 13.3, unseen comp top-1 5.2 → 8.1 (+2.9 points, about 42 images), and HM top-1 9.7 → 14.5. That HM is close to single-task EffB0 (8.5 / 15.1). Seen comp top-1 falls only slightly (65.7 → 65.3).
- **The λ = 0.3 gain is a shift in operating point, not a better trade-off.** It needs a smaller calibration bias (6.44 vs 7.26), so its uncalibrated point is less biased toward seen pairs. Across the full curve, though, it is worse: calibrated best HM 59.4 vs 60.1 and AUSUC 46.7 vs 49.0. Its val score (91.36) is also below R5, so validation would not pick it.
- **Calibrated metrics fall steadily as λ grows.** AUSUC is 49.0 / 49.0 / 48.0 / 46.7 for λ = 0 / 0.03 / 0.1 / 0.3, and calibrated best HM is 60.1 / 60.1 / 59.5 / 59.4. Stronger invariance costs ranking quality.
- **Leakage goes further below chance as λ grows, and unseen disease accuracy moves only at λ = 0.3.** Leak is −0.02 / −0.04 / −0.11 / −0.20. The probe already sits at chance without R6, so the larger λ makes f_d *anti*-predictive of the majority crop rather than removing real leakage. The unseen disease gain at λ = 0.3 (+3.1 points) may come from pushing the disease head away from the majority-crop disease prior, not from removing crop information. This probe cannot tell those two apart.
- **Unconditional DANN is worse than conditional R6 on almost every metric.** At the same λ = 0.1, unseen comp top-1 is 4.8 vs 5.5, unseen disease 9.9 vs 10.7, HM top-1 9.0 vs 10.2, calibrated best HM 58.8 vs 59.5 (the lowest in the table), and AUSUC 46.7 vs 48.0. Only HM top-5 is higher (49.2, from unseen top-5 34.0). Its leak (−0.09) is about the same as conditional R6. Restricting D to C(d) therefore helps a little, which fits the idea that the unconditional adversary also strips disease information that crops carry.
- **Overall, H1 is still not supported.** No R6 setting beats R5 on the selection rule (except the no-op λ = 0.03), on calibrated best HM, or on AUSUC. The one large uncalibrated gain (λ = 0.3) disappears once the seen/unseen trade-off is calibrated. All rows are single seeds, and only the λ = 0.3 unseen gain (about 42 images) is clearly larger than the noise described in [Caveats](#caveats).

## Stage 3 R4: learnable prompts on the taxonomy-text base

Sources: `outputs/s3_r4/` (λ_kg 8, main row) and `outputs/s3_r4_nokg/` (λ_kg 0 sub-row), each `eval/metrics.json` and `history.csv`. Code in `stage3/s3_r4/` (see its README). R4 keeps the 2b base (frozen BioCLIP, taxonomy-text cosine heads, no adapters), but the class vectors come from learnable prompts passed through the frozen text encoder: a CoOp context (4 tokens per head, initialised from "a photo of a") plus CSP class-name token deltas (61 crop + 574 disease subwords), with the KgCoOp anchor `λ_kg · ‖T − T⁰‖²` to the unchanged taxonomy prompts. Token LR 1e-3, attribute dropout 0.3, 30 epochs, seed 42. Same split, augmentation, schedule, selection rule and evaluator (333 valid pairs) as R5, so these numbers **are** comparable with the main table, unlike the Colab R4 below. Epoch 0 reproduces 2b / R5 epoch 0 exactly (val score 24.89).

| Metric | **R4** (λ_kg 8) | R4 no KG (λ_kg 0) | R5 (reference) |
|---|---|---|---|
| Best epoch / epochs run | 22 / 30 | 28 / 30 | 19 / 27 |
| Final drift ‖T − T⁰‖² crop / disease | **0.07 / 0.13** | 1.17 / 1.40 | — |
| Val crop / disease top-1 | 91.1 / 67.7 | 96.6 / 76.3 | **97.8 / 85.4** |
| Val selection score (2b base: 89.57) | 79.36 | 86.44 | **91.58** |
| Seen crop / disease top-1 | 83.3 / 52.9 | 89.1 / 59.9 | **91.7 / 67.5** |
| Seen comp top-1 / top-5 | 49.7 / 78.5 | 58.0 / 85.1 | **65.7 / 89.6** |
| Unseen crop top-1 | 64.0 | 66.6 | **70.2** |
| Unseen disease top-1 | **13.7** | 11.5 | 10.2 |
| Unseen comp top-1 / top-5 | **6.7 / 41.0** | 6.4 / 32.1 | 5.2 / 29.1 |
| Unseen class-balanced top-1 | **10.3** | 8.2 | 7.3 |
| HM top-1 / top-5 (uncalibrated) | **11.8 / 53.8** | 11.6 / 46.6 | 9.7 / 44.0 |
| Calibrated best HM | 49.6 | 54.0 | **60.1** |
| Seen / unseen acc at best HM | 41.4 / 61.9 | 48.0 / 61.7 | 54.4 / 67.3 |
| Best unseen bias | 8.03 | 6.42 | 7.26 |
| AUSUC | 33.5 | 40.3 | **49.0** |
| f_d leak (majority baseline) | 0.46 ± 0.18 | 0.46 ± 0.18 | −0.02 ± 0.19 |

### What R4 shows
- **R4 loses on the selection rule, to R5 and to its own 2b base.** Val selection score is 79.36 (λ_kg 8) and 86.44 (no KG), against 89.57 for 2b with fixed taxonomy prompts and 91.58 for R5. Learning the prompts makes the seen-pair classifier worse, so under `guide.md` neither R4 row would be kept.
- **The KgCoOp anchor at λ_kg 8 holds the prompts almost fixed and the model underfits.** Drift settles at 0.07 / 0.13 by epoch 5 and barely moves afterwards. Train accuracy at the best epoch is 62.8 / 36.9 (crop / disease, with augmentation and attribute dropout on), against 79.7 / 55.0 for the no-KG run at its best epoch. Seen comp top-1 drops to 49.7 (R5: 65.7).
- **Without the anchor, the prompts jump early and then stay put.** Drift is already 1.12 / 1.38 at epoch 5 and ends at 1.17 / 1.40. The R4 README attributed this to the large first step from the 1e-3 token LR, but the [token-LR sweep](#r4-token-lr-sweep) shows the same final drift (1.08–1.13 / 1.41–1.45) at every LR down to 5e-5. It does not keep growing, so it is not runaway overfitting.
- **R4 λ_kg 8 has the best uncalibrated unseen numbers of any trained model on the top-5 side.** Unseen disease top-1 is 13.7 (R5: 10.2; R5 + R6 λ 0.3: 13.3), unseen comp top-5 41.0 (single-task EffB0: 35.4) and HM top-5 53.8 (single-task EffB0: 50.6). Unseen comp top-1 (6.7) and HM top-1 (11.8) beat every S3 run except R5 + R6 λ 0.3 (8.1 / 14.5), and stay below single-task EffB0 (8.5 / 15.1).
- **The unseen gain comes from staying near zero-shot, and calibration removes it.** Calibrated best HM is 49.6 and AUSUC 33.5, the lowest of any trained local run (R5: 60.1 / 49.0). It also needs the largest calibration bias of any run (8.03). This is the same pattern as the Colab CSP + KgCoOp runs: a stronger anchor keeps more of BioCLIP's zero-shot unseen disease knowledge (zero-shot: 25.7) but gives up seen accuracy, and the full seen/unseen trade-off curve gets worse.
- **KG vs no KG: the anchor trades seen for unseen but does not improve top-1.** Going from λ_kg 0 to 8, unseen disease top-1 rises 11.5 → 13.7 and unseen top-5 32.1 → 41.0. Unseen comp top-1 barely moves (6.4 → 6.7, about 4 images), while seen comp top-1 falls 58.0 → 49.7 and AUSUC 40.3 → 33.5. The no-KG run is below R5 on every seen and calibrated metric. Its uncalibrated unseen gain (5.2 → 6.4, about 17 images) is within the noise.
- **Leakage is unchanged, as expected.** R4 has no adapters, so its disease feature is N(v). The probe gives the same 0.46 as the 2b base row, which confirms the evaluator wiring. R4 changes only the text side, so it cannot affect this probe.
- **Overall, learnable prompts do not beat fixed taxonomy prompts here.** Neither R4 row beats 2b on val or R5 on any calibrated metric. The uncalibrated unseen gains are an operating-point shift toward zero-shot, like R6 λ 0.3. Both rows are single seeds.

### R4 token-LR sweep

Sources: `outputs/s3_r4_lr{5e-4,1e-4,5e-5}/` and `outputs/s3_r4_lr{5e-4,1e-4,5e-5}_nokg/` (each `eval/metrics.json` and `history.csv`), summarised in `outputs/s3_r4_lr_sweep_summary.md`. Run with `python -m stage3.s3_r4.sweep`. Each run is the unchanged R4 `train.py` / `evaluate.py` with only `lr_new` (the LR of every trainable R4 parameter: CoOp context and CSP name-token deltas) changed, and λ_kg 0 for the `_nokg` rows. Everything else matches the main R4 rows (seed 42, 30 epochs, batch 64, 2 warmup epochs, cosine to 0.01 × LR, 333-pair evaluator). The lr 1e-3 columns are the existing main R4 rows, not retrained. Per `guide.md` this is a named sweep sub-row: it is reported next to the main row and does not replace it. Values are percentages, except drift, bias and leak.

**λ_kg 8**

| Metric | **lr 1e-3** (main) | lr 5e-4 | lr 1e-4 | lr 5e-5 |
|---|---|---|---|---|
| Best epoch / epochs run | 22 / 30 | 30 / 30 | 27 / 30 | 25 / 30 |
| Drift ‖T − T⁰‖² crop / disease, epoch 1 → 5 → best | 0.05 / 0.10 → 0.08 / 0.14 → 0.07 / 0.13 | 0.05 / 0.09 → 0.07 / 0.13 → 0.07 / 0.13 | 0.04 / 0.09 → 0.06 / 0.12 → 0.07 / 0.12 | 0.04 / 0.09 → 0.06 / 0.11 → 0.07 / 0.12 |
| Train crop / disease top-1 at best epoch | 62.8 / 36.9 | 61.6 / 37.2 | 59.1 / 30.5 | 57.1 / 26.4 |
| Val crop / disease top-1 | 91.1 / 67.7 | **91.7** / 67.5 | 88.2 / 59.0 | 86.6 / 54.8 |
| Val selection score | 79.36 | **79.58** | 73.62 | 70.72 |
| Seen crop / disease top-1 | 83.2 / **52.8** | **83.4** / 52.2 | 80.8 / 47.4 | 78.8 / 44.3 |
| Seen comp top-1 / top-5 | **49.7 / 78.5** | 49.0 / 78.0 | 43.9 / 75.2 | 40.3 / 71.2 |
| Unseen crop top-1 | **64.0** | 63.1 | 62.5 | 61.7 |
| Unseen disease top-1 | 13.7 | 13.7 | **13.8** | 13.3 |
| Unseen comp top-1 / top-5 | 6.7 / 41.0 | **7.3** / 41.2 | 5.7 / **41.8** | 6.0 / 40.1 |
| Unseen class-balanced top-1 | **10.3** | 10.2 | 8.1 | 7.6 |
| HM top-1 / top-5 (uncalibrated) | 11.8 / 53.8 | **12.7 / 54.0** | 10.1 / 53.7 | 10.5 / 51.3 |
| Calibrated best HM | **49.6** | 49.4 | 46.0 | 43.1 |
| Seen / unseen acc at best HM | 41.4 / 61.9 | 40.8 / 62.6 | 37.5 / 59.6 | 34.2 / 58.1 |
| Best unseen bias | 8.03 | 8.31 | 8.04 | 8.19 |
| AUSUC | **33.5** | 32.8 | 29.3 | 26.4 |

**λ_kg 0 (no KG)**

| Metric | **lr 1e-3** (main no-KG) | lr 5e-4 | lr 1e-4 | lr 5e-5 |
|---|---|---|---|---|
| Best epoch / epochs run | 28 / 30 | 25 / 30 | 27 / 30 | 29 / 30 |
| Drift ‖T − T⁰‖² crop / disease, epoch 1 → 5 → best | 0.18 / 0.55 → 1.12 / 1.38 → 1.17 / 1.41 | 0.06 / 0.31 → 1.07 / 1.41 → 1.13 / 1.45 | 0.05 / 0.10 → 1.00 / 1.36 → 1.09 / 1.41 | 0.05 / 0.09 → 0.86 / 1.33 → 1.08 / 1.42 |
| Train crop / disease top-1 at best epoch | 79.7 / 55.0 | 79.0 / 53.9 | 73.2 / 43.9 | 69.0 / 37.4 |
| Val crop / disease top-1 | **96.6 / 76.3** | 96.1 / 75.9 | 95.1 / 68.7 | 92.0 / 61.3 |
| Val selection score | **86.44** | 85.99 | 81.89 | 76.60 |
| Seen crop / disease top-1 | **89.0 / 59.9** | 87.8 / 59.4 | 85.9 / 53.3 | 80.7 / 47.2 |
| Seen comp top-1 / top-5 | **58.0 / 85.1** | 56.9 / 83.7 | 50.6 / 80.1 | 43.4 / 74.6 |
| Unseen crop top-1 | 66.6 | **67.2** | 64.3 | 59.9 |
| Unseen disease top-1 | **11.5** | 11.0 | 10.1 | 10.1 |
| Unseen comp top-1 / top-5 | **6.4** / 32.1 | 6.3 / 26.5 | **6.5 / 33.5** | 5.2 / 33.1 |
| Unseen class-balanced top-1 | 8.2 | 8.0 | **11.0** | 10.4 |
| HM top-1 / top-5 (uncalibrated) | **11.6** / 46.6 | 11.3 / 40.2 | 11.5 / **47.2** | 9.3 / 45.9 |
| Calibrated best HM | **54.0** | 52.3 | 47.9 | 43.5 |
| Seen / unseen acc at best HM | 48.0 / 61.7 | 44.8 / 62.7 | 40.6 / 58.4 | 35.0 / 57.3 |
| Best unseen bias | 6.42 | 6.79 | 5.87 | 5.30 |
| AUSUC | **40.3** | 39.4 | 33.1 | 27.3 |

The f_d leak is 0.46 ± 0.18 for all eight runs, since R4 has no adapters and its disease feature is N(v).

#### What the token-LR sweep shows
- **A lower token LR does not remove the drift jump without the anchor; it only delays it by a few epochs.** At λ_kg 0, drift at epoch 1 is much smaller at low LR (disease 0.09–0.10 at 1e-4 / 5e-5 vs 0.55 at 1e-3; epoch 1 is still in warmup), but by epoch 5 it is 0.86–1.07 (crop) / 1.33–1.41 (disease) at every LR, and at the best epoch all four runs sit at 1.08–1.17 / 1.41–1.45. So the ≈ 1.1–1.4 drift in the no-KG row is not caused by the 1e-3 step size, as `guide.md` and the R4 README assumed: the prompts reach the same distance from T⁰ at 20× lower LR. With λ_kg 8, drift is 0.07 / 0.12–0.13 at every LR; the anchor sets it, not the LR.
- **Lower LR only makes the model underfit more.** Since drift ends up the same, the main effect of a smaller LR is slower fitting: train accuracy at the best epoch, val score, seen accuracy, calibrated best HM and AUSUC all fall steadily as the LR goes down, in both groups. At λ_kg 0, val score is 86.44 / 85.99 / 81.89 / 76.60 and AUSUC 40.3 / 39.4 / 33.1 / 27.3 for lr 1e-3 / 5e-4 / 1e-4 / 5e-5. The early drift jump at 1e-3 is therefore not what costs R4 its seen accuracy (next step 6's hypothesis); the lower LRs have the same drift and lower seen accuracy.
- **Selection rule: lr 5e-4 wins narrowly at λ_kg 8, lr 1e-3 wins at λ_kg 0.** At λ_kg 8, lr 5e-4 has val score 79.58 vs 79.36 for the main row (+0.22, about 3 of 1342 val predictions), so under `guide.md` it is the val-preferred λ_kg 8 setting. Its best epoch is the last one (30), so it may still be under-trained. On test it is a wash: unseen comp top-1 +0.6 (6.7 → 7.3, about 9 images) and HM top-1 11.8 → 12.7, while seen comp top-1 −0.7, calibrated best HM −0.2 and AUSUC −0.7. At λ_kg 0 the main 1e-3 row stays best on val (86.44) and on every seen and calibrated metric. The best R4 row overall is still lr 1e-3, λ_kg 0 (86.44), below the 2b base (89.57) and R5 (91.58).
- **Recommendation: keep the token LR at 1e-3; 5e-4 is not adopted.** The 5e-4 win is within noise, appears only in the λ_kg 8 group, and reverses in the λ_kg 0 group. Adopting it would mean re-running R4 + R5 and All at `--lr_prompt 5e-4` (`guide.md`: an adopted LR applies to every run with the module), but in those runs the prompts drift only 0.015–0.019 at 1e-3, so a lower token LR is unlikely to change them.
- **Unseen disease accuracy depends on λ_kg, not the LR.** It is 13.3–13.8 for every λ_kg 8 run and 10.1–11.5 for every λ_kg 0 run, the same split as between the two main R4 rows. This fits the earlier reading that R4's unseen disease gain comes from the anchor keeping the classifier near zero-shot.
- All eight runs are single seeds (42).

## Stage 3 R4 + R5 and All (R4 + R5 + R6)

Sources: `outputs/s3_r4r5/` and `outputs/s3_all/`, each `eval/metrics.json` and `history.csv`. Code in `stage3/s3_r4r5/` and `stage3/s3_all/` (see `stage3/s3_r4r5/README.md`). R4 + R5 uses R4's learnable prompts (CoOp context + CSP name tokens, KgCoOp anchor λ_kg 8) as the text side and R5's adapters A_c, A_d as the image side. All also adds R6's conditional crop discriminator on f_d (GRL, λ_max 0.1, DANN ramp). Every module, including the prompt tokens, uses LR 1e-3 (the [R4 token-LR sweep](#r4-token-lr-sweep) did not adopt a lower token LR). Same split, schedule, selection rule and evaluator (333 valid pairs) as R4 / R5, seed 42. Epoch 0 reproduces 2b's exactly (val loss 12.79333, crop 47.69, disease 2.08).

| Metric | R4 (λ_kg 8) | R5 | **R4 + R5** | R5 + R6 | **All (R4 + R5 + R6)** |
|---|---|---|---|---|---|
| Best epoch / epochs run | 22 / 30 | 19 / 27 | 19 / 27 | 23 / 30 | 25 / 30 |
| Drift ‖T − T⁰‖² crop / disease at best epoch | 0.07 / 0.13 | — | 0.016 / 0.019 | — | 0.015 / 0.018 |
| Val crop / disease top-1 | 91.1 / 67.7 | 97.8 / **85.4** | **98.1** / 85.1 | 97.9 / 84.8 | 97.9 / 84.4 |
| Val selection score | 79.36 | **91.58** | **91.58** | 91.36 | 91.13 |
| Seen crop / disease top-1 | 83.3 / 52.9 | 91.7 / 67.5 | 90.9 / 66.0 | **92.1 / 68.6** | 91.1 / 66.7 |
| Seen comp top-1 / top-5 | 49.7 / 78.5 | 65.7 / 89.6 | 64.3 / 88.5 | **67.0 / 89.7** | 64.9 / 88.6 |
| Unseen crop top-1 | 64.0 | **70.2** | 69.2 | 69.8 | 68.4 |
| Unseen disease top-1 | **13.7** | 10.2 | 10.4 | 10.7 | 10.4 |
| Unseen comp top-1 / top-5 | **6.7 / 41.0** | 5.2 / 29.1 | 4.6 / 34.5 | 5.5 / 30.7 | 5.0 / 34.4 |
| Unseen class-balanced top-1 | **10.3** | 7.3 | 7.8 | 7.9 | 6.3 |
| HM top-1 / top-5 (uncalibrated) | **11.8 / 53.8** | 9.7 / 44.0 | 8.6 / 49.6 | 10.2 / 45.7 | 9.4 / 49.6 |
| Calibrated best HM | 49.6 | 60.1 | **60.5** | 59.5 | 59.4 |
| Seen / unseen acc at best HM | 41.4 / 61.9 | 54.4 / 67.3 | 53.9 / 69.1 | 56.9 / 62.4 | 54.8 / 64.9 |
| Best unseen bias | 8.03 | 7.26 | 7.07 | 6.93 | 7.12 |
| AUSUC | 33.5 | **49.0** | 47.9 | 48.0 | 47.2 |
| f_d probe acc (chance 53.3) | 74.6 | 52.4 | 54.0 ± 8.7 | 48.2 | 51.0 ± 9.2 |
| f_d leak (majority baseline) ↓ | 0.46 ± 0.18 | −0.02 ± 0.19 | 0.02 ± 0.19 | **−0.11 ± 0.21** | −0.05 ± 0.20 |

### What R4 + R5 and All show
- **R4 + R5 ties R5 on the selection rule.** Both get 1229 of 1342 val predictions right (val score 91.58). R4 + R5 gets 2 more crops right and 2 fewer diseases. It stops at the same epoch as R5 (best 19, early stop 27). The selection rule does not prefer either row, and All (91.13) is the lowest of the four adapter runs.
- **With adapters in the model, the prompts barely move.** Drift peaks after epoch 1 (0.04 / 0.08) and then *shrinks* to 0.016 / 0.019, about a quarter (crop) to a seventh (disease) of R4 alone (0.07 / 0.13). The adapters fit the seen classes, so the classification loss stops pushing the tokens and L_kg pulls them back toward T⁰. In practice R4 + R5 is R5 with nearly fixed taxonomy prompts, which explains why it lands so close to R5.
- **R4's unseen disease gain does not carry over.** R4 alone reaches 13.7 unseen disease top-1. R4 + R5 and All have 10.4, back in R5's 10–11% band (Finding 3). That gain came from the text classifier staying near zero-shot while the image feature stayed fixed. Once the adapters retrain the image feature on seen pairs, it is gone.
- **Against R5 on test: top-5 up, top-1 down, calibrated metrics mixed.** Unseen comp top-5 rises 29.1 → 34.5 (+77 images) and HM top-5 44.0 → 49.6, the only change clearly above the noise in [Caveats](#caveats). Unseen comp top-1 falls 5.2 → 4.6 (−9 images) and seen comp top-1 65.7 → 64.3 (−29 images). Calibrated best HM is 60.5, the highest of any S3 run (R5: 60.1), but AUSUC falls 49.0 → 47.9. So the prompts move the correct unseen pair up into the top 5 more often, but not to rank 1.
- **Adding R6 to R4 + R5 repeats the R5 → R5 + R6 pattern.** Uncalibrated top-1 improves slightly (unseen comp 4.6 → 5.0, +6 images; seen 64.3 → 64.9; HM top-1 8.6 → 9.4), while calibrated best HM (60.5 → 59.4) and AUSUC (47.9 → 47.2) fall, as they did from R5 to R5 + R6 (60.1 → 59.5, 49.0 → 48.0). The f_d leak goes from 0.02 to −0.05, both inside the noise around 0. The discriminator reaches a higher train accuracy than in R5 + R6 (69% vs 60% at the best epoch, L_inv 0.98 vs 1.31), but the val probe still finds no crop information beyond the majority baseline.
- **Overall, the modules do not add up, and All is not the best unseen / HM model** (guide.md expected it to be). R4 + R5 is a tie with R5 on val and a mixed result on test. All loses on val, calibrated best HM and AUSUC. The best uncalibrated S3 results are still R5 + R6 λ 0.3 (HM top-1 14.5) and R4 alone (HM top-5 53.8), and the best calibrated results are still R5 (AUSUC 49.0) and R4 + R5 (best HM 60.5, +0.4 over R5, within noise). Both rows are single seeds with the token LR at 1e-3.

## Stage 3 R4 variant: CSP + KgCoOp (Colab)

Source: `CSP_KgCooP/CSP_KgCoOp.ipynb` (cell outputs; `CSP.ipynb` is the same notebook), with the no-KG 3-seed runs saved in `CSP_KgCooP/R4_CSP/` and `CSP_KgCooP/R4_CoOp_CSP/`. Frozen BioCLIP ViT-B/16 with cached 10-view features. Instead of CoOp's shared context, CSP (Nayak et al., ICLR 2023) learns a delta Δθ on each class-name token (61 crop + 574 disease subwords = 325,120 params) inside the fixed prompts `a photo of a {crop} leaf` / `a photo of a leaf with {disease}`. KgCoOp (Yao et al., CVPR 2023) adds `λ_kg · mean_k ‖T[k] − T⁰[k]‖²`, which keeps the learned text classifier close to the frozen zero-shot one. Other settings: attribute dropout 0.3, csp_lr 1e-4, csp_wd 0.05, 30 epochs.

The best epoch is chosen by HM on a pseudo-unseen validation split (10 held-out multi-crop compositions: val_seen 723 / val_unseen 538 images). The model is then refit on all 8,000 training images for that many epochs and tested once.

**These numbers are not comparable with the main table.** This evaluator scores open-world over all 55 × 175 = 9,625 pairs, not the 333 valid pairs, and reports AUSUC on a 0–1 scale (shown here ×100). R4 (CoOp) is the notebook's own CoOp row under the same evaluator. For the comparable local R4 (taxonomy prompts, 333-pair evaluator), see [Stage 3 R4](#stage-3-r4-learnable-prompts-on-the-taxonomy-text-base).

| Metric | R4 CoOp | R4 CSP (no KG) | R4 CoOp + CSP (no KG) | **R4 CSP + KgCoOp** (λ_kg 8) | R4 CoOp + CSP + KgCoOp (λ_kg 8) | R4 CSP + KgCoOp (λ_kg 25, lr 5e-5) |
|---|---|---|---|---|---|---|
| Seeds | 0/1/2 | 0/1/2 | 0/1/2 | 42 | 42 | 42 |
| Trainable params | 5,120 | 325,120 | 330,240 | 325,120 | 330,240 | 325,120 |
| Val HM (selection) | 9.35 ± 3.01 | **9.61 ± 1.22** | 7.71 ± 1.12 | 7.54 | 8.62 | 6.79 |
| Seen comp top-1 | 46.43 ± 1.79 | 43.90 ± 6.04 | **52.95 ± 2.87** | 37.05 | 30.45 | 27.55 |
| Seen comp top-5 | 70.85 ± 1.65 | 68.72 ± 5.77 | **77.82 ± 3.00** | 59.95 | 54.00 | 48.95 |
| Unseen comp top-1 | 3.11 ± 1.28 | 2.37 ± 0.35 | 2.76 ± 0.73 | 2.28 | 3.87 | **4.77** |
| Unseen comp top-5 | 15.68 ± 2.85 | 14.99 ± 1.45 | 15.17 ± 2.25 | 19.75 | 24.38 | **25.55** |
| Unseen crop top-1 | **70.79 ± 2.25** | 65.56 ± 4.73 | 70.67 ± 0.67 | 57.46 | 56.49 | 52.07 |
| Unseen disease top-1 | 7.34 ± 1.02 | 7.34 ± 0.33 | 7.80 ± 1.88 | 9.12 | **10.08** | 9.74 |
| Unseen class-balanced top-1 | 4.20 ± 1.20 | 4.05 ± 0.37 | **4.81 ± 1.10** | 2.49 | 3.81 | 3.45 |
| HM top-1 | 5.79 ± 2.25 | 4.49 ± 0.60 | 5.24 ± 1.33 | 4.29 | 6.86 | **8.13** |
| HM top-5 | — | — | — | 29.71 | **33.59** | 33.58 |
| AUSUC (×100) | **4.6 ± 0.7** | 3.0 ± 0.8 | 4.9 ± 0.8 | 3.5 | 3.7 | 3.7 |
| Best epoch | 21/9/18 | 2/4/8 | 11/23/8 | 4 | 2 | 7 |

Validation sweep for R4 CSP + KgCoOp (seed 42, val only, `R4_csp_kg_sweep_val_s42.json`):

| λ_kg | csp_lr | Best epoch | Final drift ‖T − T⁰‖² | Val seen | Val unseen | Val HM |
|---|---|---|---|---|---|---|
| 0 | 1e-4 | 2 | 0.361 | 45.6 | 5.0 | 9.0 |
| 0 | 3e-4 | 7 | 1.583 | **64.5** | **5.2** | **9.6** |
| 4 | 1e-4 | 4 | 0.280 | 53.4 | 4.7 | 8.6 |
| 4 | 3e-4 | 3 | 0.401 | 52.1 | 5.0 | 9.2 |
| 8 | 1e-4 | 4 | **0.187** | 48.7 | 4.1 | 7.5 |
| 8 | 3e-4 | 4 | 0.279 | 50.4 | 4.8 | 8.8 |

### What R4 CSP + KgCoOp shows
- **KgCoOp does what it is designed to do, but that does not help here.** It keeps the prompts close to the zero-shot classifier (drift 0.19 at λ_kg 8 vs 1.58 unregularised). In the val sweep, though, the unregularised λ_kg = 0 runs have the best val HM (9.0–9.6), and λ_kg = 8, lr 1e-4 (the setting that was run on test) has the worst (7.5). The notebook's claim that KgCoOp improves seen/unseen balance is not supported by its own sweep.
- **On test it lowers seen accuracy without a matching unseen gain.** CSP + KgCoOp (λ 8) reaches seen top-1 37.1 against 43.9 for CSP without KG, with unseen top-1 2.3 vs 2.4 and HM 4.3 vs 4.5. The stronger anchor (λ 25, lr 5e-5) gives the best unseen top-1 (4.8) and HM (8.1) in the table, but seen falls to 27.6 and val HM to 6.8. That run would not be picked by the val selection rule; its settings were chosen to limit drift, not from val.
- **The higher HM comes from collapsing toward zero-shot, not from better composition.** As λ_kg grows, seen accuracy and unseen crop accuracy fall (57.5 → 52.1), and unseen top-5 rises toward the zero-shot profile. AUSUC (3.5–3.7) stays below CoOp (4.6) and CoOp + CSP without KG (4.9), so across the full seen/unseen trade-off the KgCoOp variants are worse.
- **The disease bottleneck is unchanged.** Unseen disease top-1 is 9.1–10.1% for every KgCoOp variant, the same 7–11% band as every other trained model (Finding 3). In the unseen test errors for CSP + KgCoOp (λ 8), 55.2% of images get the crop right and the disease wrong, against 6.8% crop-only errors. The main confusions are within-crop disease swaps (tomato early blight → septoria leaf spot / late blight / target spot, grape black rot → esca).
- **Model selection is noisy.** Best epochs are 2–7, and val unseen accuracy moves in 1–4% steps on 538 images (1% ≈ 5 images), so val HM picks near-initialisation checkpoints. The KgCoOp rows are single seed 42, while the no-KG rows are means of 3 seeds with std up to 6 points on seen top-1. No difference in this table is clearly beyond noise except the seen-accuracy drop.

## Caveats
- Only 10 of the 333 valid pairs are unseen. A few percentage points of unseen accuracy is about 30–60 images, so small differences are noisy.
- The S3 R5 `metrics.json` uses a different format from the EffB0 runs and has no closed-world numbers. Seen `class_balanced_acc` is `NaN` by design: `stage3/s3_r5/metrics.py` averages per-pair accuracy over unseen pairs only, so on the seen split the list is empty. It is not a bug.
- Both evaluation scripts are labelled as using all valid pairs, which is why the composition numbers are treated as comparable here. The scoring code itself has not been checked line by line, so confirm it before putting these numbers side by side in a report.
- The scalar-bias "calibrated best HM" is chosen on the test sets (`bias_chosen_on: test_seen + test_unseen (oracle)`), so it is optimistic for every run. Only the cross-fitted numbers are held-out.
- All runs are single seeds (42). R5's AUSUC lead over tuned EffB0 (0.1 points) and the R5 vs R5 + R6 differences (≤ 1.3 points) are not meaningful on their own.

## Suggested next steps
1. Run the advanced logit-adjustment calibration (per-head temperature, pair prior, cross-fitted) on S3 R5's logits. Its large seen bias (7.26) suggests calibration has the most room to help it.
2. Evaluate other backbones (multi-task and independent) to see whether the multi-task vs single-task gap appears beyond EffB0.
3. Add closed-world composition numbers to `stage3/s3_r5/evaluate.py` so its output matches the EffB0 format.
4. R6 is done, including the λ sweep and the unconditional DANN ablation (see [the ablation section](#stage-3-r5--r6-ablations-λ-sweep-and-unconditional-dann)). No setting improves calibrated HM or AUSUC over R5. If R6 is pursued further, re-run λ = 0.3 with 2–3 more seeds to check whether its uncalibrated unseen gain (HM top-1 14.5) holds, and run the advanced logit-adjustment calibration on it alongside R5 (step 1).
5. Run the Stage 2b base through the Stage 3 evaluator (test sets; its N(v) leakage is already the base row above) so R4, R5, R4 + R5 and All can be compared against their own base on test.
6. R4 is done at λ_kg 8 and 0, including the token-LR sweep (lr {5e-4, 1e-4, 5e-5} × λ_kg {8, 0}; see [R4 token-LR sweep](#r4-token-lr-sweep)), and R4 + R5 and All are done at token LR 1e-3 (see [Stage 3 R4 + R5 and All](#stage-3-r4--r5-and-all-r4--r5--r6)). A lower LR does not avoid the drift and only lowers seen accuracy. lr 5e-4 wins on val at λ_kg 8 by 0.22 points (within noise), so 1e-3 is kept. If 5e-4 is adopted after all, re-run R4 + R5 and All with `--lr_prompt 5e-4`, and consider a 5e-4 run with more epochs, since its best epoch was the last.

## Stage 2 decision: R8a logit adjustment for Stage 3

Decided on the validation selection score (mean of val crop top-1 and val disease top-1), as `guide.md` requires. No test data was used.

| Run | R8a τ_LA | Val selection score |
|---|---|---|
| 2b BioCLIP + taxonomy | off | **89.57** |
| 2d = 2b + R8a (scheme default) | 1.0 | 71.16 |
| 2d sweep | 0.5 | 87.48 |

**Decision: R8a is off for every Stage 3 run.** 2d does not beat 2b at the scheme default (τ_LA = 1.0, −18.4 points) or at the τ_LA = 0.5 sweep (−2.1 points). The Stage 3 base is 2b alone.

## Stage 2 and CoOp runs (Colab notebooks)

Test-set metrics (percentages) from the Colab notebooks in `stage2/` and `CSP_KgCooP/`. "Basic" = 2b (taxonomy text, no R8a); 2d is 2b + R8a at the given τ_LA. CoOp is a separate experiment; the CSP / KgCoOp comparison is in [Stage 3 R4 variant: CSP + KgCoOp](#stage-3-r4-variant-csp--kgcoop-colab). Zero-shot is epoch 0, before any training.

| Metric | Zero-shot | 2b Basic | 2d τ=0.5 | 2d τ=1.0 | CoOp | CoOp + LA 0.5 |
|---|---|---|---|---|---|---|
| Seen T1 | 5.25 | **63.80** | 60.60 | 52.35 | 41.70 | 44.50 |
| Seen T5 | 21.50 | **88.70** | 88.15 | 82.55 | 71.00 | 75.05 |
| Unseen T1 | **20.86** | 3.66 | 2.97 | 3.25 | 6.28 | 5.80 |
| HM T1 | 8.39 | 6.92 | 5.66 | 6.11 | **10.92** | 10.26 |
| Unseen T5 | **46.89** | 25.76 | 23.69 | 28.52 | 34.39 | 36.95 |
| Unseen crop | 37.02 | **66.71** | **66.71** | 55.66 | 60.77 | 62.78 |
| Unseen disease | **30.46** | 8.01 | 6.42 | 6.08 | 10.22 | 8.43 |
| Balanced unseen | 14.79 | 4.97 | 3.66 | 3.03 | **10.33** | 9.09 |
| AUSUC | 2.88 | **43.53** | 40.28 | 34.36 | 27.67 | 32.24 |
| Calibrated HM | 8.68 | **57.25** | 54.05 | 50.09 | 43.09 | 47.24 |
| Best val score | 24.89 | **89.57** | 87.48 | 71.16 | 75.26 | 73.40 |
| Best epoch / epochs run | 0 / 0 | 30 / 30 | 17 / 25 | 10 / 18 | 28 / 30 | 30 / 30 |

- **R8a hurts here.** Both 2d runs are below 2b on seen accuracy, AUSUC and val score, and gain nothing on unseen, which backs the "R8a off" decision above.
- **CoOp trades seen for unseen.** Versus 2b: unseen T1 6.3 vs 3.7, HM T1 10.9 vs 6.9, but seen T1 41.7 vs 63.8 and AUSUC 27.7 vs 43.5. Zero-shot still has the best unseen numbers (20.9 T1, 30.5 disease), so training on seen pairs costs unseen disease accuracy in every run.
- **2b vs R5 (test).** 2b: seen 63.8 / unseen 3.7 / HM 6.9 / AUSUC 43.5; R5: 65.7 / 5.2 / 9.7 / 49.0. R5 is ahead on each, but the Colab `evaluate()` has not been checked against `stage3/s3_r5/evaluate.py`, so treat this as indicative.
