# Results: Compositional Zero-Shot Plant Disease Classification (DPD-Small)

This file compares the five evaluated runs in `outputs/`:

| Run | Approach | Source |
|---|---|---|
| **S3 R5** | Frozen BioCLIP, taxonomy-text heads (Stage 2b base) plus separate crop/disease residual adapters A_c, A_d (Stage 3) | `outputs/s3_r5/eval/metrics.json` |
| **S3 R5 + R6** | S3 R5 plus a conditional crop discriminator on the disease feature behind a gradient-reversal layer (λ_max = 0.1) | `outputs/s3_r5r6/eval/metrics.json` |
| **EffB0 multi-task** | One EfficientNet-B0 backbone with two heads | `outputs/multitask_effb0/eval/metrics.json` |
| **EffB0 single-task** | Two independent EfficientNet-B0 models (crop, disease) | `outputs/SingleTask_effb0_InitialAttempt/eval/metrics.json` |
| **EffB0 single-task + logit tuning** | Same checkpoints as single-task, plus logit adjustment (per-head temperature/τ, pair prior, cross-fitted) | `outputs/FineTunedResultsB0(Logit+Hyperparameter)/eval/metrics.json` |

All five use the same test splits: `test_seen` has 2000 images and `test_unseen` has 1448. Compositions are scored as `log p(crop|x) + log p(disease|x)` over the 333 valid pairs (323 seen, 10 unseen).

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

## Findings

### 1. With one backbone, EffB0 multi-task does worse than single-task (independent) EffB0
- Seen accuracy is about the same for both (composition top-1 67.9% vs 68.4%).
- On unseen data the multi-task model is clearly worse: crop top-1 is 66.9% vs 72.1%, composition top-1 is 4.9% vs 8.5%, and HM top-1 is 9.1 vs 15.1.
- A likely cause is that the shared features learn which crops and diseases appear together in training, which hurts on new pairings. Two separate models cannot learn those pairings.

### 2. After calibration the gaps mostly close
- With the plain scalar-bias sweep, every model reaches a best HM of about 60–61.
- AUSUC is close for S3 R5 and single-task EffB0 (48.8–49.0) and lower for EffB0 multi-task (47.1).
- The tuned EffB0 run reaches **64.9 HM top-1**, the best result overall. That number comes from tuning the calibration settings on the test sets. Its cross-fitted estimate, which tunes on one part of the test data and scores on the rest, is 64.6, so the gain is real. S3 R5 has not had this tuning yet, so this comparison is not like-for-like.

### 3. Disease recognition on unseen pairs is the bottleneck for every model
- Unseen disease top-1 is only about 10–11% everywhere, while unseen crop top-1 is 67–72%.
- The disease head predicts the diseases it saw with each crop during training. None of these models solves that.
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
- Only λ_max = 0.1 has been run. The λ sweep {0, 0.03, 0.3} and the unconditional-DANN ablation in `stage3/s3_r5r6/README.md` have not.

## Caveats
- Only 10 of the 333 valid pairs are unseen. A few percentage points of unseen accuracy is about 30–60 images, so small differences are noisy.
- The S3 R5 `metrics.json` uses a different format from the EffB0 runs and has no closed-world numbers. Seen `class_balanced_acc` is `NaN` by design: `stage3/s3_r5/metrics.py` averages per-pair accuracy over unseen pairs only, so on the seen split the list is empty. It is not a bug.
- Both evaluation scripts are labelled as using all valid pairs, which is why the composition numbers are treated as comparable here. The scoring code itself has not been checked line by line, so confirm it before putting these numbers side by side in a report.
- The DINOv2 and EffB3 baselines have no eval outputs yet, so they are not included.
- S3 R5 and R5 + R6 are single seeds (42). R5's AUSUC lead over tuned EffB0 (0.1 points) and the R5 vs R5 + R6 differences (≤ 1.3 points) are not meaningful on their own.

## Suggested next steps
1. Run the advanced logit-adjustment calibration (per-head temperature, pair prior, cross-fitted) on S3 R5's logits. Its large seen bias (7.26) suggests calibration has the most room to help it.
2. Evaluate the DINOv2 and EffB3 baselines (multi-task and independent) to see whether the multi-task vs single-task gap appears with other backbones.
3. Add closed-world composition numbers to `stage3/s3_r5/evaluate.py` so its output matches the EffB0 format.
4. R6 is done (see the R5 + R6 section): it removes no measurable leakage that R5 had not already removed and does not change unseen accuracy beyond noise. Run the λ = 0 sanity row only if a reviewer needs it; the rest of the λ sweep is unlikely to be worth the compute.
5. Run the Stage 2b base through the Stage 3 evaluator (test sets; its N(v) leakage is already the base row above) so R5 can be compared against its own base, then continue with R4 + R5 and All.

## Stage 2 decision: R8a logit adjustment for Stage 3

Decided on the validation selection score (mean of val crop top-1 and val disease top-1), as `guide.md` requires. No test data was used.

| Run | R8a τ_LA | Val selection score |
|---|---|---|
| 2b BioCLIP + taxonomy | off | **89.57** |
| 2d = 2b + R8a (scheme default) | 1.0 | 71.16 |
| 2d sweep | 0.5 | 87.48 |

**Decision: R8a is off for every Stage 3 run.** 2d does not beat 2b at the scheme default (τ_LA = 1.0, −18.4 points) or at the τ_LA = 0.5 sweep (−2.1 points). The Stage 3 base is 2b alone.

## Stage 2 and CoOp runs (Colab notebooks)

Test-set metrics (percentages) from the Colab notebooks in `stage2/` and `CSP_KgCooP/`. "Basic" = 2b (taxonomy text, no R8a); 2d is 2b + R8a at the given τ_LA. CoOp is a separate experiment, to be compared with CSP (not run yet). Zero-shot is epoch 0, before any training.

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
