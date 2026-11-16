# Results: Compositional Zero-Shot Plant Disease Classification (DPD-Small)

This file compares the four evaluated runs in `outputs/`:

| Run | Approach | Source |
|---|---|---|
| **S3 R5** | Frozen BioCLIP, taxonomy-text heads (Stage 2b base) plus separate crop/disease residual adapters A_c, A_d (Stage 3) | `outputs/s3_r5/eval/metrics.json` |
| **EffB0 multi-task** | One EfficientNet-B0 backbone with two heads | `outputs/multitask_effb0/eval/metrics.json` |
| **EffB0 single-task** | Two independent EfficientNet-B0 models (crop, disease) | `outputs/SingleTask_effb0_InitialAttempt/eval/metrics.json` |
| **EffB0 single-task + logit tuning** | Same checkpoints as single-task, plus logit adjustment (per-head temperature/τ, pair prior, cross-fitted) | `outputs/FineTunedResultsB0(Logit+Hyperparameter)/eval/metrics.json` |

All four use the same test splits: `test_seen` has 2000 images and `test_unseen` has 1448. Compositions are scored as `log p(crop|x) + log p(disease|x)` over the 333 valid pairs (323 seen, 10 unseen).

## Main comparison

All values are percentages.

| Metric | **S3 R5** | EffB0 multi-task | EffB0 single-task | EffB0 single-task + logit tuning |
|---|---|---|---|---|
| **Seen** crop top-1 | **91.7** | 90.1 | 90.1 | 90.1 |
| Seen disease top-1 | 67.5 | **69.4** | 68.9 | 69.0 |
| Seen comp top-1 | 65.7 | 67.9 | 68.4 | **68.5** |
| Seen comp top-5 | **89.6** | 88.4 | 88.6 | 88.6 |
| **Unseen** crop top-1 | 70.2 | 66.9 | **72.1** | **72.1** |
| Unseen disease top-1 | 10.2 | **10.8** | 9.9 | 9.9 |
| Unseen comp top-1 | 5.2 | 4.9 | **8.5** | **8.5** |
| Unseen comp top-5 | 29.1 | 30.0 | **35.4** | **35.4** |
| Unseen class-balanced top-1 | 7.3 | — | — | — |
| HM top-1 (uncalibrated) | 9.7 | 9.1 | **15.1** | **15.1** |
| HM top-5 (uncalibrated) | 44.0 | 44.8 | **50.6** | **50.6** |
| Calibrated best HM (bias sweep) | 60.1 | 60.7 | 61.1 | **61.2** |
| Seen / unseen acc at best HM | 54.4 / 67.3 | 58.3 / 63.4 | 59.0 / 63.3 | 59.1 / 63.5 |
| Best unseen bias | 7.26 | 6.46 | 6.43 | 6.43 |
| AUSUC | **49.0** | 47.1 | 48.8 | 48.9 |

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

The S3 R5 evaluator does not report closed-world numbers.

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

A 5-fold logistic-regression probe predicts the crop from the disease feature, on the 428 **validation** images whose disease occurs on ≥ 2 crops (30 such diseases in val; 42 multi-crop diseases in train). Chance is 20.5%. `leak` rescales probe accuracy so 0 = chance and 1 = perfect.

| Feature | Probe acc | Leak ↓ |
|---|---|---|
| Base N(v) (frozen BioCLIP) | 74.6 ± 8.4 | 0.68 ± 0.11 |
| R5 f_d | 52.4 ± 8.8 | **0.40 ± 0.11** |

### What R5 shows
- **Validation improves, test unseen does not.** R5 beats the 2b base on the val selection score (+2.0), but val only contains seen pairs. On test it has the best seen crop top-1 (91.7) and seen comp top-5 (89.6) of all runs, yet unseen comp top-1 is 5.2% and HM top-1 is 9.7, close to EffB0 multi-task (4.9 / 9.1) and below single-task EffB0 (8.5 / 15.1).
- **Strongest seen bias of any run.** It needs the largest calibration bias (7.26 vs about 6.4–6.5) and reaches the lowest calibrated best HM (60.1). AUSUC is nevertheless the highest (49.0, marginally above tuned EffB0 at 48.9), so the full seen/unseen trade-off curve is competitive; the uncalibrated operating point is what is poor.
- **The adapter removes crop information from f_d, but this does not carry over to unseen disease accuracy.** Leakage drops from 0.68 to 0.40 (the stds of about 0.11 mean this is a large but not precise drop), while unseen disease top-1 stays at 10.2%, the same 10–11% bottleneck every model shows (Finding 3). Less crop information in the disease feature is not by itself enough to stop the disease head favouring diseases seen with that crop.
- **The R5 effect cannot be isolated yet.** There is no test-set eval for the 2b base, so it is unknown how much of R5's test behaviour comes from the text heads (2b) versus the separate adapters (R5). Evaluating 2b with `stage3/s3_r5/evaluate.py`'s protocol is needed for that comparison.

## Caveats
- Only 10 of the 333 valid pairs are unseen. A few percentage points of unseen accuracy is about 30–60 images, so small differences are noisy.
- The S3 R5 `metrics.json` uses a different format from the EffB0 runs and has no closed-world numbers. Seen `class_balanced_acc` is `NaN` by design: `stage3/s3_r5/metrics.py` averages per-pair accuracy over unseen pairs only, so on the seen split the list is empty. It is not a bug.
- Both evaluation scripts are labelled as using all valid pairs, which is why the composition numbers are treated as comparable here. The scoring code itself has not been checked line by line, so confirm it before putting these numbers side by side in a report.
- The DINOv2 and EffB3 baselines have no eval outputs yet, so they are not included.
- S3 R5 is a single seed (42). Its AUSUC lead over tuned EffB0 (0.1 points) is not meaningful.

## Suggested next steps
1. Run the advanced logit-adjustment calibration (per-head temperature, pair prior, cross-fitted) on S3 R5's logits. Its large seen bias (7.26) suggests calibration has the most room to help it.
2. Evaluate the DINOv2 and EffB3 baselines (multi-task and independent) to see whether the multi-task vs single-task gap appears with other backbones.
3. Add closed-world composition numbers to `stage3/s3_r5/evaluate.py` so its output matches the EffB0 format.
4. Run the Stage 2b base through the Stage 3 evaluator (test sets; its N(v) leakage is already the base row above) so R5 can be compared against its own base, then continue with R5 + R6, R4 + R5 and All.

## Stage 2 decision: R8a logit adjustment for Stage 3

Decided on the validation selection score (mean of val crop top-1 and val disease top-1), as `guide.md` requires. No test data was used.

| Run | R8a τ_LA | Val selection score |
|---|---|---|
| 2b BioCLIP + taxonomy | off | **89.57** |
| 2d = 2b + R8a (scheme default) | 1.0 | 71.16 |
| 2d sweep | 0.5 | 87.48 |

**Decision: R8a is off for every Stage 3 run.** 2d does not beat 2b at the scheme default (τ_LA = 1.0, −18.4 points) or at the τ_LA = 0.5 sweep (−2.1 points). The Stage 3 base is 2b alone.
