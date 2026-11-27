# Results: Compositional Zero-Shot Plant Disease Classification (DPD-Small)

This file keeps only the runs that follow `guide.md`: the `training_scheme.md` recipe (same split, augmentation, AdamW wd 0.05, grad clip 1.0, 30 epochs, early stopping, seed 42) and an evaluator over the **333 valid pairs** (323 seen, 10 unseen). Every number below was read from that run's `outputs/<run>/eval/metrics.json`. `test_seen` has 2000 images and `test_unseen` has 1448. Compositions are scored as `log p(crop|x) + log p(disease|x)`.

Run IDs match the folder names: code in `stage<N>/stage<N><x>/`, runs in `outputs/stage<N><x>[_<sub-row>]/`. Stage 1: 1a = multi-task, 1b = single-task. Stage 3: 3a = R4, 3b = R5, 3c = R5 + R6, 3d = R4 + R5, 3e = All.

## Which results are comparable

The code and outputs of every "Removed" row below have been deleted from the repo.

| Source | Status | Reason |
|---|---|---|
| `stage1a`, `stage1b` (Stage 1 re-runs) | **Kept** | Guide Stage 1 runs, trained with the scheme's wd 0.05 and grad clip 1.0. |
| `stage3a`, `stage3a_nokg`, `stage3b`, `stage3c`, `stage3d`, `stage3e` | **Kept** | The five guide Stage 3 runs, plus the R4 sub-row without λ_kg that the guide asks for. |
| 2b epoch 0 (zero-shot) | **Kept, Colab evaluator** | `outputs/stage2b/eval/metrics_epoch0.json`: the guide's "epoch 0 is a sub-row" for 2b, 333-pair protocol, same Colab run as the 2b row. The Colab `evaluate()` has not been checked against the local evaluator. |
| `stage3a_lr*` (token-LR sweep), `stage3c_lam*`, `stage3c_uncond` | **Kept as sub-rows** | Named sweep sub-rows (guide: one setting changed, reported next to the main row, never replacing it). |
| 2b val score (89.57) and the R8a decision | **Kept** | The guide requires the Stage 2 decision and both val scores here before Stage 3. |
| `multitask_effb0`, `SingleTask_effb0` | Removed (outdated) | Trained with the old wd and no grad clip. The guide says they "must be rerun", and the `stage1a` / `stage1b` runs replace them. |
| `FineTunedB0(Logit+Hyperparameter)` | Removed (not comparable) | Stage 1 allows no logit adjustment, calibration tuning or hyperparameter search, and its settings were tuned on the test sets. |
| `zeroshot_bioclip`, `zeroshot_clip` | Removed (not in guide) | They use their own pair prompts ("a photo of a tomato leaf with …"), not the 2b taxonomy heads, so they are not the 2b epoch-0 zero-shot sub-row. Generic CLIP has no place in the guide at all. |
| `stage2b`, `stage2d`, `stage2d_tau0.5` (Stage 2 Colab runs, notebooks in `stage2/`) | **Kept, Colab evaluator (†)** | The 2b base and the two R8a runs, kept for the Stage 2 decision. Their test numbers come from the Colab `evaluate()`, which has not been checked against the local evaluator. |
| Stage 2 Colab test numbers (CoOp + LA) | Removed (not comparable) | CoOp + LA is not a guide run. 2b still needs a test eval with `stage3/stage3b/evaluate.py`. |
| `stage3a_coop` (CoOp only, `coop_colab.ipynb`) | **Kept as a † sub-row** | CoOp ablation of `3a R4` (4 shared context tokens, no CSP tokens, no KgCoOp), same `training_scheme.md` recipe, seed 42. Run in Colab (`stage3/stage3a/coop_colab.ipynb`), so it uses the Colab evaluator. † = compare with care until that evaluator is checked against the local one. |
| Colab CSP + KgCoOp (`CSP(another_scheme)/`) | Not comparable (kept as another scheme) | Open-world evaluator over all 9,625 pairs, plain-name prompts, a different selection rule and refitting on the full train set. The local R4 rows replace it. |
| `csp_raw`, `csp_kgcoop` (Pure CSP, `n_ctx 0`) | **Kept as sub-rows** | Pure CSP ablation of `3a R4` (prefix frozen, `n_ctx 0`, seed 42), trained with the `training_scheme.md` recipe. |

### Two evaluators, which metrics match

Stage 1 is scored with `stage1/stage1a/evaluate.py` and Stage 3 with `stage3/stage3b/evaluate.py`. I checked the code of both:

- **Same in both** (comparable across every row): the candidate pair list (sorted train pairs, then sorted unseen pairs), the composition score, composition top-1/top-5, HM, class-balanced unseen top-1, and the scalar-bias sweep (calibrated best HM, AUSUC).
- **Different** (marked †, compare only within Stage 1 or within Stage 3): **crop and disease accuracy**. Stage 1 takes the argmax of each head over all 55 / 175 classes. Stage 3 reads the crop and disease off the top-1 *composition*, so the prediction is limited to the 333 valid pairs. Stage 1 did not save logits, so its numbers cannot be recomputed the Stage 3 way without re-running the evaluation.

## All comparable results

Values are percentages, except val score (mean of val crop and disease top-1), best bias and leak. **Bold** = best in the column. "Main" rows are the guide runs; indented "sub" rows change one setting of the main row above them.

| Run | Role | Val score | Seen T1 | Seen T5 | Unseen T1 | Unseen T5 | Bal. unseen T1 | HM T1 | HM T5 | Cal. best HM | AUSUC | Best bias | Crop S † | Dis S † | Crop U † | Dis U † | Leak ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Stage 1: EfficientNet-B0** | | | | | | | | | | | | | | | | | |
| 1a EffB0 multi-task | main | 92.85 | 67.1 | 87.6 | 4.1 | 27.6 | 3.6 | 7.8 | 42.0 | 60.3 | 46.7 | 6.58 | 89.4 | 68.6 | 68.5 | 8.4 | — |
| 1b EffB0 single-task | main | **93.67** ‡ | **67.6** | 88.5 | **8.6** | 34.8 | 7.9 | **15.2** | 50.0 | **60.9** | 48.2 | 6.60 | 90.1 | 68.6 | 72.5 | 9.9 | — |
| **Stage 2: frozen BioCLIP** | | | | | | | | | | | | | | | | | |
| 2b epoch 0 (BioCLIP zero-shot, taxonomy text) | sub (guide) | 24.9 | 5.3 | 21.5 | **20.9** | **46.9** | **14.8** | 8.4 | 29.5 | 8.7 | 2.9 | 1.87 | 48.5 | 7.2 | 37.0 | **30.5** | — |
| 2b taxonomy text (Stage 3 base) | main | 89.57 | 63.8 | 88.7 | 3.7 | 25.8 | 5.0 | 6.9 | 39.9 | 57.2 | 43.5 | 7.79 | 91.2 | 65.7 | 66.7 | 8.0 | 0.46 ± 0.18 |
| ↳ 2d = 2b + R8a, τ_LA 1.0 † | sub (guide) | 71.16 | 52.4 | 82.6 | 3.2 | 28.5 | 3.0 | 6.1 | 42.4 | 50.1 | 34.4 | 8.36 | 86.3 | 54.2 | 55.7 | 6.1 | — |
| ↳ 2d sweep, τ_LA 0.5 † | sub | 87.48 | 60.6 | 88.2 | 3.0 | 23.7 | 3.7 | 5.7 | 37.3 | 54.0 | 40.3 | 8.03 | 90.0 | 62.3 | 66.7 | 6.4 | — |
| **Stage 3: DISCO modules on 2b** | | | | | | | | | | | | | | | | | |
| 3a R4 (λ_kg 8, token LR 1e-3) | main | 79.36 | 49.7 | 78.5 | 6.7 | 41.0 | 10.3 | 11.8 | 53.8 | 49.6 | 33.5 | 8.03 | 83.3 | 52.8 | 64.0 | 13.7 | 0.46 ± 0.18 |
| ↳ λ_kg 8, token LR 5e-4 | sub | 79.58 | 49.1 | 78.1 | 7.3 | 41.2 | 10.2 | 12.7 | **54.0** | 49.4 | 32.8 | 8.31 | 83.4 | 52.3 | 63.1 | 13.7 | 0.46 |
| ↳ λ_kg 8, token LR 1e-4 | sub | 73.62 | 43.9 | 75.2 | 5.7 | 41.8 | 8.1 | 10.1 | 53.7 | 46.0 | 29.3 | 8.04 | 80.9 | 47.4 | 62.5 | 13.8 | 0.46 |
| ↳ λ_kg 8, token LR 5e-5 | sub | 70.72 | 40.3 | 71.2 | 6.0 | 40.1 | 7.6 | 10.5 | 51.3 | 43.1 | 26.4 | 8.19 | 78.8 | 44.3 | 61.7 | 13.3 | 0.46 |
| ↳ λ_kg 0 (no KG), token LR 1e-3 | sub (guide) | 86.44 | 58.0 | 85.1 | 6.4 | 32.1 | 8.2 | 11.6 | 46.6 | 54.0 | 40.3 | 6.42 | 89.1 | 59.9 | 66.6 | 11.5 | 0.46 |
| ↳ λ_kg 0, token LR 5e-4 | sub | 85.99 | 56.9 | 83.7 | 6.3 | 26.5 | 8.0 | 11.3 | 40.2 | 52.3 | 39.4 | 6.79 | 87.8 | 59.4 | 67.2 | 11.0 | 0.46 |
| ↳ λ_kg 0, token LR 1e-4 | sub | 81.89 | 50.6 | 80.1 | 6.5 | 33.5 | 11.0 | 11.5 | 47.2 | 47.9 | 33.1 | 5.87 | 85.9 | 53.3 | 64.3 | 10.1 | 0.46 |
| ↳ λ_kg 0, token LR 5e-5 | sub | 76.60 | 43.4 | 74.6 | 5.2 | 33.1 | 10.4 | 9.3 | 45.9 | 43.5 | 27.3 | 5.30 | 80.7 | 47.3 | 59.9 | 10.1 | 0.46 |
| ↳ `n_ctx 0` (Pure CSP), λ_kg 8 | sub | 79.66 | 49.6 | 78.0 | 5.3 | 38.6 | 8.4 | 9.6 | 51.6 | 50.2 | 33.6 | 8.56 | 83.6 | 53.0 | 62.4 | 12.8 | 0.46 ± 0.18 |
| ↳ `n_ctx 0` (Pure CSP), λ_kg 0 (no KG) | sub | 84.20 | 56.3 | 83.4 | 5.2 | 25.6 | 6.1 | 9.5 | 39.1 | 51.5 | 36.1 | 8.63 | 87.0 | 59.0 | 64.4 | 10.7 | 0.46 ± 0.18 |
| ↳ CoOp only (`stage3a_coop`, no CSP tokens), λ_kg 0 † | sub (Colab eval) | 75.26 | 41.7 | 71.0 | 6.3 | 34.4 | 10.3 | 10.9 | 46.3 | 43.1 | 27.7 | 3.74 | 81.9 | 44.5 | 60.8 | 10.2 | 0.46 ± 0.18 |
| 3b R5 (separate adapters) | main | 91.58 | 65.7 | **89.6** | 5.2 | 29.1 | 7.3 | 9.7 | 44.0 | 60.1 | **49.0** | 7.26 | 91.6 | 67.5 | 70.2 | 10.2 | −0.02 ± 0.19 |
| 3c R5 + R6 (conditional GRL, λ 0.1) | main | 91.36 | 67.0 | **89.6** | 5.5 | 30.7 | 7.9 | 10.2 | 45.7 | 59.5 | 48.0 | 6.93 | **92.1** | **68.6** | 69.8 | 10.7 | −0.11 ± 0.21 |
| ↳ λ 0 | sub | 91.58 | 65.7 | **89.6** | 5.2 | 29.1 | 7.3 | 9.7 | 44.0 | 60.1 | **49.0** | 7.26 | 91.6 | 67.5 | 70.2 | 10.2 | −0.02 ± 0.19 |
| ↳ λ 0.03 | sub | 91.65 | 65.7 | **89.6** | 5.2 | 28.9 | 7.2 | 9.7 | 43.7 | 60.1 | **49.0** | 7.25 | 91.6 | 67.5 | 70.2 | 10.2 | −0.04 ± 0.22 |
| ↳ λ 0.3 | sub | 91.36 | 65.3 | 89.1 | 8.1 | 33.7 | 8.0 | 14.5 | 48.9 | 59.4 | 46.7 | 6.44 | 91.6 | 66.8 | **70.5** | 13.3 | −0.20 ± 0.24 |
| ↳ unconditional DANN, λ 0.1 | sub | 91.28 | 66.9 | 89.3 | 4.8 | 34.0 | 6.4 | 9.0 | 49.2 | 58.8 | 46.7 | 6.53 | 92.0 | 68.5 | 69.5 | 9.9 | −0.09 ± 0.24 |
| 3d R4 + R5 | main | 91.58 | 64.3 | 88.5 | 4.6 | 34.5 | 7.8 | 8.6 | 49.6 | 60.5 | 47.9 | 7.07 | 90.9 | 66.0 | 69.2 | 10.4 | 0.02 ± 0.19 |
| 3e All (R4 + R5 + R6) | main | 91.13 | 64.9 | 88.6 | 5.0 | 34.4 | 6.3 | 9.4 | 49.6 | 59.4 | 47.2 | 7.12 | 91.1 | 66.6 | 68.4 | 10.4 | −0.05 ± 0.20 |

How to read the columns:

- **Seen / Unseen T1, T5**: composition accuracy, choosing among all 333 valid pairs.
- **Bal. unseen T1**: unseen top-1 averaged per pair, so each of the 10 unseen pairs counts equally.
- **HM**: harmonic mean of seen and unseen composition accuracy, with no calibration.
- **Cal. best HM, AUSUC, best bias**: a scalar bias is added to the unseen-pair scores and swept. Best HM is the best point on that sweep, and AUSUC is the area under the seen/unseen accuracy curve. The bias is chosen **on the test sets**, so best HM is optimistic for every row. A larger best bias means the model is more biased toward seen pairs.
- **Crop / Dis S, U †**: per-head accuracy on the seen (S) and unseen (U) test split. These use a different definition in Stage 1 and Stage 3 (see above).
- **Leak ↓**: crop-leakage probe on the disease feature (below). For 2b and R4 the probed feature is the frozen N(v).
- **Val score**: mean of val crop and val disease top-1 (per-head argmax over all 55 / 175 classes) at the best epoch. It is the `guide.md` selection rule and the only metric that may be used to pick between rows. Every row uses the same val split (`split_train_val`, seed 42), so the column is comparable across stages. Stage 3 values come from `eval/metrics.json`; Stage 1 values come from `outputs/<run>/history.csv`, because the Stage 1 evaluator does not write the val score. 1a: crop 97.17, disease 88.52 at epoch 15.
- **‡ 1b val score**: the crop and disease models are selected separately, so 93.67 is the mean of two best epochs from two checkpoints: crop model 98.21 (epoch 16) and disease model 89.12 (epoch 20).

Run details: all rows use seed 42. Best epochs are R4 22, R5 19, R5 + R6 23, R4 + R5 19, All 25. Every new module uses LR 1e-3 (guide decision); the token-LR sweep changes only the R4 prompt-token LR. The prompt drift ‖T − T⁰‖² (crop / disease) at the best epoch is 0.07 / 0.13 for R4, 1.17 / 1.40 for R4 with λ_kg 0 (1.08–1.13 / 1.41–1.45 at every lower LR), and 0.016 / 0.019 for R4 + R5 and 0.015 / 0.018 for All.

Pure CSP (`n_ctx 0`) best epochs are 25 (λ_kg 8) and 29 (λ_kg 0), with prompt drift (crop / disease) 0.071 / 0.114 and 0.876 / 0.803 respectively (closed-world seen / unseen T1: 49.9 / 74.1 for λ_kg 8; 56.4 / 71.0 for λ_kg 0). CoOp only (`stage3a_coop`) has best epoch 28.

## Stage 2 decision: R8a logit adjustment

Decided on the val selection score, as `guide.md` requires. No test data was used.

| Run | R8a τ_LA | Val selection score |
|---|---|---|
| 2b BioCLIP + taxonomy | off | **89.57** |
| 2d = 2b + R8a (guide default) | 1.0 | 71.16 |
| 2d sweep | 0.5 | 87.48 |

For reference, the test numbers (Colab evaluator †, table above) point the same way: 2d at τ_LA 1.0 / 0.5 against 2b gives seen T1 52.4 / 60.6 vs 63.8, unseen disease 6.1 / 6.4 vs 8.0, calibrated best HM 50.1 / 54.0 vs 57.2 and AUSUC 34.4 / 40.3 vs 43.5. Only unseen T5 at τ_LA 1.0 is higher (28.5 vs 25.8). Unseen differences are a few images and are noise.

**Decision: R8a is off for every Stage 3 run.** 2d is below 2b at τ_LA 1.0 (−18.4) and at 0.5 (−2.1), so the Stage 3 base is 2b alone. 2a, 2c and 2e have no results yet (`stage2/stage2c/` has a notebook, but none of its numbers are recorded).

## Crop-leakage probe

A 5-fold logistic-regression probe predicts the crop from the disease feature, on the 428 val images whose disease occurs on ≥ 2 crops (30 such diseases in val; 42 in train). `leak = (probe_acc − chance) / (1 − chance)`. Chance is the **majority crop per disease**: for each disease, the crop it most often occurs on in train. That reaches 53.3% with no crop evidence, so leak 0 means the feature knows nothing about the crop beyond the disease itself. A leak of 0 ± 0.2 is within the noise across folds.

## Findings

1. **No Stage 3 run beats the Stage 1 single-task baseline on HM.** The guide makes Stage 1 the reference to beat. Single-task EffB0 has the best seen top-1 (67.6), uncalibrated HM top-1 (15.2) and calibrated best HM (60.9) of all rows, and the best unseen top-1 (8.6) of the trained rows (2b epoch 0 zero-shot scores 20.9 unseen but only 5.3 seen). The best Stage 3 rows come close on single metrics: R5 + R6 λ 0.3 reaches HM top-1 14.5, R4 + R5 a calibrated best HM of 60.5, and R5 has the best AUSUC (49.0 vs 48.2).
2. **Sharing one backbone hurts unseen pairs.** EffB0 multi-task and single-task are about equal on seen (67.1 vs 67.6), but on unseen pairs multi-task is far worse: unseen top-1 4.1 vs 8.6, HM top-1 7.8 vs 15.2, AUSUC 46.7 vs 48.2. A likely cause is that shared features learn which crops and diseases appear together in training.
3. **Disease recognition on unseen pairs is the bottleneck for every model.** Unseen disease accuracy is 8–14% in every trained row (zero-shot BioCLIP at 2b epoch 0 reaches 30.5, and training on seen pairs removes it), while unseen crop accuracy is 60–73%. The models predict the diseases they saw with each crop during training.
4. **R5 gives the best Stage 3 val score and calibrated curve, but not better unseen accuracy.** R5 raises the val score over the 2b base (91.58 vs 89.57) and has the best AUSUC (49.0). Its uncalibrated unseen top-1 is 5.2 and HM top-1 9.7, and it needs the largest calibration bias of the adapter runs (7.26), so the uncalibrated operating point is strongly biased toward seen pairs.
5. **Crop leakage is not what limits unseen disease accuracy (H1 not supported).** The frozen feature leaks (0.46), but R5's disease adapter already brings leakage to zero (−0.02) with no change in unseen disease accuracy (10.2). R6 pushes leakage further below zero as λ grows (−0.04 / −0.11 / −0.20 for λ 0.03 / 0.1 / 0.3), while calibrated best HM (60.1 → 59.4) and AUSUC (49.0 → 46.7) fall. λ 0 reproduces R5 exactly, so the changes come from the reversed gradient and not the code path. The unconditional DANN is worse than the conditional version on nearly every metric.
6. **The large uncalibrated gains are operating-point shifts, not better models.** R5 + R6 λ 0.3 (HM top-1 14.5, unseen disease 13.3) and R4 λ_kg 8 (unseen T5 41.0, HM T5 53.8, unseen disease 13.7) both lose calibrated best HM and AUSUC against R5. The val selection rule would not pick either of them.
7. **Learnable prompts (R4) do not beat fixed taxonomy prompts.** Both R4 rows are below the 2b base on val (79.36 and 86.44 vs 89.57). The KgCoOp anchor holds the prompts near zero-shot and the model underfits (seen T1 49.7). Without the anchor the prompts drift to the same distance at every token LR from 1e-3 down to 5e-5, so a lower LR only slows fitting. lr 5e-4 wins on val at λ_kg 8 by 0.22 (about 3 val images) and loses at λ_kg 0, so the token LR stays at 1e-3.
8. **The modules do not add up.** R4 + R5 ties R5 on val (91.58, the same 1229 of 1342 val predictions correct); with adapters present the prompts barely move (drift 0.016 / 0.019), so it is close to R5 with fixed prompts. It raises unseen T5 (29.1 → 34.5) and calibrated best HM (60.5), and lowers AUSUC (47.9). All (91.13) is the lowest adapter run on val, below R5 on calibrated best HM and AUSUC, so the guide's expectation that All is the best unseen / HM model does not hold.
9. Removing the learnable CoOp context (`n_ctx 0`, pure CSP) shows the same pattern and does not improve on `3a R4` (val 84.20 / 79.66 vs 86.44 / 79.36; HM 9.5 / 9.6 vs 11.6 / 11.8; AUSUC 36.1 / 33.6 vs 40.3 / 33.5). CoOp only (no CSP tokens, no anchor, †) is worse still on val, seen T1 and AUSUC than pure CSP at λ_kg 0 (75.26 vs 84.20, 41.7 vs 56.3, 27.7 vs 36.1) and than R4 (86.44), while unseen disease is about equal (10.2 vs 10.7 vs 11.5) and its unseen T1 and HM are within noise (6.3 / 10.9 vs 5.2 / 9.5). Both components help the fit, the CSP class tokens more than the shared CoOp context, and neither moves unseen disease.

## Caveats

- Only 10 of the 333 pairs are unseen. One percentage point of unseen accuracy is about 14 images, so differences of a few points are noisy.
- All rows are single seeds (42). Val score differences of a few tenths (for example 91.65 vs 91.58) are a handful of val images.
- Calibrated best HM and AUSUC are computed with the bias chosen on the test sets, for every row.
- Crop and disease accuracy (†) are defined differently in the two evaluators (see [Two evaluators](#two-evaluators-which-metrics-match)).
- Stage 1 has no val score in `metrics.json`, and the Stage 3 evaluator has no closed-world numbers, so neither appears in the table.

## Next steps

1. Evaluate the 2b base on the test sets with `stage3/stage3b/evaluate.py`, so every Stage 3 row can be compared against its own base.
2. Run the remaining Stage 2 rows from the guide (2a image only, 2c taxonomy + LLM, 2e = 2c + R8a).
3. Make the evaluators agree on crop/disease accuracy: save logits in the Stage 1 evaluator and report both definitions in each. Also add closed-world numbers to the Stage 3 evaluator.
4. Re-run R5 + R6 λ 0.3 with 2–3 more seeds to check whether its uncalibrated unseen gain holds.
