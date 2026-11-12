# Results: Compositional Zero-Shot Plant Disease Classification (DPD-Small)

This file compares the four evaluated runs in `outputs/`:

| Run | Approach | Source |
|---|---|---|
| **BioCLIP adapter (MT)** | Frozen BioCLIP image encoder plus two residual adapters and heads (multi-task) | `outputs/multitask_bioclip_adapter/eval/metrics.json` |
| **EffB0 multi-task** | One EfficientNet-B0 backbone with two heads | `outputs/multitask_effb0/eval/metrics.json` |
| **EffB0 single-task** | Two independent EfficientNet-B0 models (crop, disease) | `outputs/SingleTask_effb0_InitialAttempt/eval/metrics.json` |
| **EffB0 single-task + logit tuning** | Same checkpoints as single-task, plus logit adjustment (per-head temperature/τ, pair prior, cross-fitted) | `outputs/FineTunedResultsB0(Logit+Hyperparameter)/eval/metrics.json` |

All four use the same test splits: `test_seen` has 2000 images and `test_unseen` has 1448. Compositions are scored as `log p(crop|x) + log p(disease|x)` over the 333 valid pairs (323 seen, 10 unseen).

## Main comparison

All values are percentages.

| Metric | **BioCLIP adapter (MT)** | EffB0 multi-task | EffB0 single-task | EffB0 single-task + logit tuning |
|---|---|---|---|---|
| **Seen** crop top-1 | **90.5** | 90.1 | 90.1 | 90.1 |
| Seen disease top-1 | 61.2 | **69.4** | 68.9 | 69.0 |
| Seen comp top-1 | 59.4 | 67.9 | 68.4 | **68.5** |
| Seen comp top-5 | 84.7 | 88.4 | **88.6** | **88.6** |
| **Unseen** crop top-1 | **74.0** | 66.9 | 72.1 | 72.1 |
| Unseen disease top-1 | **11.1** | 10.8 | 9.9 | 9.9 |
| Unseen comp top-1 | **8.9** | 4.9 | 8.5 | 8.5 |
| Unseen comp top-5 | **42.1** | 30.0 | 35.4 | 35.4 |
| HM top-1 (uncalibrated) | **15.5** | 9.1 | 15.1 | 15.1 |
| HM top-5 (uncalibrated) | **56.3** | 44.8 | 50.6 | 50.6 |
| Calibrated best HM (bias sweep) | **61.9** | 60.7 | 61.1 | 61.2 |
| Seen / unseen acc at best HM | 52.0 / 76.5 | 58.3 / 63.4 | 59.0 / 63.3 | 59.1 / 63.5 |
| Best unseen bias | 4.54 | 6.46 | 6.43 | 6.43 |
| AUSUC | 46.8 | 47.1 | 48.8 | **48.9** |

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

The adapter's evaluator does not report closed-world numbers.

## Findings

### 1. The BioCLIP adapter trades seen accuracy for unseen accuracy
- It is the best model on every unseen metric. Unseen composition top-5 is 42.1% against 35.4% for the next best, unseen crop top-1 is 74.0%, and uncalibrated HM top-1 is 15.5.
- Seen composition top-1 is about 8–9 points below the CNNs. Almost all of that loss comes from the disease head (61.2% vs about 69%). Seen crop accuracy matches the CNNs.
- The likely reason: BioCLIP is frozen and only the small adapters are trained, so the model cannot fit seen disease patterns as closely as a fully fine-tuned CNN. That same limit seems to stop it overfitting to seen pairs, so it generalises better to unseen combinations.

### 2. With one backbone, EffB0 multi-task does worse than single-task (independent) EffB0
- Seen accuracy is about the same for both (composition top-1 67.9% vs 68.4%).
- On unseen data the multi-task model is clearly worse: crop top-1 is 66.9% vs 72.1%, composition top-1 is 4.9% vs 8.5%, and HM top-1 is 9.1 vs 15.1.
- A likely cause is that the shared features learn which crops and diseases appear together in training, which hurts on new pairings. Two separate models cannot learn those pairings.
- The adapter avoids this even though it is also multi-task, because its frozen backbone cannot be reshaped to fit the training pairs.

### 3. After calibration the gaps mostly close
- With the plain scalar-bias sweep, every model reaches a best HM of about 61–62. The adapter is only marginally ahead (61.9).
- AUSUC favours single-task EffB0 (48.8–48.9 vs 46.8 for the adapter). Across the whole seen/unseen trade-off curve the adapter is not better; it reaches a better point on that curve when no calibration is applied.
- The adapter's best operating point is lopsided (52.0% seen, 76.5% unseen). It also needs a smaller bias (4.5 vs about 6.4), which means it is less biased toward seen pairs to begin with.
- The tuned EffB0 run reaches **64.9 HM top-1**, the best result overall. That number comes from tuning the calibration settings on the test sets. Its cross-fitted estimate, which tunes on one part of the test data and scores on the rest, is 64.6, so the gain is real. The adapter has not had this tuning yet, so this comparison is not like-for-like.

### 4. Disease recognition on unseen pairs is the bottleneck for every model
- Unseen disease top-1 is only about 10–11% everywhere, while unseen crop top-1 is 67–74%.
- The disease head predicts the diseases it saw with each crop during training. None of these models solves that; the adapter only does slightly better (11.1%).
- Closed-world unseen accuracy is about 75–77%. When the model only has to choose among the unseen pairs it does well; it fails because seen pairs outscore the correct unseen pair.

## Caveats
- Only 10 of the 333 valid pairs are unseen. A few percentage points of unseen accuracy is about 30–60 images, so small differences are noisy.
- The adapter's `metrics.json` uses a different format from the EffB0 runs. It has no closed-world numbers, and seen `class_balanced_acc` is `NaN`, which looks like a small bug in `adapter/metrics.py`.
- Both evaluation scripts are labelled as using all valid pairs, which is why the composition numbers are treated as comparable here. The scoring code itself has not been checked line by line, so confirm it before putting these numbers side by side in a report.
- The DINOv2 and EffB3 baselines have no eval outputs yet, so they are not included.

## Suggested next steps
1. Run the advanced logit-adjustment calibration (per-head temperature, pair prior, cross-fitted) on the BioCLIP adapter's logits. If the adapter keeps its unseen advantage while getting some seen accuracy back, it would probably beat 64.9 HM.
2. Evaluate the DINOv2 and EffB3 baselines (multi-task and independent) to see whether the multi-task vs single-task gap appears with other backbones.
3. Fix the `NaN` class-balanced accuracy in the adapter evaluator, and add closed-world composition numbers so its output matches the EffB0 format.
