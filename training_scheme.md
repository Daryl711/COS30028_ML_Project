# Training scheme (applies to every experiment in this repo)

This is the single training recipe for all experiments in Stages 1–3 (`guide.md`).
It is the BioCLIP adapter scheme (`adapter/config.yaml`, `adapter/train.py`), which in turn copies the
DINOv2 baseline. **If an experiment's config disagrees with this file, this file wins.**
Results that were not trained with this scheme must not appear in the same table as results that were.

The rule: an experiment may change **only** (a) the model and (b) the method-specific loss terms and
hyperparameters listed in §6. Everything in §1–§5 is fixed.

---

## 1. Data

| Setting | Value |
|---|---|
| Image folder | `dataset/DPD_Small_dataset` (original resolution, not `DPD_Small_256`) |
| CSVs | `DPD_Small_train.csv` (8,000), `DPD_Small_test_seen.csv` (2,000), `DPD_Small_test_unseen.csv` (1,448) |
| Validation | 10 % of **each composition** held out from train (`val_fraction: 0.1`), at least 1 image per pair stays in train, split seeded with `seed` (`split_train_val` in `adapter/dataset.py`) |
| Test sets | Never read during training, model selection or hyperparameter choice |
| Input size | 224 × 224 |
| `num_workers` | 4 |

## 2. Augmentation and preprocessing

Train transform (identical for every model):

```python
T.RandomResizedCrop(224, scale=(0.6, 1.0))
T.RandomHorizontalFlip()
T.RandomVerticalFlip(p=0.2)
T.TrivialAugmentWide()
T.ToTensor()
T.Normalize(MEAN, STD)
T.RandomErasing(p=0.25)
```

Eval transform: `Resize(256)` (= 224 / 0.875) → `CenterCrop(224)` → `ToTensor` → `Normalize(MEAN, STD)`.

Augmentation runs **online every step**. Feature caching (the "10 cached views" in
`PROPOSED_METHOD.md` §3.2) is **not** used, so a frozen-backbone model sees exactly the same augmentation
as a fine-tuned one.

`MEAN`/`STD` is the only allowed difference, and it follows the backbone's pretraining:

| Backbone | Normalization |
|---|---|
| EfficientNet-B0 (ImageNet) | ImageNet mean/std |
| BioCLIP | CLIP mean/std `(0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758)` |

## 3. Optimisation

| Setting | Value |
|---|---|
| Optimizer | AdamW, one param group per trainable module |
| Weight decay | **0.05** on every group |
| LR, new modules (heads, adapters, text classifiers, prompt tokens, discriminator) | **1e-3** |
| LR, pretrained backbone (only when the backbone is trainable, i.e. EfficientNet in Stage 1) | 2e-4 |
| Batch size | 64 (`accum_steps: 1`) |
| Epochs | 30 (maximum) |
| Schedule | 2 epochs linear warm-up, then cosine decay to 1 % of the peak LR, stepped **per optimizer step** |
| Mixed precision | fp16 AMP with `GradScaler` on CUDA; losses computed in fp32 (`.float()` on logits) |
| Gradient clipping | global norm **1.0** over all trainable params (unscale first) |
| Seed | **42** (Python, NumPy, torch, CUDA, `PYTHONHASHSEED`) |

## 4. Loss

```
L = 1.0 · CE(crop logits, crop) + 1.0 · CE(disease logits, disease)  [+ method terms, §6]
```

- Label smoothing **0.1** on both CE terms.
- Head dropout **0.3** on the feature before each head.
- Method terms (§6) are added to this sum; they never replace it.

## 5. Model selection, checkpoints and evaluation

| Setting | Value |
|---|---|
| Selection score | mean of val crop top-1 and val disease top-1 |
| Early stopping | patience **8** epochs on the selection score |
| Checkpoints | `best.pt` (best selection score), `last.pt` (last epoch), each with the full config |
| Run outputs | `outputs/<experiment>/`: `best.pt`, `last.pt`, `history.csv`, `config.yaml`, `eval/metrics.json` |
| Evaluated checkpoint | always `best.pt` |

**Evaluation protocol** (`adapter/evaluate.py` + `adapter/metrics.py`; every stage uses the same one):

- Pair score `log p(crop|x) + log p(disease|x)`, over **valid pairs only**: the 323 train pairs plus the 10
  `test_unseen` pairs (333 candidates, generalized setting).
- Report: seen / unseen composition top-1 and top-5, crop and disease top-1 on each split, class-balanced
  unseen accuracy, HM top-1 / top-5, calibrated best HM from the bias sweep, and AUSUC.
- **No test-set tuning.** Temperatures, biases or priors fitted on test data are not part of this scheme.
  The bias sweep is reported only as the AUSUC curve and its best HM, labelled as such.
- Inference is "Mode A" in `PROPOSED_METHOD.md`: independent heads, no pair branch, no feasibility mask.

---

## 6. What each experiment is allowed to change

| Stage | Experiment | Backbone | Trainable | Allowed method hyperparameters |
|---|---|---|---|---|
| 1 | EffB0 multi-task | EfficientNet-B0, fine-tuned, `drop_path_rate 0.2` | backbone + 2 linear heads | none |
| 1 | EffB0 single-task (×2) | EfficientNet-B0, fine-tuned, `drop_path_rate 0.2` | backbone + 1 linear head per model | none |
| 2 | 2a image only | BioCLIP ViT-B/16, frozen | 2 linear heads on N(v) | none |
| 2 | 2b–2e | BioCLIP ViT-B/16, frozen (image and text encoders) | 2 text-initialised cosine heads on N(v), **no adapters** | text source, η (2c, 2e), R8a on/off (2d, 2e) |
| 3 | R4 / R5 / R5+R6 / R4+R5 / all | as Stage 2 | 2b heads (taxonomy text) + the module(s) switched on; R8a only if Stage 2 chose it | §6.1 |

R6 is only ever run together with R5 (it acts on the disease adapter's output).

Values not listed here are §1–§5 values.

### 6.1 Method hyperparameters (fixed; taken from `PROPOSED_METHOD.md` §5)

| Module | Hyperparameter | Value |
|---|---|---|
| R5 adapters (Stage 3 only) | shape, dropout, init | A_c, A_d: 512 → 256 → 512, GELU, dropout 0.1, up-projection zero-init |
| Cosine text heads | logit scale | BioCLIP's pretrained logit scale, fixed |
| Cosine text heads | init | text classifier $T^0$ from the run's text source; taxonomy text in Stage 3 |
| 2c / 2e descriptions | K, η | 20 per class, η = 0.3 |
| R8a logit adjustment | τ_LA, prior | 1.0; training-split class frequencies; in the loss only, never at prediction |
| R4 prompts | context tokens M, attribute dropout | 4, 0.3 |
| R4 KgCoOp anchor | λ_kg | 8.0 |
| R6 conditional GRL | λ_max, ramp, discriminator | 0.1, DANN ramp `λ(p)=λ_max(2/(1+e^{-10p})−1)`, 2-layer MLP (512 → 256 → C) |

Sweeps (η, λ_max, …) are separate runs, named as such, and do not change the default.

### 6.2 Overrides of `PROPOSED_METHOD.md`

`PROPOSED_METHOD.md` §5 lists a slightly different recipe. For this project the following values win:

| Setting | PROPOSED_METHOD.md | This scheme |
|---|---|---|
| LR adapters / prompts | 5e-4 | 1e-3 |
| Features | cached, 10 views | online augmentation |
| Seeds | 3 (0, 1, 2) | 1 (42); extra seeds only if every row of a table gets them |
| Model selection | HM on pseudo-unseen val | mean val top-1 of both heads |
| Augmentation | RandAugment N=2, M=7 | TrivialAugmentWide + flips + RandomErasing |

---

## 7. Compliance of existing code and outputs

| Item | Status | Needed to comply |
|---|---|---|
| `adapter/` (`config.yaml`, `train.py`) | Reference implementation | — |
| `configs/baseline.yaml` (EffB0) | `weight_decay: 1.0e-2`, no `grad_clip` | set `weight_decay: 5.0e-2`, add `grad_clip: 1.0` |
| `baseline/MultiTaskBaseline/InitialAttempt(Effb0)/train.py` | does not apply gradient clipping | add clipping as in `adapter/train.py` `run_epoch` |
| `outputs/multitask_effb0`, `outputs/SingleTask_effb0` | trained with wd 1e-2, no clipping | retrain as Stage 1 |
| `outputs/FineTunedResultsB0(Logit+Hyperparameter)` | calibration tuned on test | not comparable; keep out of the stage tables |
| `outputs/multitask_bioclip_adapter` | follows the scheme; adapters + linear heads (no text) | reference row only; not a stage experiment (Stage 2 has no adapters, Stage 3 uses text heads) |

## 8. Checklist before reporting a run

- [ ] `outputs/<experiment>/config.yaml` matches §1–§5 except for the §6 lines.
- [ ] Seed 42, val split from train, test sets untouched until `evaluate`.
- [ ] Evaluated `best.pt` with the generalized valid-pair protocol (§5).
- [ ] Row added to `results.md` from `eval/metrics.json`, never typed by hand.
