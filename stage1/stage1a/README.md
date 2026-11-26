# EfficientNet-B0 baseline (training scheme, Stage 1)

The B0 baseline retrained so that it counts as **Stage 1** of the training scheme. The model, data, augmentation,
loss and schedule are the same as the initial attempt in `../InitialAttempt(Effb0)`. Two settings did not match the
scheme there, and this folder fixes them:

| Setting | Scheme | Initial attempt (`outputs/multitask_effb0`) | This folder |
|---|---|---|---|
| Weight decay | 0.05 | 0.01 | **0.05** |
| Gradient clipping | global norm 1.0 | none (`train.py` never clipped) | **1.0** |
| Run names | `s1_*` | `multitask_effb0`, `indep_*_effb0` | `s1_mt_effb0`, `s1_crop_effb0`, `s1_disease_effb0` |

Everything else already matched: 10% val split per composition (seed 42), 224 px, batch 64, 30 epochs, 2-epoch
warm-up then per-step cosine decay, AMP, label smoothing 0.1, dropout 0.3, patience 8, learning rates 2e-4
(backbone) / 1e-3 (heads).

The initial attempt is left untouched so its old results stay reproducible. Don't report those as Stage 1.

## How the model works

```
224×224 image
   │
   ▼
EfficientNet-B0 backbone (pretrained on ImageNet)
   │   stem conv → 16 MBConv blocks in 7 stages → 1×1 conv → global average pool
   ▼
1280-number feature vector
   ├─ Dropout(0.3) → Linear(1280 → 55)   → crop logits
   └─ Dropout(0.3) → Linear(1280 → 175)  → disease logits
```

- Loss = 1.0 × crop cross-entropy + 1.0 × disease cross-entropy, with label smoothing 0.1.
- AdamW: 2e-4 for the backbone, 1e-3 for the heads, weight decay 0.05.
- Gradients are clipped to a global norm of 1.0 after AMP unscaling (`scaler.unscale_` then
  `clip_grad_norm_`), so the threshold applies to the real gradient, not the scaled one.
- Composition score at test time: log p(crop) + log p(disease) over valid pairs.

The independent baseline (Stage 1b) is the same network with a single head: one crop-only model and one
disease-only model, evaluated together.

## Files

| File | What changed from `../InitialAttempt(Effb0)` |
|---|---|
| `config.yaml` | new: scheme settings (above); replaces `configs/baseline.yaml` for this baseline |
| `train.py` | from `../EffB3`: reads `config.yaml` next to it by default; supports `grad_clip` and `accum_steps` |
| `model.py` | from `../EffB3`: adds `build_optimizer()`, same two learning rates |
| `evaluate.py`, `plot_confusions.py` | usage paths only |
| `dataset.py`, `metrics.py` | unchanged |
| `train_local_effb0_s1.ipynb` | trains and evaluates all three Stage 1 runs on a local GPU |

## How to run

Run from the repo root (the folder that contains `dataset/`), or open `train_local_effb0_s1.ipynb` and run all cells.

```bash
# Stage 1a: multi-task (one model, two heads)
python baseline/MultiTaskBaseline/EffB0/train.py

# Stage 1b: independent baseline (two models)
python baseline/MultiTaskBaseline/EffB0/train.py --mode crop    --experiment s1_crop_effb0
python baseline/MultiTaskBaseline/EffB0/train.py --mode disease --experiment s1_disease_effb0

# evaluate
python baseline/MultiTaskBaseline/EffB0/evaluate.py --checkpoints outputs/s1_mt_effb0/best.pt --out outputs/s1_mt_effb0/eval
python baseline/MultiTaskBaseline/EffB0/evaluate.py --checkpoints outputs/s1_crop_effb0/best.pt outputs/s1_disease_effb0/best.pt --out outputs/s1_indep_effb0/eval
```

Results go to `outputs/s1_mt_effb0/`, `outputs/s1_crop_effb0/`, `outputs/s1_disease_effb0/` and
`outputs/s1_indep_effb0/eval/`, in the same format as before: `best.pt`, `last.pt`, `history.csv`, `config.yaml`,
and `eval/metrics.json` plus plots and `logits.npz` (input for `../calibrate.py`).

## Time and memory (RTX 3060 Laptop, 6 GB)

- GPU memory: ~3.2 GB at batch 64 with mixed precision, so no gradient accumulation is needed. That keeps
  BatchNorm statistics over 64 images, the same as the initial attempt.
- Time: ~97 s per epoch, so about 50 minutes per 30-epoch run and ~2.5 hours for all three runs. Early stopping
  can end a run sooner.
