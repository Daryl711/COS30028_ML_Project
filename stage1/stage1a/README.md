# Stage 1a: EfficientNet-B0 multi-task baseline (training scheme, Stage 1)

This folder holds the Stage 1 EfficientNet-B0 code. It trains **1a** (multi-task) directly; **1b**
(single-task, `../stage1b/`) runs the same code with `--mode crop` / `--mode disease`.

The run follows `training_scheme.md`: weight decay 0.05, gradient clipping at global norm 1.0, 10% val
split per composition (seed 42), 224 px, batch 64, 30 epochs, 2-epoch warm-up then per-step cosine decay,
AMP, label smoothing 0.1, dropout 0.3, patience 8, learning rates 2e-4 (backbone) / 1e-3 (heads).

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

The single-task baseline (Stage 1b) is the same network with a single head: one crop-only model and one
disease-only model, evaluated together.

## Files

| File | Contents |
|---|---|
| `config.yaml` | scheme settings (above) |
| `train.py` | training loop; reads `config.yaml` next to it by default; supports `grad_clip` and `accum_steps` |
| `model.py` | backbone + heads + loss + `build_optimizer()` (two learning rates) |
| `evaluate.py`, `plot_confusions.py` | test-set scoring and confusion plots |
| `dataset.py`, `metrics.py` | data loading, label space, metrics |
| `train_local_effb0_s1.ipynb` | trains and evaluates all three Stage 1 runs (1a, 1b crop, 1b disease) on a local GPU |

## How to run

Run from the repo root (the folder that contains `dataset/`), or open `train_local_effb0_s1.ipynb` and run all cells.

```bash
# Stage 1a: multi-task (one model, two heads)
python stage1/stage1a/train.py
python stage1/stage1a/evaluate.py --checkpoints outputs/stage1a/best.pt --out outputs/stage1a/eval

# Stage 1b: single-task (two models), see ../stage1b/README.md
python stage1/stage1b/train.py --mode crop
python stage1/stage1b/train.py --mode disease
python stage1/stage1b/evaluate.py
```

Results go to `outputs/stage1a/`, `outputs/stage1b_crop/`, `outputs/stage1b_disease/` and
`outputs/stage1b/eval/`: `best.pt`, `last.pt`, `history.csv`, `config.yaml`, and `eval/metrics.json` plus plots
and `logits.npz` (raw test logits).

## Time and memory (RTX 3060 Laptop, 6 GB)

- GPU memory: ~3.2 GB at batch 64 with mixed precision, so no gradient accumulation is needed.
- Time: ~97 s per epoch, so about 50 minutes per 30-epoch run and ~2.5 hours for all three runs. Early stopping
  can end a run sooner.
