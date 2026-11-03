# EfficientNet-B3 baseline

A bigger version of the B0 baseline. It uses the same two-head design, data, augmentation, loss and
schedule. Only the backbone size and the image resolution change. Any difference from B0 therefore
comes from **model scale and resolution**, not from a new method.

## How the model works

```
300×300 image
   │
   ▼
EfficientNet-B3 backbone (pretrained on ImageNet-1k)
   │   stem conv → 26 MBConv blocks in 7 stages → 1×1 conv → global average pool
   ▼
1536-number feature vector
   ├─ Dropout(0.3) → Linear(1536 → 55)   → crop logits
   └─ Dropout(0.3) → Linear(1536 → 175)  → disease logits
```

### The backbone

EfficientNet is a convolutional network built from **MBConv** blocks. Each block:

1. **Expands** the channels with a 1×1 convolution (usually 6× wider).
2. Applies a **depthwise convolution** (3×3 or 5×5). Each channel is filtered on its own, which is
   much cheaper than a normal convolution.
3. Runs **squeeze-and-excitation**: it averages each channel over the whole image, passes that
   through a tiny network, and uses the result to scale each channel up or down. This lets the
   block focus on the channels that matter for this image, such as lesion colour.
4. **Projects** back down to fewer channels with a 1×1 convolution, and adds a skip connection.

B0 to B7 are the same design scaled by one rule called **compound scaling**: depth, width and input
resolution grow together, because a higher-resolution image needs more layers to see across it and
more channels to hold the extra detail.

|  | B0 (current baseline) | B3 (this folder) |
|---|---|---|
| Input size | 224 px | 300 px |
| Width (channels) | 1.0× | 1.2× |
| Depth (layers) | 1.0× | 1.4× |
| Feature vector | 1280 | 1536 |
| Parameters (with our two heads) | 4.3M | 11.0M |

The weights are timm's `efficientnet_b3.ra2_in1k`: trained on ImageNet-1k with timm's RandAugment
recipe, the same pretraining family as B0.

### The heads, loss and training

These are the same as B0:

- Two heads, each Dropout + one Linear layer, reading the same feature vector.
- Loss = 1.0 × crop cross-entropy + 1.0 × disease cross-entropy, with label smoothing 0.1.
- AdamW: 2e-4 for the backbone, 1e-3 for the heads, weight decay 0.01.
- 2 warmup epochs, then cosine decay, for up to 30 epochs with early stopping (patience 8).
- Composition score at test time: log p(crop) + log p(disease) over valid pairs.

## Why it might help

The most common mistakes of B0 are fine-grained: paddy healthy → blast (62% of those images),
soybean caterpillar → diabrotica (70%). Blast spots and insect bite marks are small, and resizing to
224 px blurs them. At 300 px the image has 1.8× as many pixels, and B3 has the extra depth and width
to use them.

What it probably **won't** fix: the low unseen-disease accuracy (~10% for B0). That failure comes
from the disease head learning "which diseases go with this crop". A bigger network trained the same
way is likely to learn the same shortcut. The DINOv2 baseline in `../DINOv2` targets that problem.

## What changed from B0

| Setting | B0 | B3 | Why |
|---|---|---|---|
| `backbone` | `efficientnet_b0` | `efficientnet_b3.ra2_in1k` | the experiment |
| `img_size` | 224 | 300 | B3's native resolution |
| `batch_size` | 64 | 32 | 64 at 300 px needs ~9.6 GB; your GPU has 6 GB |
| `accum_steps` | (none) | 2 | adds up 2 batches of 32 before each update, so each update still uses 64 images like B0 |

Everything else in `config.yaml` is identical to `configs/baseline.yaml`.

One small difference remains: BatchNorm layers compute their statistics over 32 images per forward
pass instead of 64. Gradient accumulation can't change that. It rarely matters at batch 32, but
mention it when comparing against B0.

## Files

| File | What changed from `../InitialAttempt(Effb0)` |
|---|---|
| `config.yaml` | new: B3 settings (above) |
| `model.py` | adds `build_optimizer()`, same two learning rates as B0 |
| `train.py` | reads `config.yaml` next to it by default; supports `accum_steps` and `grad_clip` |
| `evaluate.py`, `plot_confusions.py` | usage paths only |
| `dataset.py`, `metrics.py` | unchanged |

## How to run

Run from the repo root (the folder that contains `dataset/`).

```bash
# multi-task (one model, two heads)
python baseline/MultiTaskBaseline/EffB3/train.py

# independent baseline (two models)
python baseline/MultiTaskBaseline/EffB3/train.py --mode crop    --experiment indep_crop_effb3
python baseline/MultiTaskBaseline/EffB3/train.py --mode disease --experiment indep_disease_effb3

# evaluate
python baseline/MultiTaskBaseline/EffB3/evaluate.py --checkpoints outputs/multitask_effb3/best.pt --out outputs/multitask_effb3/eval
python baseline/MultiTaskBaseline/EffB3/evaluate.py --checkpoints outputs/indep_crop_effb3/best.pt outputs/indep_disease_effb3/best.pt --out outputs/indep_effb3/eval
```

Results go to `outputs/multitask_effb3/`, `outputs/indep_crop_effb3/` and so on, in the same format
as B0: `best.pt`, `last.pt`, `history.csv`, `config.yaml`, and `eval/metrics.json` plus plots.

## Time and memory (RTX 3060 Laptop, 6 GB)

- GPU memory: ~4.9 GB at batch 32 with mixed precision.
- Time: ~108 s per epoch (B0: ~97 s). Loading and augmenting the images takes most of that,
  so the bigger network barely slows things down. A full 30-epoch run takes about 55 minutes, and
  all three runs (multitask + independent crop + independent disease) about 3 hours. Early stopping
  can end a run sooner.
