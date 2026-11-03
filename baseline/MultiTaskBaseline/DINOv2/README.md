# DINOv2 ViT-S/14 baseline

The same two-head design as the B0 baseline, but the EfficientNet backbone is replaced with a
**Vision Transformer (ViT)** pretrained by **DINOv2**. Data, augmentation, image size (224 px),
heads, loss, schedule and batch size match B0. The other changes are the settings a transformer
needs to fine-tune well (see the table below).

This experiment asks: **do general-purpose, self-supervised features generalize better to unseen
crop-disease pairs than ImageNet-classifier features?**

## How the model works

```
224×224 image
   │
   ▼
Cut into a 16×16 grid of 14×14-pixel patches  (256 patches)
   │
   ▼
Each patch → Linear layer → 384-number token, plus a position embedding (where the patch was)
Add one extra learnable [CLS] token at the front                  → 257 tokens
   │
   ▼
12 transformer blocks, each:
   LayerNorm → multi-head self-attention (6 heads) → add back
   LayerNorm → MLP (384 → 1536 → 384)             → add back
   │
   ▼
Final LayerNorm, take the [CLS] token → 384-number feature vector
   ├─ Dropout(0.3) → Linear(384 → 55)   → crop logits
   └─ Dropout(0.3) → Linear(384 → 175)  → disease logits
```

### The backbone: a Vision Transformer

A CNN like EfficientNet looks at small neighbourhoods first and only sees the whole leaf in its last
layers. A ViT works differently:

- The image is split into **patches** that are treated like words in a sentence.
- **Self-attention** lets every patch look at every other patch from the very first layer. A lesion
  patch can be compared directly with healthy leaf tissue, leaf edges or other lesions anywhere in
  the image.
- The **[CLS] token** has no image content of its own. Through attention it gathers a summary of all
  patches, and its final value is the feature vector the heads read.

"ViT-S/14" means the **Small** size (384-dim tokens, 12 blocks, 6 attention heads, 22M parameters)
with **14-pixel** patches.

### The pretraining: DINOv2

This is the main reason to try this model. B0 and B3 were pretrained to **classify** ImageNet's 1,000
labels, so their features are tuned to separate those labels. DINOv2 (Meta, 2023) was trained with
**no labels at all**, on 142 million curated images (the LVD-142M set):

1. **Self-distillation:** there are two copies of the network, a student and a teacher. The teacher
   is a slow moving average of the student. Both see different random crops of the same image, and
   the student must produce the same output as the teacher. To do that, the network has to describe
   *what is in the image* in a way that survives cropping, scaling and colour changes.
2. **Masked patches (iBOT):** some patches are hidden from the student, and it must predict the
   teacher's features for them. This forces detailed, patch-level understanding of texture and
   shape, which suits spots, mildew and rust pustules.

The result is a general description of visual content: texture, colour patterns, shapes. DINOv2
features are known to work well on fine-grained natural images, including plants, often even when
the backbone is frozen.

### Why it might help unseen pairs

B0 gets ~10% disease accuracy on unseen pairs. Its disease head learned shortcuts like "a tomato leaf
means a tomato disease", so early blight on tomato, never seen in training, gets labelled as a
disease tomato *did* have in training.

DINOv2 starts from features that describe **what the lesion looks like**, not **which class it
belongs to**. If the features for "rust pustules" are similar whether they sit on wheat or sugarcane,
the disease head can recognise rust on sugarcane even though that pair never appeared in training.
Layer-wise learning-rate decay (below) helps keep those general features during fine-tuning.

This is a hypothesis to test, not a guarantee. Fine-tuning on the seen pairs can still teach the
same crop shortcut.

## What changed from B0

| Setting | B0 | DINOv2 | Why |
|---|---|---|---|
| `backbone` | `efficientnet_b0` | `vit_small_patch14_dinov2.lvd142m` | the experiment |
| `img_size` | 224 | 224 (passed to the model) | timm builds DINOv2 for 518 px by default; `model.py` passes 224 and timm resizes the pretrained position embeddings to a 16×16 grid |
| `lr_backbone` | 2e-4 | 5e-5 (last block) | transformers are easily damaged by large updates |
| `layer_decay` | (none) | 0.75 | each earlier block learns 0.75× as fast as the block above it (see below) |
| `weight_decay` | 0.01 | 0.05, none on norms, biases, position embedding, [CLS] | standard for ViT fine-tuning |
| `drop_path_rate` | 0.2 | 0.1 | standard for ViT-S; randomly skips whole blocks during training |
| `grad_clip` | off | 1.0 | caps the gradient size; stops the occasional large update that can destabilize a ViT |

Unchanged: batch 64, 30 epochs, 2 warmup epochs + cosine decay, head learning rate 1e-3, head
dropout 0.3, label smoothing 0.1, mixed precision, early stopping (patience 8), augmentation and the
ImageNet normalization (DINOv2 uses the same mean and std).

### Layer-wise learning-rate decay

Early layers hold the most general features (edges, textures). Late layers are the most specific to
the pretraining task. So the backbone's learning rate shrinks layer by layer going down:

| Part of the backbone | Learning rate |
|---|---|
| Block 12 (last) + final norm | 5.0e-5 |
| Block 9 | 2.1e-5 |
| Block 6 | 8.9e-6 |
| Block 3 | 3.8e-6 |
| Block 1 (first) | 2.1e-6 |
| Patch embedding, position embedding, [CLS] | 1.6e-6 |
| Crop and disease heads (new) | 1.0e-3 |

These are the peak values. Warmup and cosine decay scale them all over training, as in B0.

## Files

| File | What changed from `../InitialAttempt(Effb0)` |
|---|---|
| `config.yaml` | new: DINOv2 settings (above) |
| `model.py` | `DinoV2Heads` builds the ViT at `img_size`; `build_optimizer()` adds layer-wise learning-rate decay |
| `train.py` | reads `config.yaml` next to it by default; supports `accum_steps` and `grad_clip` |
| `evaluate.py`, `plot_confusions.py` | usage paths only |
| `dataset.py`, `metrics.py` | unchanged |

## How to run

Run from the repo root (the folder that contains `dataset/`).

```bash
# multi-task (one model, two heads)
python baseline/MultiTaskBaseline/DINOv2/train.py

# independent baseline (two models)
python baseline/MultiTaskBaseline/DINOv2/train.py --mode crop    --experiment indep_crop_dinov2
python baseline/MultiTaskBaseline/DINOv2/train.py --mode disease --experiment indep_disease_dinov2

# evaluate
python baseline/MultiTaskBaseline/DINOv2/evaluate.py --checkpoints outputs/multitask_dinov2/best.pt --out outputs/multitask_dinov2/eval
python baseline/MultiTaskBaseline/DINOv2/evaluate.py --checkpoints outputs/indep_crop_dinov2/best.pt outputs/indep_disease_dinov2/best.pt --out outputs/indep_dinov2/eval
```

The first run downloads the pretrained weights (~85 MB) from Hugging Face. Results go to
`outputs/multitask_dinov2/`, `outputs/indep_crop_dinov2/` and so on, in the same format as B0.

## Time and memory (RTX 3060 Laptop, 6 GB)

- GPU memory: ~3.3 GB at batch 64 with mixed precision.
- Time: ~94 s per epoch (B0: ~97 s). Loading and augmenting the images takes most of that.
  A full 30-epoch run takes about 47 minutes, and all three runs (multitask + independent crop +
  independent disease) about 2.5 hours. Early stopping can end a run sooner.
