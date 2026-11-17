# DINOv2: is the backbone frozen during training?

**No.** The DINOv2 ViT-S/14 backbone is **not frozen**. The whole model is fine-tuned, but the
backbone learns with much smaller learning rates than the new crop and disease heads, and its early
layers learn the slowest of all (layer-wise learning-rate decay).

## How the code shows it

- **Nothing is frozen.** There is no `requires_grad = False` or freezing step anywhere in the DINOv2
  code. In [`model.py`](model.py) (`DinoV2Heads.__init__`) the backbone is created with
  `timm.create_model(...)`, and all its weights stay trainable.
- **The backbone is in the optimizer.** `build_optimizer` in [`model.py`](model.py) passes every
  backbone parameter to AdamW (through timm's `param_groups_layer_decay`), then adds the heads as a
  separate group. [`train.py`](train.py) uses this optimizer for all training runs.

## Learning rate of each part

Settings from [`config.yaml`](config.yaml): `lr_head: 1.0e-3`, `lr_backbone: 5.0e-5`, `layer_decay: 0.75`.
Each block gets 0.75x the rate of the block above it. These rates were checked against timm 1.0.30 and
are the peak values; the 2-epoch warmup and the cosine schedule scale all of them up and down together.

| Part | Scale | Learning rate |
|---|---|---|
| Crop and disease heads (new layers) | – | 1.0e-3 |
| Final norm + block 11 (last block) | 1.0 | 5.0e-5 |
| Block 10 | 0.75 | 3.8e-5 |
| Block 9 | 0.56 | 2.8e-5 |
| Block 8 | 0.42 | 2.1e-5 |
| Block 7 | 0.32 | 1.6e-5 |
| Block 6 | 0.24 | 1.2e-5 |
| Block 5 | 0.18 | 8.9e-6 |
| Block 4 | 0.13 | 6.7e-6 |
| Block 3 | 0.10 | 5.0e-6 |
| Block 2 | 0.075 | 3.8e-6 |
| Block 1 | 0.056 | 2.8e-6 |
| Block 0 (first block) | 0.042 | 2.1e-6 |
| Patch embedding, position embedding, CLS token | 0.032 | 1.6e-6 |

So:

- the **early layers**, which hold DINOv2's general visual features (edges, textures, shapes), barely change;
- the **last blocks** adapt more to plant and disease images;
- the **heads** are new and start from random weights, so they learn fastest (20x the last block's rate).

This sits between a frozen backbone (only the heads learn) and full fine-tuning with one learning rate
for everything. It is the usual way to fine-tune a Vision Transformer.

## Other settings that keep the fine-tuning stable

| Setting | Value | Why |
|---|---|---|
| `warmup_epochs` | 2 | Learning rates start near 0, so the random heads don't send large gradients into the pretrained backbone at the start |
| `grad_clip` | 1.0 | Caps the gradient size and stops early instability, common in ViTs |
| `drop_path_rate` | 0.1 | Randomly skips transformer blocks during training (regularisation) |
| `weight_decay` | 0.05 | Usual value for ViTs; biases, norms, position embedding and CLS token get none |
| `amp` | true | Mixed precision, for speed and lower GPU memory |

## Compared with EfficientNet-B0 / B3

|  | EfficientNet-B0 / B3 | DINOv2 ViT-S/14 |
|---|---|---|
| Backbone frozen? | No | No |
| Backbone learning rate | One rate for all layers (2e-4) | Layer-wise decay, 5e-5 at the top down to 1.6e-6 at the bottom |
| Head learning rate | 1e-3 | 1e-3 |
| Weight decay | 0.01 | 0.05 |
| Gradient clipping | No | Yes (1.0) |

## Possible extra experiment

A **frozen-backbone** DINOv2 run (train only the heads; also called a *linear probe*) would show how
good DINOv2's features are for plant diseases without any fine-tuning. Comparing it with the fine-tuned
run shows how much the fine-tuning itself adds.
