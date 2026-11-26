# Stage 3, run R5 + R6: conditional crop-invariance on the disease feature

R5 (`stage3/stage3b/`) plus a crop discriminator on the disease feature f_d behind a gradient-reversal
layer (GRL). Nothing else changes: same data, split, schedule, model selection and evaluation as R5.
The classification path is R5's, so prediction and every test metric are computed exactly as for R5.

```
image (224) -> BioCLIP image encoder (frozen) -> v (512)
   f_c = N(v + A_c(v)) -> crop text head            (R5)
   f_d = N(v + A_d(v)) -> disease text head         (R5)
   f_d -> GRL(lambda) -> D: 512 -> 256 -> ReLU -> 55     (R6, training only)
          softmax restricted to C(d) = crops carrying the image's disease (train split)
          only for images whose disease is on >= 2 crops (42 diseases, ~63% of train images)

loss = crop CE + disease CE (label smoothing 0.1) + L_inv (plain CE)
lambda(p) = lambda_max * (2 / (1 + e^{-10p}) - 1),  p = step / total steps,  lambda_max = 0.1
```

The discriminator learns to tell the crop apart *within* a disease ("potato or tomato?" for an
early-blight image); the reversed gradient trains A_d to make that impossible. Single-crop diseases are
left alone, because there the disease determines the crop.

## Run (from the repo root)

```bash
python -m stage3.stage3c.train                                   # -> outputs/stage3c/
python -m stage3.stage3c.evaluate --checkpoint outputs/stage3c/best.pt

# optional rows (PROPOSED_METHOD.md §5, §7)
python -m stage3.stage3c.train --lambda_max 0.3 --experiment stage3c_lam0.3   # lambda sweep {0, 0.03, 0.1, 0.3}
python -m stage3.stage3c.train --unconditional --experiment stage3c_uncond    # plain DANN ablation
```

### Sweep + ablation in one go (`sweep.py`)

`sweep.py` trains each optional row with the unchanged `train.py`, then scores it with `evaluate.py`.
Each run is its own process. At the end it writes a table comparing every run with R5 and the main
λ 0.1 run to `outputs/stage3c_sweep_summary.md`.

```bash
python -m stage3.stage3c.sweep                       # lam0, lam0.03, lam0.3, uncond (~45 min each)
python -m stage3.stage3c.sweep --runs uncond         # any subset: lam0 lam0.03 lam0.3 uncond
python -m stage3.stage3c.sweep --summary_only        # just rebuild the table
```

| Key | Experiment dir | Setting |
|---|---|---|
| `lam0` | `outputs/stage3c_lam0/` | conditional, λ_max 0 (sanity: should match R5) |
| `lam0.03` | `outputs/stage3c_lam0.03/` | conditional, λ_max 0.03 |
| `lam0.3` | `outputs/stage3c_lam0.3/` | conditional, λ_max 0.3 |
| `uncond` | `outputs/stage3c_uncond/` | unconditional DANN, λ_max 0.1 |

Runs that already have `eval/metrics.json` are skipped, so an interrupted sweep resumes where it left off.
Pass `--force` to retrain them.

`--lambda_max 0` is a sanity row: the discriminator still trains but no gradient reaches A_d, so it
should reproduce R5.

## Checks

- Epoch 0 equals R5's (and 2b's) epoch 0: the adapters start as the identity.
- The script asserts 42 multi-crop diseases.
- `history.csv` adds `train_inv_loss`, `train_disc_acc` (discriminator accuracy within C(d)) and `lambda`.
  A disc accuracy that stays near the majority-crop rate means the GRL is winning.

## Outputs

As R5. `eval/metrics.json["leakage"]` holds the probe on f_d as `r5r6_f_d` (majority-crop baseline,
train-split mapping, scored on val) and the base N(v) as `base_N_v`. Compare `r5r6_f_d.leak` with
`outputs/stage3b/eval/metrics.json["leakage"]["r5_f_d"]["leak"]`.
