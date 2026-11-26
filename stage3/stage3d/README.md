# Stage 3, runs R4 + R5 and All (R4 + R5 + R6)

`stage3/stage3d/` (R4 + R5) and `stage3/stage3e/` (All) combine the existing Stage 3 modules unchanged:
R4's prompt learner and L_kg (`stage3/stage3a/model.py`), R5's adapters (`stage3/stage3b/model.py`) and,
for All, R6's GRL, DANN ramp and crop mask (`stage3/stage3c/model.py`). Data, split, augmentation,
schedule, model selection and evaluator are the same as every other Stage 3 run.

```
image (224) -> BioCLIP image encoder (frozen) -> v (512)
   f_c = N(v + A_c(v)) -> s * cos(Dropout(f_c), T_c)          (R5 image side)
   f_d = N(v + A_d(v)) -> s * cos(Dropout(f_d), T_d)
   T_c, T_d = learnable prompts through the frozen text encoder (R4 text side)
   All only: f_d -> GRL(lambda) -> D: 512 -> 256 -> 55, softmax over C(d), diseases on >= 2 crops   (R6)

R4 + R5: loss = crop CE + disease CE + λ_kg · L_kg
All    : loss = crop CE + disease CE + λ_kg · L_kg + L_inv
```

## Run (from the repo root)

```bash
python -m stage3.stage3d.train                                  # -> outputs/stage3d/
python -m stage3.stage3d.evaluate --checkpoint outputs/stage3d/best.pt

python -m stage3.stage3e.train                                   # -> outputs/stage3e/
python -m stage3.stage3e.evaluate --checkpoint outputs/stage3e/best.pt

# options (both packages): --epochs, --experiment, --lambda_kg, --lr_prompt
# All only: --lambda_max, --unconditional
python -m stage3.stage3d.train --lr_prompt 1e-4 --experiment stage3d_lrp1e-4
```

One training loop serves both: `stage3e/train.py` calls `stage3d/train.py:main()` with its own config, and
the `r6:` section of that config switches on the discriminator and L_inv.

## Design choices

- **Token LR.** Every module uses `lr_new` 1e-3 (guide.md). `lr_prompt` (default `null` = `lr_new`) sets
  only the prompt tokens' LR, so if the R4 token-LR sweep (`python -m stage3.stage3a.sweep`) adopts a lower
  value, guide.md's rule ("it applies to that module in every run that contains it") can be followed
  without touching the adapters' or discriminator's LR. Any such run is a named sub-row.
- **Epoch 0 equals 2b's.** Token deltas are zero and the adapters are the identity, so `history.csv`
  row 0 should match 2b / R4 / R5 (val loss 12.79334, crop 0.4769, disease 0.0209).
- **Checkpoint.** `model` is an ordinary R5Net: this epoch's prompt output as T_c, T_d plus the adapters.
  So evaluation is `stage3/stage3b/evaluate.py` unchanged and needs no text encoder. `net` holds every
  trainable part (prompt deltas, adapters, and the discriminator for All).
- **Leakage probe** on f_d = N(v + A_d(v)): `leakage.r4r5_f_d` / `leakage.all_f_d`.

## Outputs

`history.csv` has R4's columns (`train_kg_loss`, `drift_crop`, `drift_disease`), plus `lr_prompt`, and for All
R6's columns (`train_inv_loss`, `train_disc_acc`, `lambda`). `eval/metrics.json` has the same format as R5.
