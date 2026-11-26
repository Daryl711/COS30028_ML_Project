# Stage 3, run R5: separate crop and disease adapters

Stage 3 base (2b: frozen BioCLIP, taxonomy-text cosine heads, no logit adjustment) plus two residual
adapters, so crop and disease each get their own feature subspace. Nothing else changes from 2b.

```
image (224) -> BioCLIP image encoder (frozen) -> v (512)
   f_c = N(v + A_c(v)) -> Dropout 0.3 -> cos(., T_c) * s -> 55 crop logits
   f_d = N(v + A_d(v)) -> Dropout 0.3 -> cos(., T_d) * s -> 175 disease logits
A_c, A_d: 512 -> 256 -> 512, GELU, dropout 0.1, up-projection zero-init (identity at epoch 0)
T_c, T_d: initialised from BioCLIP text embeddings of the taxonomy prompts, then trained
s: BioCLIP's pretrained logit scale, fixed
```

## Run (from the repo root)

```bash
pip install -r stage3/stage3b/requirements.txt
python -m stage3.stage3b.train                                   # -> outputs/stage3b/
python -m stage3.stage3b.evaluate --checkpoint outputs/stage3b/best.pt
```

Needs `dataset/DPD_Small_dataset/`, the three CSVs, and `dataset/text/plant_descriptions.csv` +
`diseases_descriptions.csv`.

## Checks

- **Epoch 0 must equal 2b's epoch 0** (BioCLIP zero-shot with taxonomy text): the adapters start as the
  identity. `history.csv` row 0 is that number.
- The script asserts 55 crops / 175 diseases, 323 seen + 10 unseen valid pairs; the split is 7,329 train / 671 val,
  the same as the 2b notebook.

## Outputs

| File | Contents |
|---|---|
| `best.pt`, `last.pt` | trainable weights only (T_c, T_d, adapters, logit scale) + config + class names |
| `history.csv` | epoch 0 (zero-shot) and per-epoch train/val metrics |
| `config.yaml` | exact config used |
| `eval/metrics.json` | test_seen / test_unseen metrics (generalized valid-pair protocol), HM, AUSUC, and the crop-leakage probe |

`eval/metrics.json["leakage"]` holds the probe (PROPOSED_METHOD.md §6.5) on the 428 val images whose disease
is on ≥ 2 crops (42 diseases): `r5_f_d` probes the R5 disease feature, `base_N_v` probes N(v), the base's
disease feature. Lower `leak` = less crop information in the disease feature. Val is small, so report `leak_std`.
