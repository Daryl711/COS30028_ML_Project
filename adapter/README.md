# BioCLIP + disentangling adapters

Frozen BioCLIP ViT-B/16 (`hf-hub:imageomics/bioclip`) with two post-encoder residual adapters
(crop and disease, 512 -> 256 -> 512, zero-initialized up-projection) feeding a crop head and a
disease head. Only the adapters and heads train (~0.64M of 150M parameters). See `model.py`.

## Training scheme

Training is identical to the DINOv2 baseline (`baseline/MultiTaskBaseline/DINOv2/`), so results
are directly comparable. `config.yaml` copies every data and training value from
`baseline/MultiTaskBaseline/DINOv2/config.yaml`:

- data: `DPD_Small_dataset`, 224 px, 10% per-composition validation split (seed 42)
- augmentation: RandomResizedCrop(0.6–1.0), flips, TrivialAugmentWide, RandomErasing(0.25)
- loss: crop CE + disease CE, weights 1.0 / 1.0, label smoothing 0.1
- AdamW, weight decay 0.05, batch 64, 30 epochs, 2 warmup epochs then cosine decay
- fp16 AMP, gradient clipping 1.0, early stopping (patience 8) on mean val top-1 of both heads

Differences, all marked `adapter` in `config.yaml`:

| Setting | DINOv2 | Adapter |
|---|---|---|
| Backbone | ViT-S/14, fine-tuned | BioCLIP ViT-B/16, frozen |
| Trainable | backbone + heads | adapters + heads |
| Backbone LR / layer decay / drop path | 5e-5 / 0.75 / 0.1 | n/a (frozen) |
| Adapter LR | — | 1e-3 (same as heads) |
| Normalization | ImageNet mean/std | CLIP mean/std (BioCLIP's pretraining stats) |

Only `mode: multitask` is supported, since the adapter model always has both heads.

## Run (from the repo root)

```bash
pip install -r adapter/requirements.txt
python -m adapter.train
python -m adapter.evaluate --checkpoint outputs/multitask_bioclip_adapter/best.pt
```

Training writes `best.pt`, `last.pt`, `history.csv` and `config.yaml` to
`outputs/multitask_bioclip_adapter/`. Evaluation writes `eval/metrics.json` there: seen/unseen
pair top-1/top-5, crop and disease accuracy, class-balanced unseen accuracy, HM and AUSUC. Pair
scores are `log p(crop) + log p(disease)` over valid pairs only (generalized protocol): the
training compositions plus the `test_unseen` compositions, the same candidate set as the
MultiTask baselines. The AUSUC bias is added to the unseen-pair scores only.

The previous three-seed, two-phase protocol is kept in `_backup_three_seed_protocol/`.

## Files

- `config.yaml`: all settings.
- `model.py`: BioCLIP loading, adapters, two heads.
- `dataset.py`: CSV loading, validation split, transforms (same as DINOv2 except normalization).
- `train.py`: training loop (same as DINOv2's).
- `evaluate.py`: test-set evaluation.
- `metrics.py`: pair scoring, HM, AUSUC.
