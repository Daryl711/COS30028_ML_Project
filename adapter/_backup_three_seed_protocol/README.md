# BioCLIP sequential bottleneck adapters with DPD compositional evaluation

This runner uses the OpenCLIP BioCLIP checkpoint `hf-hub:imageomics/bioclip`, installs sequential
bottleneck adapters after the original MLP branch in image blocks 8–11, freezes all original
parameters, and trains crop and disease heads. BioCLIP's final normalization, CLS pooling, and
pretrained visual projection remain in the feature path.

## Validation and training protocol

`train.py` reads the DINOv2-compatible seven-column CSVs. It deterministically creates validation
from `DPD_Small_train.csv` with seed 0: eligible whole crop-disease compositions become
`val_unseen`; 10% (floor per composition) of the remaining pairs become `val_seen`. It checks the
41-candidate count, component overlap constraints, label-space dimensions (55 crops, 175 diseases),
and expected split sizes (6,739 / 723 / 538). A mismatch stops the run so a changed source CSV or
selection convention cannot silently change the benchmark.

For each of three configured seeds, phase one trains on the 6,739-image dev train set and selects
the epoch with highest uncalibrated validation HM from full-label-space top-1 on `val_seen` and
`val_unseen`; patience is measured in epochs without HM improvement. Phase two initializes a fresh
BioCLIP/adapters/heads model and trains on the full 8,000-image train CSV for exactly the selected
epoch count. It then evaluates `test_seen` and `test_unseen` once. Test data does not affect epoch
selection or final fitting.

Each training image contributes ten deterministic views: one clean view and nine augmented views.
Augmentations are generated on demand using
`base_seed + image_id * 100 + view_id`; the source CSV row ID is retained across train/validation
filtering, and the seed is reset for each image/view. Thus each view is repeatable across epochs
without materializing image tensors or post-BioCLIP features on disk. Test and validation use one
clean view per image.

For each image, pair scores are `log p(crop) + log p(disease)` and ranking considers every one of
the 55 x 175 label combinations, including compositions absent from train and evaluation splits.
Reported metrics include seen/unseen pair top-1 and top-5, unseen crop and disease accuracy,
class-balanced unseen pair accuracy, HM, and AUSUC. AUSUC sweeps a bias added to every pair absent
from the relevant training composition set. Results are written per seed and as mean ± sample std
in `three_seed_summary.json`.

## Data layout

The default interface matches `baseline/MultiTaskBaseline/DINOv2/dataset.py`:

```text
dataset/
  DPD_Small_256/                  # current config; files referenced by each CSV's image column
  DPD_Small_train.csv
  DPD_Small_test_seen.csv
  DPD_Small_test_unseen.csv
```

Each CSV has columns `image,crop_label,disease_label,comp_label,crop_name,disease_name,comp_name`
(with or without a header). Paths in `image` are relative to `image_dir`. Update the dataclasses in
`config.py` to adjust paths, seeds, views, batch size, optimizer, schedule, or output location.

## Install and run

Install a PyTorch/torchvision build appropriate for the machine's CUDA version, then install the
remaining dependencies and launch from the repository root:

```bash
pip install -r adapter/requirements.txt
python -m adapter.train
```

The first launch downloads BioCLIP weights. Three seeds `(0, 1, 2)` run by default; `--seeds 3 4 5`
overrides them but still requires exactly three values. BF16 autocast is enabled on CUDA when
supported. Per-seed dev checkpoints, final checkpoints, metrics, augmentation manifests, and the
aggregate report are saved under `outputs/bioclip_adapter/`.

## Files

- `config.py`: model, data, and experiment settings.
- `model.py`: BioCLIP loading, adapter block wrapper, and two heads.
- `dataset.py`: CSV loading, seed-0 compositional split, transforms, fixed multi-view dataset.
- `metrics.py`: open-world pair scoring, HM, and AUSUC.
- `train.py`: two-phase three-seed fit, validation selection, and final evaluation.
- `utils.py`: checkpointing and parameter counts.
