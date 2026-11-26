# Model folder

## Deployed models

| File | Model | Source checkpoints |
|---|---|---|
| `s3_all.onnx` (331 MB, fp32) | Custom approach: S3 All, frozen BioCLIP ViT-B/16 + learnt prompts (R4), crop / disease adapters (R5), crop-invariance GRL (R6) | `outputs/s3_all/best.pt` |
| `effb0_independent.onnx` (32 MB) | Baseline 1: independent EfficientNet-B0 crop + disease networks | `outputs/s1_crop_effb0/best.pt`, `outputs/s1_disease_effb0/best.pt` |
| `effb0_multitask.onnx` (16 MB) | Baseline 2: multi-task EfficientNet-B0 | `outputs/s1_mt_effb0/best.pt` |
| `*_gallery.f16` | Gallery embeddings (float16) for the similarity-ranked training matches | built from the ONNX files |

Each file returns `crop_logits`, `disease_logits` and `embedding`.

`s3_all.onnx` is over GitHub's 100 MB file limit, so it is gitignored and hosted on Hugging Face only (rebuild it with `export_s3_all_onnx.py`). It is fp32 so it matches `best.pt` exactly: its full test-set accuracy reproduces `outputs/s3_all/eval/metrics.json` (calibrated HM 59.3% vs 59.4%). int8 quantisation lost 1.9 points of HM and was not used. `unseen_bias` in `config.json` is each run's calibrated best bias from `outputs/<run>/eval/metrics.json`.

To rebuild them (from the repo root; needs `onnx` and `onnxruntime`):

```bash
python UserInterface/tools/export_s3_all_onnx.py --check         # custom approach + full test-set check (needs open_clip_torch)
python UserInterface/tools/export_effb0_onnx.py --check          # baselines + ONNX-vs-PyTorch check
python UserInterface/tools/export_gallery_embeddings.py s3_all
python UserInterface/tools/export_gallery_embeddings.py effb0_independent
python UserInterface/tools/export_gallery_embeddings.py effb0_multitask
python UserInterface/tools/build_benchmarks.py                     # benchmarks dialog data
```

The page reproduces the evaluation transform exactly (PIL bilinear resize, centre crop with Python rounding, EXIF orientation ignored as in training). On 120 test images its top-5 matches the PyTorch pipeline for both models.


The page reads `model/config.json`. While its `models` list is empty, the page works without predictions and shows "No model deployed yet".

## Adding a model

1. Export the PyTorch model to ONNX. The ONNX file must take one float32 input of shape `[1, 3, S, S]`, already normalised, and return either:
   - two outputs: crop logits `[1, 55]` and disease logits `[1, 175]`, or
   - one output: composition logits `[1, 333]`, in composition-id order (column 4 of the CSVs).
2. Put the `.onnx` file in this folder.
3. Copy the entry from `config.example.json` into `config.json` and fill it in. The first model in the list is loaded by default. Each model adds a choice to the model dropdown, and with two or more the **Compare models** button runs two of them side by side on the same image.

### Optional output

The ONNX file may return one extra output; a model without it still works.

- **Embedding** `[1, D]` image embedding. Together with `gallery_embeddings` it ranks the closest training matches by cosine similarity and shows a similarity badge on each. Build the file with:

  ```bash
  pip install onnxruntime numpy pillow
  python UserInterface/tools/export_gallery_embeddings.py <model id>
  ```

## config.json fields

| Field | Meaning |
|---|---|
| `id` | Short unique id |
| `name`, `short_name` | Name in the dropdown and About dialog; short name in the results row |
| `file` | Path of the `.onnx` file, relative to the page |
| `input` | Optional input name. Defaults to the model's first input |
| `input_size` | `S` in `[1, 3, S, S]` |
| `resize` | `"squash"` = `Resize((S, S))`. `"center_crop"` = `Resize(S / crop_pct)` then `CenterCrop(S)` |
| `crop_pct` | For `center_crop` only. Default 0.875 |
| `mean`, `std` | Normalisation. Defaults to ImageNet. CLIP/BioCLIP use `[0.4815, 0.4578, 0.4082]` / `[0.2686, 0.2613, 0.2758]` |
| `role` | Optional group shown before the name, e.g. `Custom approach`, `Baseline 1`, `Baseline 2` |
| `benchmark_id` | Optional run folder in `outputs/` (e.g. `s3_r5`); marks that row as deployed in the benchmarks dialog |
| `outputs` | ONNX output names: `crop` and `disease`, or `composition`; optionally `embedding` |
| `gallery_embeddings` | Optional float32 file from `export_gallery_embeddings.py` (one row per gallery thumbnail) |
| `unseen_bias` | Added to unseen compositions' scores (calibration). Default 0 |
| `metrics` | List of `{label, value}` tiles shown under the samples |
| `about` | List of paragraphs for the About dialog |

The preprocessing must match the Python evaluation pipeline. After exporting, compare the browser's top-5 with PyTorch's on a few test images.

Files over 10 MB need Git LFS on Hugging Face. `huggingface-cli upload` handles this automatically.
