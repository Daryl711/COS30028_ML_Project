# Model folder

The page reads `model/config.json`. While its `models` list is empty, the page works without predictions and shows "No model deployed yet".

## Adding a model

1. Export the PyTorch model to ONNX. The ONNX file must take one float32 input of shape `[1, 3, S, S]`, already normalised, and return either:
   - two outputs: crop logits `[1, 55]` and disease logits `[1, 175]`, or
   - one output: composition logits `[1, 333]`, in composition-id order (column 4 of the CSVs).
2. Put the `.onnx` file in this folder.
3. Copy the entry from `config.example.json` into `config.json` and fill it in. The first model in the list is loaded by default. Each further model adds a choice to the model dropdown.

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
| `outputs` | ONNX output names: `crop` and `disease`, or `composition` |
| `unseen_bias` | Added to unseen compositions' scores (calibration). Default 0 |
| `metrics` | List of `{label, value}` tiles shown under the samples |
| `about` | List of paragraphs for the About dialog |

The preprocessing must match the Python evaluation pipeline. After exporting, compare the browser's top-5 with PyTorch's on a few test images.

Files over 10 MB need Git LFS on Hugging Face. `huggingface-cli upload` handles this automatically.
