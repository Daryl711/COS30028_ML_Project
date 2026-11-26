# Plant Disease Identifier (web UI)

A static web page that takes a leaf photo, predicts its crop-disease class (55 crops, 175 diseases, 333 combinations, 10 of them unseen in training) and shows the training images it matches most closely. The models run in the browser with ONNX Runtime Web, so there is no server and the photo never leaves the device.

```text
UserInterface/
├── web/              the page (index.html, app.js, style.css), models, data and thumbnails
├── tools/            scripts that build the page's data and export the models to ONNX
└── WEB_UI_GUIDE.md   full guide: interface, prediction pipeline, ONNX export
```

## Run it locally

Browsers block `fetch()` from `file://` pages, so serve the folder over HTTP:

```bash
cd UserInterface/web
python -m http.server 8000
# open http://localhost:8000
```

Upload or drop a leaf photo, or pick a sample test image. The page shows:

- the top-1 crop, disease and crop-disease class, with a seen / unseen badge, and the top-5 ranking
- the closest training images (nearest overall, same class, same crop, same disease on other crops), ranked by cosine similarity of the model's embeddings
- an image inspector with zoom / pan, a side-by-side model comparison and a benchmarks dialog

## Rebuild the data and models

From the repo root (needs `torch`, `timm`, `open_clip_torch`, `onnx`, `onnxruntime`, `numpy`, `pandas`, `pillow`):

```bash
python UserInterface/tools/build_web_data.py                       # classes.json, gallery thumbnails, sample images
python UserInterface/tools/export_s3_all_onnx.py --check           # custom approach (s3_all) + full test-set check
python UserInterface/tools/export_effb0_onnx.py --check            # baselines + ONNX-vs-PyTorch check
python UserInterface/tools/export_gallery_embeddings.py s3_all
python UserInterface/tools/export_gallery_embeddings.py effb0_independent
python UserInterface/tools/export_gallery_embeddings.py effb0_multitask
python UserInterface/tools/build_benchmarks.py                     # benchmarks dialog data
```

`web/model/README.md` explains how to add another model, and `WEB_UI_GUIDE.md` covers the rest in detail.
