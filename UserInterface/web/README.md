---
title: Plant Disease Identifier
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
---

Identifies the crop and disease in a leaf photo, including crop-disease combinations that never appear in the training images (DPD-Small: 55 crops, 175 diseases, 323 seen and 10 unseen combinations). The model runs in the browser with ONNX Runtime Web. The page shows:

- the top-1 crop, disease and composed class, with a seen / unseen (zero-shot) badge, and the top-5 ranking
- the closest training images: same class, same crop, and the same disease on other crops (ranked by similarity when the model exports an embedding)
- an image inspector with zoom / pan and an optional attention heatmap
- a ground-truth check for the sample test images, and a side-by-side comparison of two models
- a benchmarks dialog with every evaluated run (`data/benchmarks.json`, built by `UserInterface/tools/build_benchmarks.py`)

COS30082 Applied Machine Learning · ML Project.
