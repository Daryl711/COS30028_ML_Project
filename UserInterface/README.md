# Multi-task Crop & Disease Classifier

Gradio interface for multi-task plant leaf classification: predicts **crop** and **disease** together, with Top-5 rankings, score tables, harmonic-mean confidence, calibration indicators, and ranking curves.

---

## Model weights (not in Git)

The trained weight files are **gitignored** (large binary files). They are **not** stored in this repository. They are also testing weights for the skeleton.

### Download weights

Download the weights from Google Drive and place them in the same folder as `app.py` (e.g. `UserInterface/`):

| File | Description |
|------|-------------|
| `best.pt` | Multi-task DINOv2 backbone + crop/disease linear heads |
| `heads_seed2.pt` | Prototype classification heads (seed 2) |

**Google Drive folder:**  
*[Add your Google Drive link here]*

After downloading, your folder should look like:

```text
UserInterface/
├── app.py
├── requirements.txt
├── weights/MultitaskDinoV2/best.pt              ← from Google Drive
├── weights/R4_CSP_KgCOOP/heads_seed2.pt       ← from Google Drive
└── README.md
```

### Optional class name files

If you have label lists from training, add:

- `crop_labels.txt` — one crop name per line (55 lines)
- `disease_labels.txt` — one disease name per line (175 lines)

Without these, the app shows `Crop_0`, `Disease_0`, etc.

---

## Setup

### 1. Create / activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```text
gradio>=4.0.0
torch
torchvision
timm
Pillow
matplotlib
numpy>=1.20
accelerate
```

### 3. Place the weight files

Put `best.pt` and `heads_seed2.pt` next to `app.py` (see Google Drive section above).

---

## How to start the app

From the folder that contains `app.py`:

```bash
python app.py
```

Then open the local URL shown in the terminal (usually):

```text
http://127.0.0.1:7860
```

To share a temporary public link:

```python
# at the bottom of app.py
demo.launch(share=True)
```

---

## How to use the app

1. **Select a model**
   - **Multi-task DINOv2 (best.pt)** — full backbone + linear heads  
   - **Prototype Heads (heads_seed2.pt)** — prototype classifier on projected features  

2. **Upload a leaf image** (RGB photo of a plant leaf).

3. Click **Predict**.

4. Review the outputs:
   - **Summary** — top crop, top disease, harmonic mean confidence, soft ranking AUC  
   - **Crop Top-5** / **Disease Top-5** — bar charts (stacked top–bottom)  
   - **Score tables** — Rank | Class | Score | Confidence %  
   - **Calibration & Reliability** — task confidence, uncertainty, top-5 concentration  
   - **Ranking / Soft-AUC curves** — illustrative ranking curves (true ROC-AUC needs a labeled validation set)

---

## Project notes

- Backbone: `vit_small_patch14_dinov2.lvd142m` (224×224) via `timm`
- Tasks: 55 crop classes + 175 disease classes
- Checkpoint `best.pt` reports ~98.5% crop / ~91.2% disease validation accuracy (from training metadata)

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `FileNotFoundError: best.pt` | Download weights from Google Drive into the `app.py` folder |
| `module 'numpy' has no attribute 'trapz'` | Use the current `app.py` (uses a manual trapezoidal integral) |
| Table shows empty / wrong values | Confirm you have the latest `app.py` (tables use list-of-lists) |
| CUDA / GPU errors | App falls back to CPU automatically |
| Slow first run | `timm` may download backbone config on first load; weights themselves are local |

---

## License

Use according to your course / project rules (COS30028). Model weights and data licensing follow their original sources.
