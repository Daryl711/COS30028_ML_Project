"""
export_gallery_embeddings.py - Embeds every gallery thumbnail with a deployed ONNX model.

The web page ranks "Closest training matches" by cosine similarity when a model
config has an `embedding` output and a `gallery_embeddings` file. This script
writes that file: raw little-endian float16 (set "gallery_dtype": "float16" in
config.json) or float32 with --float32, one row per gallery thumbnail, in the
order the page flattens them (data/classes.json compositions in id order, then
each composition's `thumbs` list in order). Rows are L2-normalised first, so
float16 loses nothing that matters for cosine similarity.

Each thumbnail is embedded from its full-size original in dataset/DPD_Small_dataset
when it exists (the thumbnails are only 220 px), using the same preprocessing as
app.js (squash or center_crop, then mean / std).

Needs: pip install onnxruntime numpy pillow torchvision

Usage (from the repo root):
    python UserInterface/tools/export_gallery_embeddings.py <model id from config.json> [--float32]
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "UserInterface" / "web"
ORIGINALS = REPO / "dataset" / "DPD_Small_dataset"
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def preprocess(path, cfg):
    """The evaluation transform of the training code (torchvision, bilinear):
    center_crop = Resize(s / crop_pct) + CenterCrop(s); squash = Resize((s, s))."""
    from torchvision import transforms as T
    img = Image.open(path).convert("RGB")
    s = cfg["input_size"]
    if cfg.get("resize") == "center_crop":
        img = T.CenterCrop(s)(T.Resize(int(s / cfg.get("crop_pct", 0.875)))(img))
    else:
        img = T.Resize((s, s))(img)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(cfg.get("mean", IMAGENET[0]), dtype=np.float32)
    std = np.array(cfg.get("std", IMAGENET[1]), dtype=np.float32)
    return ((x - mean) / std).transpose(2, 0, 1)[None]


def main():
    args = [a for a in sys.argv[1:] if a != "--float32"]
    dtype = "<f4" if "--float32" in sys.argv else "<f2"
    if len(args) != 1:
        sys.exit(__doc__)
    models = json.loads((WEB / "model" / "config.json").read_text(encoding="utf-8"))["models"]
    cfg = next((m for m in models if m["id"] == args[0]), None)
    if cfg is None:
        sys.exit(f"no model with id {args[0]!r} in model/config.json")
    out_name = cfg.get("outputs", {}).get("embedding")
    if not out_name:
        sys.exit("this model has no outputs.embedding in config.json")
    out_file = WEB / cfg.get("gallery_embeddings", f"model/{cfg['id']}_gallery.{'f16' if dtype == '<f2' else 'f32'}")

    sess = ort.InferenceSession(str(WEB / cfg["file"]), providers=["CPUExecutionProvider"])
    in_name = cfg.get("input", sess.get_inputs()[0].name)

    classes = json.loads((WEB / "data" / "classes.json").read_text(encoding="utf-8"))
    thumbs = [t for c in classes["compositions"] for t in c["thumbs"]]
    # Thumbnails are always .jpg; the originals may be .jpeg / .png, so match on the file stem
    originals = {p.stem: p for p in ORIGINALS.iterdir()} if ORIGINALS.is_dir() else {}
    rows, from_thumb = [], 0
    for i, t in enumerate(thumbs, 1):
        original = originals.get(Path(t).stem)
        if original is None:
            original, from_thumb = WEB / t, from_thumb + 1
        emb = sess.run([out_name], {in_name: preprocess(original, cfg)})[0].reshape(-1)
        rows.append((emb / (np.linalg.norm(emb) or 1.0)).astype(dtype))
        if i % 100 == 0 or i == len(thumbs):
            print(f"\r{i}/{len(thumbs)}", end="", flush=True)
    print()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.stack(rows).tofile(out_file)
    print(f"wrote {out_file.relative_to(REPO)}: {len(rows)} rows x {rows[0].size} dims")
    if from_thumb:
        print(f"note: {from_thumb} images were embedded from the 220 px thumbnail (original not found)")
    want = {"gallery_embeddings": out_file.relative_to(WEB).as_posix(), "gallery_dtype": "float16" if dtype == "<f2" else "float32"}
    for k, v in want.items():
        if cfg.get(k, "float32" if k == "gallery_dtype" else None) != v:
            print(f'set  "{k}": "{v}"  for this model in config.json')


if __name__ == "__main__":
    main()
