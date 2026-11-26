"""
export_effb0_onnx.py - Exports the Stage 1 EfficientNet-B0 baselines to ONNX for the web page.

One ONNX file per baseline, input "image" [1, 3, 224, 224] (already normalised), outputs:
  crop_logits     [1, 55]
  disease_logits  [1, 175]
  embedding       [1, D]      pooled backbone feature(s), for the similarity-ranked gallery

Multi-task:   one backbone, two linear heads (outputs/stage1a/best.pt).
Independent:  the crop model and the disease model, each with its own backbone, bundled
              into one graph (outputs/stage1b_crop/best.pt + outputs/stage1b_disease/best.pt).
              The embedding is both pooled features, each L2-normalised, concatenated.

Usage (from the repo root):
    python UserInterface/tools/export_effb0_onnx.py            # both baselines
    python UserInterface/tools/export_effb0_onnx.py --check    # + compare ONNX with PyTorch on test images
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stage1" / "stage1a"))
from model import build_model  # noqa: E402  (training code, so the architecture matches exactly)

OUT_DIR = REPO / "UserInterface" / "web" / "model"
SIZE = 224

EXPORTS = {
    "effb0_multitask": [REPO / "outputs" / "stage1a" / "best.pt"],
    "effb0_independent": [REPO / "outputs" / "stage1b_crop" / "best.pt",
                          REPO / "outputs" / "stage1b_disease" / "best.pt"],
}


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], ckpt["heads"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def features(model, x):
    """Pooled backbone feature [B, C]."""
    return model.backbone.forward_head(model.backbone.forward_features(x), pre_logits=True)


class MultiTask(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, image):
        pooled = features(self.m, image)
        return self.m.heads["crop"](pooled), self.m.heads["disease"](pooled), pooled


class Independent(nn.Module):
    def __init__(self, crop_model, disease_model):
        super().__init__()
        self.c, self.d = crop_model, disease_model

    def forward(self, image):
        pc, pd = features(self.c, image), features(self.d, image)
        emb = torch.cat([nn.functional.normalize(pc, dim=1), nn.functional.normalize(pd, dim=1)], dim=1)
        return self.c.heads["crop"](pc), self.d.heads["disease"](pd), emb


def build(name):
    models = [load(p) for p in EXPORTS[name]]
    if len(models) == 1:
        return MultiTask(models[0]).eval()
    by_head = {next(iter(m.heads)): m for m in models}
    return Independent(by_head["crop"], by_head["disease"]).eval()


def export(name):
    wrapper = build(name)
    out = OUT_DIR / f"{name}.onnx"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, torch.randn(1, 3, SIZE, SIZE), str(out), dynamo=False, opset_version=17,
            input_names=["image"], output_names=["crop_logits", "disease_logits", "embedding"],
        )
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 2**20:.1f} MB)")
    return wrapper, out


def check(name, wrapper, onnx_path, n=40):
    """ONNX Runtime vs PyTorch on test images with the evaluation transform."""
    import numpy as np
    import onnxruntime as ort
    import pandas as pd
    from PIL import Image
    from torchvision import transforms as T

    tf = T.Compose([T.Resize(int(SIZE / 0.875)), T.CenterCrop(SIZE), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    images = REPO / "dataset" / "DPD_Small_dataset"
    files = pd.concat([pd.read_csv(REPO / "dataset" / f, header=None).iloc[:n // 2, 0]
                       for f in ("DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv")]).tolist()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    diff, agree = 0.0, 0
    for f in files:
        x = tf(Image.open(images / f).convert("RGB"))[None]
        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(x)]
        got = sess.run(None, {"image": x.numpy()})
        diff = max(diff, *(float(np.abs(a - b).max()) for a, b in zip(ref, got)))
        agree += int(ref[0].argmax() == got[0].argmax() and ref[1].argmax() == got[1].argmax())
    print(f"check {name}: {len(files)} test images, max |onnx - torch| = {diff:.2e}, "
          f"top-1 crop & disease agree on {agree}/{len(files)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare ONNX Runtime with PyTorch on test images")
    ap.add_argument("names", nargs="*", default=list(EXPORTS), help=f"subset of {list(EXPORTS)}")
    args = ap.parse_args()
    for name in args.names:
        wrapper, path = export(name)
        if args.check:
            check(name, wrapper, path)


if __name__ == "__main__":
    main()
