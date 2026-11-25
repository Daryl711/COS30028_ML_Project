"""
export_effb0_onnx.py - Exports the Stage 1 EfficientNet-B0 baselines to ONNX for the web page.

One ONNX file per baseline, input "image" [1, 3, 224, 224] (already normalised), outputs:
  crop_logits     [1, 55]
  disease_logits  [1, 175]
  heatmap         [1, 7, 7]   class activation map of the disease head (see below)
  embedding       [1, D]      pooled backbone feature(s), for the similarity-ranked gallery

Multi-task:   one backbone, two linear heads (outputs/s1_mt_effb0/best.pt).
Independent:  the crop model and the disease model, each with its own backbone, bundled
              into one graph (outputs/s1_crop_effb0/best.pt + outputs/s1_disease_effb0/best.pt).
              The embedding is both pooled features, each L2-normalised, concatenated.

Heatmap: the heads are linear on global-average-pooled features, so the class activation
map (CAM) of class d is w_d . F[:, y, x] over the last feature map F, which is also what
Grad-CAM gives for this architecture. The map is weighted by the disease probabilities,
sum_d p(d) * CAM_d, so it follows the model's disease belief and needs no argmax in the
graph. Negative evidence is clipped to 0.

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
sys.path.insert(0, str(REPO / "baseline" / "MultiTaskBaseline" / "EffB0"))
from model import build_model  # noqa: E402  (training code, so the architecture matches exactly)

OUT_DIR = REPO / "UserInterface" / "web" / "model"
SIZE = 224

EXPORTS = {
    "effb0_multitask": [REPO / "outputs" / "s1_mt_effb0" / "best.pt"],
    "effb0_independent": [REPO / "outputs" / "s1_crop_effb0" / "best.pt",
                          REPO / "outputs" / "s1_disease_effb0" / "best.pt"],
}


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], ckpt["heads"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def linear(head):
    """The nn.Linear inside a head (Dropout -> Linear)."""
    return [m for m in head if isinstance(m, nn.Linear)][0]


def features(model, x):
    """Last feature map [B, C, h, w] and its pooled vector [B, C]."""
    fmap = model.backbone.forward_features(x)
    return fmap, model.backbone.forward_head(fmap, pre_logits=True)


def disease_cam(fmap, disease_logits, head):
    w = linear(head).weight                                   # [175, C]
    weights = torch.softmax(disease_logits, dim=1) @ w        # [B, C]
    return torch.relu(torch.einsum("bc,bchw->bhw", weights, fmap))


class MultiTask(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, image):
        fmap, pooled = features(self.m, image)
        crop = self.m.heads["crop"](pooled)
        disease = self.m.heads["disease"](pooled)
        return crop, disease, disease_cam(fmap, disease, self.m.heads["disease"]), pooled


class Independent(nn.Module):
    def __init__(self, crop_model, disease_model):
        super().__init__()
        self.c, self.d = crop_model, disease_model

    def forward(self, image):
        _, pc = features(self.c, image)
        fmap, pd = features(self.d, image)
        crop = self.c.heads["crop"](pc)
        disease = self.d.heads["disease"](pd)
        emb = torch.cat([nn.functional.normalize(pc, dim=1), nn.functional.normalize(pd, dim=1)], dim=1)
        return crop, disease, disease_cam(fmap, disease, self.d.heads["disease"]), emb


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
            input_names=["image"], output_names=["crop_logits", "disease_logits", "heatmap", "embedding"],
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
