"""
export_s3_all_onnx.py - Exports the custom approach, Stage 3 run s3_all, to ONNX for the web page.

s3_all = R4 + R5 + R6 (stage3/stage3e). At inference it is an R5 network on frozen BioCLIP:

  v   = BioCLIP ViT-B/16 image feature (512)
  f_c = N(v + A_c(v)),  f_d = N(v + A_d(v))                      adapters
  crop logits    = s * f_c . N(T_c)                               T_c, T_d: the learnt prompts'
  disease logits = s * f_d . N(T_d)                               text-encoder output, stored in best.pt

The text encoder and R6's discriminator are not needed (the checkpoint's "model" entry already holds
T_c, T_d), so the graph is the BioCLIP vision tower plus two small adapters.

Input "image" [1, 3, 224, 224] (CLIP mean / std), outputs:
  crop_logits     [1, 55]
  disease_logits  [1, 175]
  embedding       [1, 1024]   f_c and f_d concatenated, for the similarity-ranked gallery

fp32, so the file is exactly best.pt (331 MB). It is over GitHub's 100 MB file limit, so it is
gitignored and hosted on Hugging Face only. int8 dynamic quantisation was tried and rejected: per-tensor
lost 1.9 points of calibrated HM (57.5% vs 59.4%), per-channel broke the model.

--check compares the ONNX file with PyTorch on a few images, then scores it on the full test sets with
the page's protocol next to outputs/stage3e/eval/metrics.json.

Usage (from the repo root; needs torch, open_clip_torch, onnx, onnxruntime, pandas, pillow):
    python UserInterface/tools/export_s3_all_onnx.py            # writes web/model/s3_all.onnx
    python UserInterface/tools/export_s3_all_onnx.py --check    # + ONNX-vs-PyTorch and full test-set accuracy
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from stage3.stage3b.dataset import CLIP_MEAN, CLIP_STD  # noqa: E402
from stage3.stage3b.evaluate import load_r5  # noqa: E402  (training code, so the network matches exactly)
from stage3.stage3b.model import load_bioclip  # noqa: E402

CHECKPOINT = REPO / "outputs" / "stage3e" / "best.pt"
METRICS = REPO / "outputs" / "stage3e" / "eval" / "metrics.json"
OUT_DIR = REPO / "UserInterface" / "web" / "model"
OUT = OUT_DIR / "s3_all.onnx"
SIZE = 224


class S3All(nn.Module):
    def __init__(self, visual, net):
        super().__init__()
        self.visual, self.net = visual, net

    def forward(self, image):
        v = self.visual(image)                                    # BioCLIP encode_image (unnormalised)
        f = self.net.features(v)                                  # N(v + A(v)) for crop and disease
        s = self.net.logit_scale
        crop = s * f["crop"] @ F.normalize(self.net.Tc, dim=-1).T
        disease = s * f["disease"] @ F.normalize(self.net.Td, dim=-1).T
        return crop, disease, torch.cat([f["crop"], f["disease"]], dim=1)


def build():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    clip, _ = load_bioclip(ckpt["config"]["model"]["backbone"], torch.device("cpu"))
    net = load_r5(ckpt["model"], ckpt["config"]["model"])
    return S3All(clip.visual, net).eval()


def export(wrapper):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # nn.MultiheadAttention's fused eval fast path (aten::_native_multi_head_attention) has no ONNX
    # export; the regular path is the same computation
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(wrapper, torch.randn(1, 3, SIZE, SIZE), str(OUT), dynamo=False, opset_version=17,
                          input_names=["image"], output_names=["crop_logits", "disease_logits", "embedding"])
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 2**20:.0f} MB)")


def preprocess(path):
    """The evaluation transform (stage3/stage3b/dataset.py build_transforms, train=False)."""
    from PIL import Image
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(int(SIZE / 0.875)), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    return tf(Image.open(path).convert("RGB"))[None].numpy()


def check(wrapper, n_agree=40):
    """ONNX vs PyTorch on a few images, then the ONNX file's full test-set accuracy."""
    import numpy as np
    import onnxruntime as ort
    import pandas as pd

    images = REPO / "dataset" / "DPD_Small_dataset"
    seen_df = pd.read_csv(REPO / "dataset" / "DPD_Small_test_seen.csv", header=None)
    unseen_df = pd.read_csv(REPO / "dataset" / "DPD_Small_test_unseen.csv", header=None)

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    diff = 0.0
    for f in pd.concat([seen_df.iloc[:n_agree // 2, 0], unseen_df.iloc[:n_agree // 2, 0]]):
        x = preprocess(images / f)
        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(torch.from_numpy(x))]
        got = sess.run(None, {"image": x})
        diff = max(diff, *(float(np.abs(a - b).max()) for a, b in zip(ref, got)))
    print(f"onnx vs torch on {n_agree} images: max |diff| = {diff:.2e}")

    test_accuracy(OUT)


def test_accuracy(onnx_path):
    """Full test sets, scored as the page does (log p(crop) + log p(disease) over the 333 pairs,
    + the calibrated bias on unseen pairs), next to outputs/stage3e/eval/metrics.json."""
    import numpy as np
    import onnxruntime as ort
    import pandas as pd

    images = REPO / "dataset" / "DPD_Small_dataset"
    seen_df = pd.read_csv(REPO / "dataset" / "DPD_Small_test_seen.csv", header=None)
    unseen_df = pd.read_csv(REPO / "dataset" / "DPD_Small_test_unseen.csv", header=None)

    classes = json.loads((REPO / "UserInterface" / "web" / "data" / "classes.json").read_text(encoding="utf-8"))
    comps = classes["compositions"]
    pc = np.array([c["crop"] for c in comps])
    pd_ = np.array([c["disease"] for c in comps])
    unseen_mask = np.array([not c["seen"] for c in comps])
    index = {(c["crop"], c["disease"]): i for i, c in enumerate(comps)}
    bias = json.loads(METRICS.read_text())["ausuc"]["best_bias"]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def log_softmax(z):
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def top1(df, name):
        hit0 = hitb = 0
        for i, (f, c, d) in enumerate(zip(df[0], df[1], df[2]), 1):
            crop, dis, _ = sess.run(None, {"image": preprocess(images / f)})
            s = log_softmax(crop[0])[pc] + log_softmax(dis[0])[pd_]
            t = index[(int(c), int(d))]
            hit0 += int(s.argmax() == t)
            hitb += int((s + bias * unseen_mask).argmax() == t)
            if i % 200 == 0 or i == len(df):
                print(f"\r{name}: {i}/{len(df)}", end="", flush=True)
        print()
        return hit0 / len(df), hitb / len(df)

    s0, sb = top1(seen_df, "test_seen")
    u0, ub = top1(unseen_df, "test_unseen")
    m = json.loads(METRICS.read_text())
    hm = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0  # noqa: E731
    print(f"{Path(onnx_path).name:22}{'onnx':>10}{'metrics.json':>14}")
    print(f"{'seen top-1 (bias 0)':22}{s0:>10.4f}{m['test_seen']['top1']:>14.4f}")
    print(f"{'unseen top-1 (bias 0)':22}{u0:>10.4f}{m['test_unseen']['top1']:>14.4f}")
    print(f"{'seen top-1 @ bias':22}{sb:>10.4f}{m['ausuc']['seen_at_best']:>14.4f}")
    print(f"{'unseen top-1 @ bias':22}{ub:>10.4f}{m['ausuc']['unseen_at_best']:>14.4f}")
    print(f"{'HM @ bias':22}{hm(sb, ub):>10.4f}{m['ausuc']['best_hm']:>14.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="ONNX vs PyTorch, and full test-set accuracy")
    args = ap.parse_args()
    wrapper = build()
    export(wrapper)
    if args.check:
        check(wrapper)


if __name__ == "__main__":
    main()
