import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
ROOT = Path("./weights")
BEST_PT = ROOT / "MultitaskDinoV2/best.pt"
SEED2_PT = ROOT / "R4_CSP_KgCOOP/heads_seed2.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import timm
except ImportError as e:
    raise ImportError("pip install timm") from e

# ------------------------------------------------------------
# Model 1: Multi-task DINOv2 (best.pt)
# ------------------------------------------------------------
class MultiTaskDINOv2(nn.Module):
    def __init__(self, num_crops: int = 55, num_diseases: int = 175, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE,
        )
        feat_dim = self.backbone.num_features  # 384
        self.heads = nn.ModuleDict({
            "crop": nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feat_dim, num_crops)),
            "disease": nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feat_dim, num_diseases)),
        })

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls = feats[:, 0]
        return self.heads["crop"](cls), self.heads["disease"](cls)


def load_best_pt(path: Path = BEST_PT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    num_crops = int(ckpt.get("num_crops", 55))
    num_diseases = int(ckpt.get("num_diseases", 175))
    dropout = ckpt.get("config", {}).get("model", {}).get("dropout", 0.3)
    model = MultiTaskDINOv2(num_crops, num_diseases, dropout)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(DEVICE).eval()
    return model, num_crops, num_diseases


# ------------------------------------------------------------
# Model 2: Prototype heads (heads_seed2.pt)
# ------------------------------------------------------------
class PrototypeHeads(nn.Module):
    def __init__(self, crop_theta, crop_ctx, dis_theta, dis_ctx, scale: float = 100.0):
        super().__init__()
        self.register_buffer("crop_proto", F.normalize(crop_theta.mean(dim=1), dim=-1))
        self.register_buffer("dis_proto", F.normalize(dis_theta.mean(dim=1), dim=-1))
        self.scale = float(scale)
        proj = torch.empty(384, 512)
        torch.manual_seed(42)
        nn.init.orthogonal_(proj)
        self.register_buffer("proj", proj)

    def forward(self, cls_384: torch.Tensor):
        feat = F.normalize(cls_384 @ self.proj, dim=-1)
        crop_logits = self.scale * (feat @ self.crop_proto.T)
        dis_logits = self.scale * (feat @ self.dis_proto.T)
        return crop_logits, dis_logits


def load_seed2(path: Path = SEED2_PT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    heads = PrototypeHeads(
        ckpt["crop"]["theta"],
        ckpt["crop"]["ctx"],
        ckpt["dis"]["theta"],
        ckpt["dis"]["ctx"],
        scale=ckpt.get("scale", 100.0),
    )
    heads.to(DEVICE).eval()
    num_crops = ckpt["crop"]["theta"].shape[0]
    num_diseases = ckpt["dis"]["theta"].shape[0]
    return heads, num_crops, num_diseases


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
print("Loading models …")
MODEL_BEST, NC_BEST, ND_BEST = load_best_pt()
BACKBONE = MODEL_BEST.backbone
HEADS_SEED2, NC_SEED2, ND_SEED2 = load_seed2()


def _labels(n: int, prefix: str, path: Path):
    if path.exists():
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if len(names) == n:
            return names
    return [f"{prefix}_{i}" for i in range(n)]


CROP_NAMES = _labels(NC_BEST, "Crop", ROOT / "crop_labels.txt")
DISEASE_NAMES = _labels(ND_BEST, "Disease", ROOT / "disease_labels.txt")

MODEL_OPTIONS = {
    "Multi-task DINOv2 (best.pt)": "best",
    "Prototype Heads (heads_seed2.pt)": "seed2",
}

print(f"Ready | device={DEVICE} | crops={NC_BEST} | diseases={ND_BEST}")

# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def top_k_list(logits: torch.Tensor, names: list, k: int = 5):
    """Return list-of-lists for Gradio Dataframe + dict for charts."""
    probs = F.softmax(logits, dim=-1)[0]
    vals, idxs = torch.topk(probs, k=min(k, len(names)))
    rows = []
    for r, (v, i) in enumerate(zip(vals, idxs), 1):
        rows.append([r, names[int(i)], round(float(v), 4), round(float(v) * 100, 2)])
    return rows


def top_k_dict(logits: torch.Tensor, names: list, k: int = 5) -> dict:
    rows = top_k_list(logits, names, k)
    return {r[1]: r[2] for r in rows}


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def prediction_entropy(probs: torch.Tensor) -> float:
    p = probs.clamp(min=1e-8)
    return float(-(p * p.log()).sum())


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
def make_topk_bar(preds: dict, title: str, color: str) -> Figure:
    if not preds:
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = list(preds.keys())[::-1]
    confs = [preds[l] * 100 for l in labels]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    bars = ax.barh(labels, confs, color=color, edgecolor="white", height=0.65)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, c in zip(bars, confs):
        ax.text(min(c + 1.5, 98), bar.get_y() + bar.get_height() / 2,
                f"{c:.1f}%", va="center", fontsize=10, fontweight="medium")

    fig.tight_layout(pad=1.5)
    return fig


def make_calibration_panel(
    crop_probs: torch.Tensor,
    dis_probs: torch.Tensor,
    crop_conf: float,
    dis_conf: float,
    hmean: float,
) -> Figure:
    crop_ent = prediction_entropy(crop_probs)
    dis_ent = prediction_entropy(dis_probs)
    max_ent_crop = np.log(len(crop_probs))
    max_ent_dis = np.log(len(dis_probs))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    names = ["Crop\nTop-1", "Disease\nTop-1", "Harmonic\nMean"]
    vals = [crop_conf * 100, dis_conf * 100, hmean * 100]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.6)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Task Confidence", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[1]
    norm_ent = [crop_ent / max_ent_crop * 100, dis_ent / max_ent_dis * 100]
    bars = ax.bar(["Crop", "Disease"], norm_ent, color=["#27ae60", "#c0392b"],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Normalized Entropy (%)")
    ax.set_title("Uncertainty (lower better)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, norm_ent):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[2]
    def top5_mass(p):
        return float(torch.topk(p, 5).values.sum()) * 100
    masses = [top5_mass(crop_probs), top5_mass(dis_probs)]
    bars = ax.bar(["Crop", "Disease"], masses, color=["#1abc9c", "#e67e22"],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Probability Mass (%)")
    ax.set_title("Top-5 Concentration", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, masses):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Calibration & Reliability Indicators", fontsize=12, y=1.05)
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
    return fig


def make_auc_style_curve(crop_probs: torch.Tensor, dis_probs: torch.Tensor) -> Figure:
    """
    Single-image ROC-style / ranking curve.
    True AUC needs labeled validation data. Here we plot:
    - Sorted confidence (ranking curve)
    - Cumulative probability mass
    - Approximate one-vs-rest style TPR/FPR using softmax as soft labels
      (illustrative only — not a true ROC/AUC)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

    def ranking_and_soft_roc(probs: torch.Tensor, ax, title: str, color: str):
        p = probs.detach().cpu().numpy()
        order = np.argsort(-p)
        sorted_p = p[order]
        n = len(sorted_p)

        # Cumulative mass (how concentrated is the prediction)
        cum = np.cumsum(sorted_p)

        ax.plot(np.arange(1, n + 1), sorted_p, color=color, lw=2, label="Sorted confidence")
        ax.fill_between(np.arange(1, n + 1), sorted_p, alpha=0.15, color=color)
        ax.plot(np.arange(1, n + 1), cum, color="#34495e", lw=1.5, linestyle="--",
                label="Cumulative mass")

        # Soft AUC-style: treat top class as positive, use confidence ranking
        # Approximate: TPR ≈ cum mass of top-k / total, FPR ≈ k/n
        k_range = np.arange(1, min(n, 40) + 1)
        tpr = cum[k_range - 1]
        fpr = k_range / n
        # Trapezoidal "AUC" of this soft curve (not real ROC-AUC)
        # manual trapezoidal rule (works on all NumPy versions)
        soft_auc = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2.0))

        ax2 = ax.twinx()
        ax2.plot(k_range, tpr, color="#8e44ad", lw=1.8, alpha=0.85, label=f"Soft TPR (AUC≈{soft_auc:.3f})")
        ax2.set_ylabel("Cumulative / Soft TPR", color="#8e44ad")
        ax2.tick_params(axis="y", labelcolor="#8e44ad")
        ax2.set_ylim(0, 1.05)

        ax.set_xlabel("Rank (1 = highest confidence)")
        ax.set_ylabel("Confidence")
        ax.set_title(f"{title}\nsoft ranking AUC ≈ {soft_auc:.3f}", fontweight="bold", fontsize=11)
        ax.set_xlim(1, min(n, 40))
        ax.set_ylim(0, max(sorted_p.max() * 1.15, 0.05))
        ax.spines["top"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        return soft_auc

    auc_c = ranking_and_soft_roc(crop_probs, axes[0], "Crop ranking curve", "#2ecc71")
    auc_d = ranking_and_soft_roc(dis_probs, axes[1], "Disease ranking curve", "#e74c3c")

    fig.suptitle(
        "Ranking / Soft-AUC curves (illustrative — true ROC-AUC needs labeled validation set)",
        fontsize=11, y=1.05,
    )
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.93])
    return fig, auc_c, auc_d


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
@torch.inference_mode()
def predict(image: Image.Image, model_name: str):
    empty_fig = make_topk_bar({}, "—", "#999")
    empty_table = [["—", "—", 0.0, 0.0]]
    if image is None:
        return (
            empty_fig, empty_fig, empty_fig, empty_fig,
            empty_table, empty_table,
            "Upload an image first.",
            {},
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    key = MODEL_OPTIONS.get(model_name, "best")

    if key == "best":
        crop_logits, dis_logits = MODEL_BEST(x)
    else:
        feats = BACKBONE.forward_features(x)
        cls = feats[:, 0]
        crop_logits, dis_logits = HEADS_SEED2(cls)

    crop_probs = F.softmax(crop_logits, dim=-1)[0]
    dis_probs = F.softmax(dis_logits, dim=-1)[0]

    crop_rows = top_k_list(crop_logits, CROP_NAMES, k=5)
    dis_rows = top_k_list(dis_logits, DISEASE_NAMES, k=5)
    crop_top = {r[1]: r[2] for r in crop_rows}
    dis_top = {r[1]: r[2] for r in dis_rows}

    crop_conf = crop_rows[0][2]
    dis_conf = dis_rows[0][2]
    hmean = harmonic_mean(crop_conf, dis_conf)

    crop_fig = make_topk_bar(crop_top, "Top-5 Crop Predictions", "#2ecc71")
    dis_fig = make_topk_bar(dis_top, "Top-5 Disease Predictions", "#e74c3c")
    cal_fig = make_calibration_panel(crop_probs, dis_probs, crop_conf, dis_conf, hmean)
    auc_fig, soft_auc_c, soft_auc_d = make_auc_style_curve(crop_probs, dis_probs)

    summary = (
        f"**Model:** {model_name}\n\n"
        f"**Top Crop:** `{crop_rows[0][1]}` — **{crop_rows[0][3]:.1f}%**\n"
        f"**Top Disease:** `{dis_rows[0][1]}` — **{dis_rows[0][3]:.1f}%**\n\n"
        f"**Harmonic mean confidence:** **{hmean * 100:.1f}%**\n"
        f"**Soft ranking AUC** — Crop: `{soft_auc_c:.3f}` · Disease: `{soft_auc_d:.3f}`\n\n"
        f"*Note: Soft-AUC is an illustrative ranking metric from one image. "
        f"True ROC-AUC requires a labeled validation set.*"
    )

    raw = {
        "crop": crop_top,
        "disease": dis_top,
        "harmonic_mean": round(hmean, 4),
        "soft_auc_crop": round(soft_auc_c, 4),
        "soft_auc_disease": round(soft_auc_d, 4),
    }

    return (
        crop_fig, dis_fig, cal_fig, auc_fig,
        crop_rows, dis_rows,
        summary,
        raw,
    )


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks(
    title="Multi-task Crop & Disease Classifier",
    theme=gr.themes.Soft(primary_hue="green"),
    css="""
    .gradio-container { max-width: 1400px !important; }
    #input-img img { max-height: 600px !important; width: 100% !important; object-fit: contain !important; }
    .plot-container, .js-plotly-plot, .matplotlib { width: 100% !important; }
    footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # Multi-task Crop & Disease Classifier
        Select a model → upload a leaf image → **Predict**.  
        Shows **Top-5 charts**, **score tables**, **harmonic mean**, **calibration panel**, and an **illustrative ranking / soft-AUC curve**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=420):
            model_dd = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=list(MODEL_OPTIONS.keys())[0],
                label="Model / Weights",
            )
            image_in = gr.Image(
                type="pil",
                label="Upload Leaf Image",
                height=560,
                elem_id="input-img",
            )
            btn = gr.Button("Predict", variant="primary", size="lg")

        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Summary")

    # Top-5 charts stacked vertically (full width, easier to read)
    crop_plot = gr.Plot(label="Crop Top-5 Predictions")
    dis_plot = gr.Plot(label="Disease Top-5 Predictions")

    with gr.Row():
        crop_table = gr.Dataframe(
            headers=["Rank", "Class", "Score", "Confidence %"],
            datatype=["number", "str", "number", "number"],
            label="Crop score table (Top-5)",
            interactive=False,
            wrap=True,
            row_count=5,
        )
        dis_table = gr.Dataframe(
            headers=["Rank", "Class", "Score", "Confidence %"],
            datatype=["number", "str", "number", "number"],
            label="Disease score table (Top-5)",
            interactive=False,
            wrap=True,
            row_count=5,
        )

    cal_plot = gr.Plot(label="Calibration & Reliability")
    auc_plot = gr.Plot(label="Ranking / Soft-AUC curves")
    raw_out = gr.JSON(visible=False)

    btn.click(
        fn=predict,
        inputs=[image_in, model_dd],
        outputs=[
            crop_plot, dis_plot, cal_plot, auc_plot,
            crop_table, dis_table,
            summary_out, raw_out,
        ],
    )

    gr.Markdown(
        f"""
        ---
        **Models**
        | Name | File | Notes |
        |------|------|-------|
        | Multi-task DINOv2 | `best.pt` | Full backbone + linear heads (384-d) |
        | Prototype Heads | `heads_seed2.pt` | Class prototypes (512-d) + projected DINOv2 features |

        **Outputs**
        - **Score tables** — Rank, class name, raw score, confidence %
        - **Harmonic mean** of the two top-1 confidences
        - **Calibration panel** — confidence, entropy, top-5 concentration
        - **Ranking / soft-AUC curves** — illustrative only; real ROC-AUC needs labeled validation data

        Place `crop_labels.txt` / `disease_labels.txt` next to the weights for real class names.
        """
    )


if __name__ == "__main__":
    demo.launch()
