"""Evaluate baselines on DPD-Small test_seen and test_unseen.

  # multi-task baseline
  python baseline/evaluate.py --checkpoints outputs/multitask_effb0/best.pt --out outputs/multitask_effb0/eval
  # independent baseline: pass the crop model and the disease model together
  python baseline/evaluate.py --checkpoints outputs/indep_crop_effb0/best.pt outputs/indep_disease_effb0/best.pt \
                              --out outputs/indep_effb0/eval

Writes metrics.json, predictions.csv, confusion matrices (seen test) and the calibration curve.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PlantDiseaseDataset, audit, build_label_space, build_transforms, read_split
import plot_confusions
from metrics import calibration_sweep, composition_scores, harmonic_mean, topk_accuracy
from model import build_model
from train import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def load_models(paths, device):
    models, cfg, seen_heads = [], None, set()
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        clash = seen_heads & set(ckpt["heads"])
        if clash:
            raise ValueError(f"head(s) {clash} provided by more than one checkpoint")
        seen_heads |= set(ckpt["heads"])
        model = build_model(ckpt["config"], ckpt["heads"], pretrained=False)
        model.load_state_dict(ckpt["model"])
        models.append(model.to(device).eval())
        cfg = cfg or ckpt["config"]
    if seen_heads != {"crop", "disease"}:
        raise ValueError(f"need both crop and disease heads, got {seen_heads}")
    return models, cfg


@torch.no_grad()
def collect_logits(models, loader, device):
    out = {}
    for batch in tqdm(loader, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for m in models:
                for k, v in m(x).items():
                    out.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def gt_pair_indices(df, ls):
    idx = ls.pair_index()
    return np.array([idx.get(p, -1) for p in zip(df["crop_label"].tolist(), df["disease_label"].tolist())])


def save_confusions(scores, gt, ls, out_dir):
    """Seen test confusions: composition counts (.npy), top confused pairs (.csv) and plots."""
    pred = scores.argmax(1)
    n_pairs = len(ls.pairs)
    cm = confusion_matrix(gt, pred, labels=np.arange(n_pairs))[: ls.num_seen]  # rows: seen pairs
    np.save(os.path.join(out_dir, "confusion_seen_compositions.npy"), cm)

    off = cm.copy()
    np.fill_diagonal(off[:, : ls.num_seen], 0)
    rows, cols = np.nonzero(off)
    top = sorted(zip(rows, cols, off[rows, cols]), key=lambda t: -t[2])[:30]
    pd.DataFrame([{"true": ls.pair_name(r), "predicted": ls.pair_name(c), "count": int(n),
                   "rate_of_true_class": n / max(cm[r].sum(), 1)} for r, c, n in top]
                 ).to_csv(os.path.join(out_dir, "top_confusions_seen.csv"), index=False)

    plot_confusions.plot_all(cm, ls, out_dir)


def save_calibration_plot(cal, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(cal["seen_acc"], cal["unseen_acc"], lw=1.5)
    ax.scatter([cal["seen_at_best"]], [cal["unseen_at_best"]], color="red", zorder=3,
               label=f"best HM {cal['best_hm']:.3f}")
    ax.set_xlabel("seen top-1 accuracy")
    ax.set_ylabel("unseen top-1 accuracy")
    ax.set_title(f"Seen-unseen trade-off (AUC {cal['auc']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    models, cfg = load_models(args.checkpoints, device)
    dc = cfg["data"]

    train_df = read_split(os.path.join(dc["root"], dc["train_csv"]))
    seen_df = read_split(os.path.join(dc["root"], dc["test_seen_csv"]))
    unseen_df = read_split(os.path.join(dc["root"], dc["test_unseen_csv"]))
    ls = build_label_space(train_df, seen_df, unseen_df)
    audit(train_df, seen_df, unseen_df, ls)

    tf = build_transforms(dc["img_size"], train=False)
    image_dir = os.path.join(dc["root"], dc["image_dir"])
    results, scores_by_split, gt_by_split, pred_rows = {}, {}, {}, []
    for split, df in (("seen", seen_df), ("unseen", unseen_df)):
        gt = gt_pair_indices(df, ls)
        keep = gt >= 0
        df, gt = df[keep].reset_index(drop=True), gt[keep]
        loader = DataLoader(PlantDiseaseDataset(df, image_dir, tf), batch_size=args.batch_size,
                            num_workers=dc["num_workers"], shuffle=False)
        logits = collect_logits(models, loader, device)
        scores = composition_scores(logits["crop"], logits["disease"], ls.pairs)
        scores_by_split[split], gt_by_split[split] = scores, gt

        r = {
            "n_images": int(len(df)),
            "crop": topk_accuracy(logits["crop"], df["crop_label"].to_numpy()),
            "disease": topk_accuracy(logits["disease"], df["disease_label"].to_numpy()),
            # generalized: choose among ALL seen + unseen compositions (main protocol)
            "composition": topk_accuracy(scores, gt),
        }
        # closed-world: choose only among this split's own compositions (extra analysis)
        if split == "seen":
            r["composition_closed_world"] = topk_accuracy(scores[:, : ls.num_seen], gt)
        else:
            r["composition_closed_world"] = topk_accuracy(scores[:, ls.num_seen:], gt - ls.num_seen)
        results[split] = r

        top5 = np.argsort(-scores, axis=1)[:, :5]
        for i in range(len(df)):
            pred_rows.append({"split": split, "image": df["image"][i], "true": ls.pair_name(gt[i]),
                              **{f"pred_{k + 1}": ls.pair_name(top5[i, k]) for k in range(5)}})

    results["harmonic_mean"] = {
        k: harmonic_mean(results["seen"]["composition"][k], results["unseen"]["composition"][k])
        for k in ("top1", "top5")
    }
    cal = calibration_sweep(scores_by_split["seen"], gt_by_split["seen"],
                            scores_by_split["unseen"], gt_by_split["unseen"], ls.num_seen)
    results["calibrated"] = {k: cal[k] for k in ("best_hm", "best_bias", "seen_at_best", "unseen_at_best", "auc")}
    results["checkpoints"] = args.checkpoints

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    save_confusions(scores_by_split["seen"], gt_by_split["seen"], ls, args.out)
    save_calibration_plot(cal, args.out)

    print("\n                       Top-1    Top-5")
    for split in ("seen", "unseen"):
        for key in ("crop", "disease", "composition", "composition_closed_world"):
            r = results[split][key]
            print(f"{split:>6} {key:<25} {r['top1']:.4f}   {r['top5']:.4f}")
    hm = results["harmonic_mean"]
    print(f"harmonic mean (composition)      {hm['top1']:.4f}   {hm['top5']:.4f}")
    c = results["calibrated"]
    print(f"calibrated best HM {c['best_hm']:.4f} (seen {c['seen_at_best']:.4f}, unseen "
          f"{c['unseen_at_best']:.4f}, bias {c['best_bias']:.3f}) | AUC {c['auc']:.4f}")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
