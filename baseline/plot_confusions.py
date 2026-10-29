"""Readable confusion plots for the seen test set.

evaluate.py calls plot_all() automatically. To redraw from a finished evaluation
without running the model again:

  python baseline/plot_confusions.py outputs/multitask_effb0/eval

Writes into that folder:
  confusion_seen_crops.png        all 55 crops, % of each true crop, empty cells left blank
  confusion_seen_crops_worst.png  zoom on the crops the model gets wrong most often
  top_confusions_seen.png         the most frequent crop-disease mix-ups
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from dataset import build_label_space, read_split

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WRONG_DISEASE, WRONG_CROP = "#2a78d6", "#eb6834"
# One-hue sequential ramp; zero cells are masked and left as the background.
RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
RAMP.set_bad(SURFACE)

STYLE = {
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titleweight": "bold",
}


def crop_level(cm, ls):
    """Collapse the (seen pair x all pairs) count matrix to (crop x crop)."""
    pairs = np.asarray(ls.pairs)
    rows = np.eye(ls.num_crops, dtype=int)[pairs[: cm.shape[0], 0]]  # seen pair -> crop
    cols = np.eye(ls.num_crops, dtype=int)[pairs[: cm.shape[1], 0]]  # any pair -> crop
    return rows.T @ cm @ cols


def _heatmap(ax, counts, row_names, col_names, show_counts=False, font=6.5):
    support = counts.sum(1, keepdims=True)
    share = counts / np.clip(support, 1, None)
    im = ax.imshow(np.ma.masked_equal(share, 0), cmap=RAMP, vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    for r, c in zip(*np.nonzero(counts)):
        pct = share[r, c] * 100
        text = f"{pct:.0f}" if pct >= 1 else "<1"
        if show_counts:
            text = f"{text}%\n{counts[r, c]}"
        ax.text(c, r, text, ha="center", va="center", fontsize=font,
                color="white" if share[r, c] > 0.45 else INK)
    ax.set_xticks(range(len(col_names)), col_names, rotation=90)
    ax.set_yticks(range(len(row_names)),
                  [f"{n}  ({int(s)})" for n, s in zip(row_names, support[:, 0])])
    ax.set_xticks(np.arange(-0.5, len(col_names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_names)), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_crop_matrix(ccm, names, path):
    keep = np.nonzero(ccm.sum(1) + ccm.sum(0))[0]
    ccm, names = ccm[np.ix_(keep, keep)], [names[i] for i in keep]
    n = len(names)
    acc = np.trace(ccm) / max(ccm.sum(), 1)
    with plt.rc_context({**STYLE, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, ax = plt.subplots(figsize=(0.3 * n + 3, 0.3 * n + 1.5))
        im = _heatmap(ax, ccm, names, names)
        ax.set_xlabel("predicted crop")
        ax.set_ylabel("true crop  (number of test images)")
        ax.set_title(f"Crop confusion on the seen test set: crop accuracy {acc:.1%}\n"
                     "cells = % of each true crop's images; blank = 0", loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.set_label("share of the true crop's images")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_worst_crops(ccm, names, path, k=15, min_images=5):
    """Crops with at least min_images test images, so one mistake on a 2-image crop doesn't dominate."""
    support = ccm.sum(1)
    recall = np.where(support >= min_images, np.diag(ccm) / np.clip(support, 1, None), np.inf)
    worst = [i for i in np.argsort(recall, kind="stable") if recall[i] < 1][:k]
    if not worst:
        return
    # Columns: the same crops first (keeps the diagonal), then wherever their errors went.
    targets = sorted({c for r in worst for c in np.nonzero(ccm[r])[0]} - set(worst),
                     key=lambda c: names[c])
    cols = worst + targets
    sub = ccm[np.ix_(worst, cols)]
    with plt.rc_context({**STYLE, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3.5, 0.55 * len(worst) + 2.5))
        _heatmap(ax, sub, [names[i] for i in worst], [names[i] for i in cols],
                 show_counts=True, font=7)
        ax.axvline(len(worst) - 0.5, color=MUTED, linewidth=1)
        ax.set_xlabel("predicted crop  (right of the line: crops not in the rows)")
        ax.set_ylabel("true crop  (number of test images)")
        ax.set_title(f"The {len(worst)} crops with the lowest accuracy (seen test set, crops with "
                     f"{min_images}+ test images)\ncells = % of the row's images and image count", loc="left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_top_confusions(cm, ls, path, k=20):
    off = cm.copy()
    np.fill_diagonal(off[:, : cm.shape[0]], 0)
    rows, cols = np.nonzero(off)
    order = np.argsort(-off[rows, cols], kind="stable")[:k]
    rows, cols = rows[order], cols[order]
    if not len(rows):
        return
    pairs = np.asarray(ls.pairs)
    counts = off[rows, cols]
    support = cm[rows].sum(1)
    same_crop = pairs[rows, 0] == pairs[cols, 0]
    labels = [f"{ls.pair_name(r)}  →  {ls.disease_names.get(pairs[c, 1], pairs[c, 1])}" if s
              else f"{ls.pair_name(r)}  →  {ls.pair_name(c)}"
              for r, c, s in zip(rows, cols, same_crop)]

    with plt.rc_context({**STYLE, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(11, 0.36 * len(rows) + 1.8))
        y = np.arange(len(rows))
        ax.barh(y, counts, height=0.7, color=np.where(same_crop, WRONG_DISEASE, WRONG_CROP),
                edgecolor=SURFACE, linewidth=1)
        for yi, n, s in zip(y, counts, support):
            ax.text(n + counts.max() * 0.01, yi, f"{n} of {s}  ({n / s:.0%})",
                    va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, counts.max() * 1.18)
        ax.set_xlabel("misclassified test images")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"Top {len(rows)} mix-ups on the seen test set   (true crop | disease  →  prediction)",
                     loc="left")
        handles = [Patch(color=WRONG_DISEASE, label=f"right crop, wrong disease ({same_crop.sum()})"),
                   Patch(color=WRONG_CROP, label=f"wrong crop ({(~same_crop).sum()})")]
        ax.legend(handles=[h for h, n in zip(handles, (same_crop.sum(), (~same_crop).sum())) if n],
                  loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_all(cm, ls, out_dir):
    """cm: counts, rows = seen pairs, columns = all pairs (seen first, then unseen)."""
    ccm = crop_level(cm, ls)
    names = [ls.crop_names.get(i, str(i)) for i in range(ls.num_crops)]
    plot_crop_matrix(ccm, names, os.path.join(out_dir, "confusion_seen_crops.png"))
    plot_worst_crops(ccm, names, os.path.join(out_dir, "confusion_seen_crops_worst.png"))
    plot_top_confusions(cm, ls, os.path.join(out_dir, "top_confusions_seen.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir", help="folder holding confusion_seen_compositions.npy")
    p.add_argument("--config", default="configs/baseline.yaml", help="for the dataset paths")
    args = p.parse_args()

    with open(args.config) as f:
        dc = yaml.safe_load(f)["data"]
    ls = build_label_space(*(read_split(os.path.join(dc["root"], dc[k]))
                             for k in ("train_csv", "test_seen_csv", "test_unseen_csv")))
    cm = np.load(os.path.join(args.eval_dir, "confusion_seen_compositions.npy"))
    plot_all(cm, ls, args.eval_dir)
    print(f"saved to {args.eval_dir}")


if __name__ == "__main__":
    main()
