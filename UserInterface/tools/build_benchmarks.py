"""
build_benchmarks.py - Builds web/data/benchmarks.json for the "Model benchmarks" dialog.

Reads each run's evaluation output in outputs/<run>/eval/ and writes one JSON
file with:
  * runs       test-set accuracy of every run in RUNS (seen, unseen, harmonic
               mean, top-1 / top-5, calibrated best HM)
  * confusion  per run, where saved: a crop-level confusion matrix on test_seen
               (from predictions.csv) and the most confused crop-disease pairs
               (from top_confusions_seen.csv)

Two metrics.json layouts exist: the EffB0 evaluator ("seen" / "unseen" keys)
and the Stage 3 evaluator ("test_seen" / "test_unseen" keys). Both are read.

Usage (from the repo root):
    python UserInterface/tools/build_benchmarks.py
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
OUT_FILE = REPO / "UserInterface" / "web" / "data" / "benchmarks.json"

# Runs shown in the dialog, in display order:
# (folder in outputs/, name, role, short label for the charts).
# Edit this list to change which runs appear or how they are labelled.
RUNS = [
    # The two deployed baselines (Stage 1 re-runs, the checkpoints exported to ONNX)
    ("stage1b", "Independent CNNs (EfficientNet-B0)", "Baseline 1", "B1 · Independent CNNs"),
    ("stage1a", "Multi-task CNN (EfficientNet-B0)", "Baseline 2", "B2 · Multi-task CNN"),
    # The deployed custom approach (outputs/stage3e/best.pt), then its Stage 3 component runs
    ("stage3e", "S3 All: BioCLIP + learnt prompts + adapters + crop-invariance", "Custom approach", "Custom · S3 All"),
    ("stage3b", "S3 R5: BioCLIP + crop / disease adapters", "Ablation", "Ablation · S3 R5"),
    ("stage3c", "S3 R5 + R6: adapters + crop discriminator", "Ablation", "Ablation · S3 R5 + R6"),
]
TOP_CONFUSIONS = 10


def pretty(name):
    """'pepper_bell' -> 'Pepper bell'"""
    s = name.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def pretty_pair(pair):
    """'paddy | blast' -> 'Paddy · Blast'"""
    crop, disease = [p.strip() for p in pair.split("|")]
    return f"{pretty(crop)} · {pretty(disease)}"


def r(x):
    return None if x is None or x != x else round(float(x), 4)  # x != x catches NaN


def read_metrics(d):
    """Normalises both metrics.json layouts into one dict of fractions (0-1)."""
    if "seen" in d:  # EffB0 evaluator
        s, u, cal = d["seen"], d["unseen"], d.get("calibrated", {})
        return {
            "seen": {"crop": r(s["crop"]["top1"]), "disease": r(s["disease"]["top1"]),
                     "top1": r(s["composition"]["top1"]), "top5": r(s["composition"]["top5"])},
            "unseen": {"crop": r(u["crop"]["top1"]), "disease": r(u["disease"]["top1"]),
                       "top1": r(u["composition"]["top1"]), "top5": r(u["composition"]["top5"])},
            "hm": {"top1": r(d["harmonic_mean"]["top1"]), "top5": r(d["harmonic_mean"]["top5"])},
            "calibrated": {"best_hm": r(cal.get("best_hm")), "bias": r(cal.get("best_bias")),
                           "seen": r(cal.get("seen_at_best")), "unseen": r(cal.get("unseen_at_best")),
                           "auc": r(cal.get("auc"))},
            "n": {"seen": s["n_images"], "unseen": u["n_images"]},
        }
    s, u, cal = d["test_seen"], d["test_unseen"], d.get("ausuc", {})  # Stage 3 evaluator
    return {
        "seen": {"crop": r(s["crop_acc"]), "disease": r(s["disease_acc"]), "top1": r(s["top1"]), "top5": r(s["top5"])},
        "unseen": {"crop": r(u["crop_acc"]), "disease": r(u["disease_acc"]), "top1": r(u["top1"]), "top5": r(u["top5"])},
        "hm": {"top1": r(d["test_hm_top1"]), "top5": r(d["test_hm_top5"])},
        "calibrated": {"best_hm": r(cal.get("best_hm")), "bias": r(cal.get("best_bias")),
                       "seen": r(cal.get("seen_at_best")), "unseen": r(cal.get("unseen_at_best")),
                       "auc": r(cal.get("auc"))},
        "n": {"seen": s["n_images"], "unseen": u["n_images"]},
    }


def crop_confusion(pred_csv):
    """Crop-level confusion counts on test_seen, from the top-1 composition prediction."""
    pairs = []
    with open(pred_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "seen":
                pairs.append((pretty(row["true"].split("|")[0]), pretty(row["pred_1"].split("|")[0])))
    crops = sorted({c for p in pairs for c in p})
    idx = {c: i for i, c in enumerate(crops)}
    m = [[0] * len(crops) for _ in crops]
    for t, p in pairs:
        m[idx[t]][idx[p]] += 1
    return {"crops": crops, "matrix": m, "n_images": len(pairs)}


def top_confusions(conf_csv):
    with open(conf_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))[:TOP_CONFUSIONS]
    return [{"true": pretty_pair(x["true"]), "pred": pretty_pair(x["predicted"]),
             "count": int(x["count"]), "rate": r(x["rate_of_true_class"])} for x in rows]


def main():
    runs, confusion = [], {}
    for folder, name, role, short in RUNS:
        ev = OUTPUTS / folder / "eval"
        metrics = ev / "metrics.json"
        if not metrics.exists():
            print(f"skip {folder}: no eval/metrics.json")
            continue
        with open(metrics, encoding="utf-8") as f:
            # The Stage 3 evaluator writes NaN, which json.load accepts
            runs.append({"id": folder, "name": name, "role": role, "short": short, **read_metrics(json.load(f))})
        conf = {}
        if (ev / "predictions.csv").exists():
            conf["crop_matrix"] = crop_confusion(ev / "predictions.csv")
        if (ev / "top_confusions_seen.csv").exists():
            conf["top"] = top_confusions(ev / "top_confusions_seen.csv")
        if conf:
            confusion[folder] = conf
        print(f"ok   {folder}" + (f" (+ {', '.join(conf)})" if conf else ""))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "confusion": confusion}, f, separators=(",", ":"))
    print(f"wrote {OUT_FILE.relative_to(REPO)} ({OUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
