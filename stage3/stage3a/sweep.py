"""R4 lower token-LR sweep (guide.md "Rule", training_scheme.md §6.1, results.md next step 6). Repo root:

  python -m stage3.stage3a.sweep                          # all six runs, evaluate each, then a summary
  python -m stage3.stage3a.sweep --runs lr1e-4 lr1e-4_nokg
  python -m stage3.stage3a.sweep --epochs 1               # smoke test (experiments get an _ep1 suffix)
  python -m stage3.stage3a.sweep --summary_only           # rebuild the summary from existing metrics.json files

Each run is the unchanged stage3/stage3a/train.py with only the token LR (and, for the _nokg rows, lambda_kg 0)
overridden, launched as its own process, then scored with the unchanged stage3/stage3a/evaluate.py.
The main row (lr 1e-3) is not changed: this is a named sweep sub-row and does not replace it.

LRs, all below the 1e-3 default:
  5e-4   PROPOSED_METHOD.md §5 value; CSP grid {5e-3, 5e-4, 5e-5}
  1e-4   the Colab R4 runs; the example named in guide.md and results.md
  5e-5   CSP's official code
Each at lambda_kg 8 (main R4 setting) and 0 (tests whether the early drift jump at 1e-3 is what costs seen acc).

The 1e-3 reference rows (outputs/stage3a, outputs/stage3a_nokg) are not retrained; they are included in the
summary if their metrics.json exists. A run whose eval/metrics.json already exists is skipped unless --force.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

LRS = ["5e-4", "1e-4", "5e-5"]
RUNS = {}
for lr in LRS:
    RUNS[f"lr{lr}"] = {"experiment": f"stage3a_lr{lr}", "args": ["--lr_new", lr]}
    RUNS[f"lr{lr}_nokg"] = {"experiment": f"stage3a_lr{lr}_nokg", "args": ["--lr_new", lr, "--lambda_kg", "0"]}

SUMMARY_ROWS = [("lr 1e-3, kg 8 (main)", "stage3a")] + \
               [(f"lr {lr}, kg 8", f"stage3a_lr{lr}") for lr in LRS] + \
               [("lr 1e-3, kg 0", "stage3a_nokg")] + \
               [(f"lr {lr}, kg 0", f"stage3a_lr{lr}_nokg") for lr in LRS]


def run_step(cmd: list[str]) -> None:
    print(f"\n[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable, *cmd], cwd=REPO, check=True)


def metrics_path(output_dir: str, experiment: str) -> str:
    return os.path.join(REPO, output_dir, experiment, "eval", "metrics.json")


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def best_drift(output_dir: str, experiment: str, epoch: int) -> str:
    """Drift ‖T − T⁰‖² (crop / disease) at the selected epoch, from history.csv."""
    p = os.path.join(REPO, output_dir, experiment, "history.csv")
    if not os.path.exists(p):
        return "-"
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["epoch"])) == epoch and row.get("drift_crop"):
                return f"{float(row['drift_crop']):.2f} / {float(row['drift_disease']):.2f}"
    return "-"


def summarize(output_dir: str) -> str:
    cols = ["Run", "Best ep", "Val score", "Drift c / d", "Seen comp", "Unseen comp", "Unseen dis",
            "HM top-1", "HM top-5", "Cal. HM", "AUSUC"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, exp in SUMMARY_ROWS:
        p = metrics_path(output_dir, exp)
        if not os.path.exists(p):
            lines.append(f"| {name} | " + " | ".join(["not run"] + [""] * (len(cols) - 2)) + " |")
            continue
        with open(p) as f:
            m = json.load(f)
        row = [name, str(m["epoch"]), pct(m["val"]["val_score"]), best_drift(output_dir, exp, m["epoch"]),
               pct(m["test_seen"]["top1"]), pct(m["test_unseen"]["top1"]), pct(m["test_unseen"]["disease_acc"]),
               pct(m["test_hm_top1"]), pct(m["test_hm_top5"]), pct(m["ausuc"]["best_hm"]), pct(m["ausuc"]["auc"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    p.add_argument("--epochs", type=int, help="override epochs (e.g. 1 for a smoke test)")
    p.add_argument("--output_dir", default="outputs", help="must match output_dir in config.yaml")
    p.add_argument("--force", action="store_true", help="retrain runs that already have metrics.json")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    if not args.summary_only:
        for key in args.runs:
            r = RUNS[key]
            exp = r["experiment"] + (f"_ep{args.epochs}" if args.epochs else "")
            if os.path.exists(metrics_path(args.output_dir, exp)) and not args.force:
                print(f"[sweep] {key}: {exp} already evaluated, skipping (use --force to rerun)")
                continue
            train_cmd = ["-m", "stage3.stage3a.train", "--experiment", exp, *r["args"]]
            if args.epochs:
                train_cmd += ["--epochs", str(args.epochs)]
            run_step(train_cmd)
            run_step(["-m", "stage3.stage3a.evaluate",
                      "--checkpoint", os.path.join(args.output_dir, exp, "best.pt")])

    table = summarize(args.output_dir)
    print("\n" + table)
    out = os.path.join(REPO, args.output_dir, "stage3a_lr_sweep_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# R4 token-LR sweep (seed 42)\n\n"
                "Sweep sub-row (guide.md): the main R4 row stays at lr 1e-3. Test metrics use the generalized "
                "valid-pair protocol (333 pairs). Cal. HM / AUSUC from the calibration sweep. Drift = ‖T − T⁰‖² "
                "at the selected epoch. Selection rule: highest val score.\n\n" + table + "\n")
    print(f"\n[sweep] summary -> {out}")


if __name__ == "__main__":
    main()
