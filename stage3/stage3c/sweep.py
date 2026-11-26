"""R5 + R6 lambda sweep and unconditional-DANN ablation (PROPOSED_METHOD.md §5, §7). Run from the repo root:

  python -m stage3.stage3c.sweep                        # all four runs, then evaluate each, then a summary
  python -m stage3.stage3c.sweep --runs uncond          # just one run
  python -m stage3.stage3c.sweep --runs lam0 lam0.3     # any subset
  python -m stage3.stage3c.sweep --summary_only         # rebuild the summary from existing metrics.json files

Each run is the unchanged stage3/stage3c/train.py (same data, split, schedule, selection) with one override,
launched as its own process, then scored with the unchanged stage3/stage3c/evaluate.py:

  lam0     conditional,   lambda_max 0      sanity row: D trains, no gradient reaches A_d -> should match R5
  lam0.03  conditional,   lambda_max 0.03
  lam0.3   conditional,   lambda_max 0.3
  uncond   UNCONDITIONAL, lambda_max 0.1    plain DANN: all images, softmax over all 55 crops

The main conditional lambda_max 0.1 run is the original `python -m stage3.stage3c.train` (outputs/stage3c/);
it is not retrained here but is included in the summary if its metrics.json exists.
A run whose eval/metrics.json already exists is skipped unless --force is given.
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

RUNS = {
    "lam0":    {"experiment": "stage3c_lam0",    "args": ["--lambda_max", "0"]},
    "lam0.03": {"experiment": "stage3c_lam0.03", "args": ["--lambda_max", "0.03"]},
    "lam0.3":  {"experiment": "stage3c_lam0.3",  "args": ["--lambda_max", "0.3"]},
    "uncond":  {"experiment": "stage3c_uncond",  "args": ["--lambda_max", "0.1", "--unconditional"]},
}
# Rows shown in the summary, in order (existing reference runs + the sweep).
SUMMARY_ROWS = [
    ("R5 (no R6)", "stage3b"),
    ("R6 cond, lam 0", "stage3c_lam0"),
    ("R6 cond, lam 0.03", "stage3c_lam0.03"),
    ("R6 cond, lam 0.1 (main)", "stage3c"),
    ("R6 cond, lam 0.3", "stage3c_lam0.3"),
    ("R6 uncond DANN, lam 0.1", "stage3c_uncond"),
]


def run_step(cmd: list[str]) -> None:
    print(f"\n[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable, *cmd], cwd=REPO, check=True)


def metrics_path(output_dir: str, experiment: str) -> str:
    return os.path.join(REPO, output_dir, experiment, "eval", "metrics.json")


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def summarize(output_dir: str) -> str:
    cols = ["Run", "Best ep", "Val score", "Seen comp", "Unseen comp", "Unseen dis",
            "HM top-1", "HM top-5", "Cal. HM", "AUSUC", "f_d leak"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, exp in SUMMARY_ROWS:
        p = metrics_path(output_dir, exp)
        if not os.path.exists(p):
            lines.append(f"| {name} | " + " | ".join(["not run"] + [""] * (len(cols) - 2)) + " |")
            continue
        with open(p) as f:
            m = json.load(f)
        leak = next((v for k, v in m.get("leakage", {}).items() if k.endswith("f_d")), None)
        leak_s = f"{leak['leak']:+.2f} ± {leak['leak_std']:.2f}" if leak else "—"
        row = [name, str(m["epoch"]), pct(m["val"]["val_score"]), pct(m["test_seen"]["top1"]),
               pct(m["test_unseen"]["top1"]), pct(m["test_unseen"]["disease_acc"]),
               pct(m["test_hm_top1"]), pct(m["test_hm_top5"]), pct(m["ausuc"]["best_hm"]),
               pct(m["ausuc"]["auc"]), leak_s]
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
            train_cmd = ["-m", "stage3.stage3c.train", "--experiment", exp, *r["args"]]
            if args.epochs:
                train_cmd += ["--epochs", str(args.epochs)]
            run_step(train_cmd)
            run_step(["-m", "stage3.stage3c.evaluate",
                      "--checkpoint", os.path.join(args.output_dir, exp, "best.pt")])

    table = summarize(args.output_dir)
    print("\n" + table)
    out = os.path.join(REPO, args.output_dir, "stage3c_sweep_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# R5 + R6 lambda sweep / unconditional DANN (seed 42)\n\n"
                "Test metrics use the generalized valid-pair protocol (333 pairs). "
                "Cal. HM / AUSUC from the calibration sweep. Selection rule: highest val score.\n\n"
                + table + "\n")
    print(f"\n[sweep] summary -> {out}")


if __name__ == "__main__":
    main()
