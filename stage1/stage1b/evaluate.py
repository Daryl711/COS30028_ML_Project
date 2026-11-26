"""Evaluate Stage 1b: the crop model and the disease model together (log-probs added). Repo root:

  python stage1/stage1b/evaluate.py      # outputs/stage1b_{crop,disease}/best.pt -> outputs/stage1b/eval

Same evaluator as Stage 1a (stage1/stage1a/evaluate.py); this wrapper only fills in the default
--checkpoints and --out. Pass either flag to override it.
"""
import os
import sys

STAGE1A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1a")
sys.path.insert(0, STAGE1A)

from evaluate import main  # noqa: E402  (stage1/stage1a/evaluate.py)

if __name__ == "__main__":
    if "--checkpoints" not in sys.argv:
        sys.argv += ["--checkpoints", "outputs/stage1b_crop/best.pt", "outputs/stage1b_disease/best.pt"]
    if "--out" not in sys.argv:
        sys.argv += ["--out", "outputs/stage1b/eval"]
    main()
