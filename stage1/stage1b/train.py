"""Train Stage 1b (single-task): two independent EfficientNet-B0 models, one head each. Repo root:

  python stage1/stage1b/train.py --mode crop       # -> outputs/stage1b_crop/
  python stage1/stage1b/train.py --mode disease    # -> outputs/stage1b_disease/

Same code and config as Stage 1a (stage1/stage1a/train.py with --mode crop / disease); this wrapper only
requires --mode and sets the run name. Any other stage1a/train.py flag passes through.
"""
import os
import sys

STAGE1A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1a")
sys.path.insert(0, STAGE1A)

from train import main  # noqa: E402  (stage1/stage1a/train.py)

if __name__ == "__main__":
    args = sys.argv[1:]
    mode = next((args[i + 1] for i, a in enumerate(args[:-1]) if a == "--mode"), None)
    if mode not in ("crop", "disease") and not {"-h", "--help"} & set(args):
        sys.exit("Stage 1b needs --mode crop or --mode disease")
    if "--experiment" not in args:
        sys.argv += ["--experiment", f"stage1b_{mode}"]
    main()
