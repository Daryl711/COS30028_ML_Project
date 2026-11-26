"""Evaluate an All (R4 + R5 + R6) checkpoint. Repo root:

  python -m stage3.stage3e.evaluate --checkpoint outputs/stage3e/best.pt

The discriminator is training-only and is not in the checkpoint's "model" entry (an R5Net), so prediction
is exactly R5's protocol. The disease-feature probe is stored as "leakage.all_f_d".
"""
from stage3.stage3d.evaluate import main

if __name__ == "__main__":
    main(feature_key="all_f_d")
