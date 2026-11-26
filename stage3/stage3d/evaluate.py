"""Evaluate an R4 + R5 checkpoint. Repo root:

  python -m stage3.stage3d.evaluate --checkpoint outputs/stage3d/best.pt

The checkpoint's "model" entry is an R5Net (this epoch's prompt output as T_c, T_d, plus the adapters),
so this is stage3/stage3b/evaluate.py's protocol unchanged (generalized valid-pair test metrics, HM, AUSUC,
crop-leakage probe on f_d = N(v + A_d(v)), stored as "leakage.r4r5_f_d").
"""
import argparse

from stage3.stage3b.evaluate import load_r5, run


def main(feature_key: str = "r4r5_f_d"):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    run(args.checkpoint, args.out, args.batch_size, load_net=load_r5, feature_key=feature_key)


if __name__ == "__main__":
    main()
