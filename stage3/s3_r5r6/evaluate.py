"""Evaluate an R5 + R6 checkpoint. Repo root:

  python -m stage3.s3_r5r6.evaluate --checkpoint outputs/s3_r5r6/best.pt

The discriminator is training-only, so prediction is exactly R5's: this loads the R5 part of the
checkpoint and runs stage3/s3_r5/evaluate.py's protocol (test metrics, HM, AUSUC, leakage probe).
The disease-feature probe is stored as "leakage.r5r6_f_d".
"""
import argparse

from stage3.s3_r5.evaluate import load_r5, run

R6_ONLY = ("discriminator.", "crop_mask", "multi_crop")


def load_r5_part(sd: dict, m: dict):
    return load_r5({k: v for k, v in sd.items() if not k.startswith(R6_ONLY)}, m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    run(args.checkpoint, args.out, args.batch_size, load_net=load_r5_part, feature_key="r5r6_f_d")


if __name__ == "__main__":
    main()
