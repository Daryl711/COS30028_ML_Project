"""Evaluate an R4 checkpoint. Repo root:

  python -m stage3.s3_r4.evaluate --checkpoint outputs/s3_r4/best.pt

The checkpoint's "model" entry holds the class vectors T_c, T_d that the learned prompts produced at that
epoch, so prediction needs no text encoder. This runs stage3/s3_r5/evaluate.py's protocol unchanged
(generalized valid-pair test metrics, HM, AUSUC, crop-leakage probe).

R4 has no adapters, so its disease feature is N(v): "leakage.r4_f_d" equals "leakage.base_N_v" by
construction (guide.md: "For the base and R4 the probed feature is N(v)").
"""
import argparse

from stage3.s3_r5.evaluate import run

from .model import TextHeadNet


def load_r4(sd: dict, m: dict) -> TextHeadNet:
    net = TextHeadNet(sd["Tc"], sd["Td"], float(sd["logit_scale"]), m["head_dropout"])
    net.load_state_dict(sd)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    run(args.checkpoint, args.out, args.batch_size, load_net=load_r4, feature_key="r4_f_d")


if __name__ == "__main__":
    main()
