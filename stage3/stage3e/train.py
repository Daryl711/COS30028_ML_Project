"""Train Stage 3 run All (R4 + R5 + R6). Repo root:

  python -m stage3.s3_all.train                                        # -> outputs/s3_all/
  python -m stage3.s3_all.train --epochs 1 --experiment s3_all_smoke    # quick check
  python -m stage3.s3_all.train --unconditional --experiment s3_all_uncond   # plain-DANN ablation
  python -m stage3.s3_all.train --lr_prompt 1e-4 --experiment s3_all_lrp1e-4 # token-LR sub-row

Same loop as stage3/s3_r4r5/train.py; this config's `r6:` section adds the discriminator and L_inv:
  loss = crop CE + disease CE + lambda_kg * L_kg + L_inv  (GRL, DANN ramp, as in stage3/s3_r5r6)
"""
import os

from stage3.s3_r4r5.train import main

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main(default_config=os.path.join(HERE, "config.yaml"))
