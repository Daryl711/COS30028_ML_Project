"""Train Stage 3 run All (R4 + R5 + R6). Repo root:

  python -m stage3.stage3e.train                                        # -> outputs/stage3e/
  python -m stage3.stage3e.train --epochs 1 --experiment stage3e_smoke    # quick check
  python -m stage3.stage3e.train --unconditional --experiment stage3e_uncond   # plain-DANN ablation
  python -m stage3.stage3e.train --lr_prompt 1e-4 --experiment stage3e_lrp1e-4 # token-LR sub-row

Same loop as stage3/stage3d/train.py; this config's `r6:` section adds the discriminator and L_inv:
  loss = crop CE + disease CE + lambda_kg * L_kg + L_inv  (GRL, DANN ramp, as in stage3/stage3c)
"""
import os

from stage3.stage3d.train import main

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main(default_config=os.path.join(HERE, "config.yaml"))
