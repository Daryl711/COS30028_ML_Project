# Stage 3, run All (R4 + R5 + R6)

See [`stage3/stage3d/README.md`](../stage3d/README.md): both combined runs share one training loop and
evaluator. This package only holds the All config (with the `r6:` section) and its model.

```bash
python -m stage3.stage3e.train                                          # -> outputs/stage3e/
python -m stage3.stage3e.evaluate --checkpoint outputs/stage3e/best.pt
python -m stage3.stage3e.train --unconditional --experiment stage3e_uncond   # plain-DANN ablation
```
