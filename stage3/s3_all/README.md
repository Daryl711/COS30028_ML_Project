# Stage 3, run All (R4 + R5 + R6)

See [`stage3/s3_r4r5/README.md`](../s3_r4r5/README.md): both combined runs share one training loop and
evaluator. This package only holds the All config (with the `r6:` section) and its model.

```bash
python -m stage3.s3_all.train                                          # -> outputs/s3_all/
python -m stage3.s3_all.evaluate --checkpoint outputs/s3_all/best.pt
python -m stage3.s3_all.train --unconditional --experiment s3_all_uncond   # plain-DANN ablation
```
