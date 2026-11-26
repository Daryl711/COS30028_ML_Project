# Stage 1b: EfficientNet-B0 single-task baseline

Two independent EfficientNet-B0 models, one crop-only and one disease-only. At evaluation their log-probs are
added to score each valid pair. The code, config and training scheme are Stage 1a's
([`../stage1a/`](../stage1a/README.md)); `train.py` and `evaluate.py` here only call it with the 1b settings.

```bash
python stage1/stage1b/train.py --mode crop       # -> outputs/stage1b_crop/
python stage1/stage1b/train.py --mode disease    # -> outputs/stage1b_disease/
python stage1/stage1b/evaluate.py                # -> outputs/stage1b/eval/
```
