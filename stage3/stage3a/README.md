# Stage 3, run R4: learnable prompts on the taxonomy-text base

Stage 3 base (2b: frozen BioCLIP, taxonomy-text cosine heads, no logit adjustment), with the class vectors
produced by learnable prompts through the frozen text encoder instead of being free parameters.
Nothing else changes from 2b / R5: same data, split, augmentation, schedule, model selection and evaluator.

```
image (224) -> BioCLIP image encoder (frozen) -> v (512) -> f = N(v)          (no adapters)
   crop    : s * cos(Dropout(f), T_c) -> 55 logits
   disease : s * cos(Dropout(f), T_d) -> 175 logits

T_c = N(E_t( [V^c_1 V^c_2 V^c_3 V^c_4] [θ_crop] leaf, {taxonomy} ))
T_d = N(E_t( [V^d_1 V^d_2 V^d_3 V^d_4] leaf with [θ_disease], {taxonomy} ))
   V : CoOp context, M = 4 per head, initialised from "a photo of a"
   θ : CSP class-name tokens (61 crop + 574 disease subwords), initialised from the names
   E_t : BioCLIP text encoder, frozen, run every step (gradients flow through it to V and θ)

loss = crop CE + disease CE (label smoothing 0.1) + λ_kg · L_kg
L_kg = mean_k ||T_c[k] − T_c⁰[k]||² + mean_k ||T_d[k] − T_d⁰[k]||²      (KgCoOp)
T⁰  = embedding of the unchanged taxonomy prompts = the 2b / R5 initial classifier
λ_kg = 8 (guide.md); 0 for the sub-row
```

## Run (from the repo root)

```bash
python -m stage3.stage3a.train                                         # λ_kg 8 -> outputs/stage3a/
python -m stage3.stage3a.evaluate --checkpoint outputs/stage3a/best.pt

# sub-row without L_kg (guide.md: shows the prompt overfitting that the anchor prevents)
python -m stage3.stage3a.train --lambda_kg 0 --experiment stage3a_nokg
python -m stage3.stage3a.evaluate --checkpoint outputs/stage3a_nokg/best.pt

# lower token-LR sweep sub-row (guide.md): lr {5e-4, 1e-4, 5e-5} x λ_kg {8, 0}, train + evaluate + summary
python -m stage3.stage3a.sweep                          # -> outputs/stage3a_lr*/, outputs/stage3a_lr_sweep_summary.md
python -m stage3.stage3a.sweep --runs lr1e-4 lr1e-4_nokg
python -m stage3.stage3a.train --lr_new 1e-4 --experiment stage3a_lr1e-4   # a single run by hand
```

About 90 s per epoch on an RTX 4050 Laptop (6 GB): the text pass adds about 0.26 s per step, and the
text tower's activations peak at 3.3 GB.

## Design choices

- **Taxonomy prompts kept.** The prompts are the 2b / R5 ones (`taxonomy_prompts()`); only "a photo of a"
  and the class name become learnable. The Colab R4 (`CSP(another_scheme)/`) used plain-name prompts and is not
  comparable (see `results.md`).
- **θ sits only in the label slot.** The class name also appears again inside the taxonomy string
  ("tomato are Solanum lycopersicum, ..."); that copy stays fixed text, as in PROPOSED_METHOD.md §3.3.
- **Init + delta parametrisation.** V and θ are stored as pretrained embedding + a zero-initialised delta.
  This is the same function class as CoOp/CSP. AdamW's weight decay 0.05 then pulls a token toward its
  pretrained embedding rather than toward zero; on absolute tokens it would shrink them by about 16% over
  30 epochs and change what the text encoder reads.
- **LR 1e-3 on the tokens** (training_scheme.md §3 names prompt tokens explicitly, and §6.2 overrides
  PROPOSED_METHOD.md's 5e-4). BioCLIP's token embeddings have a per-dimension std of about 0.011, so with
  Adam this is a large step. CSP's official code uses Adam at 5e-5 (5e-4 in one example), and the Colab
  run used 1e-4. If `drift_*` in `history.csv` explodes in the λ_kg 0 run, that is the reason. A lower LR
  is a named sweep sub-row (training_scheme.md §6.1: sweeps "do not change the default"), not a change
  to the main row.
- **Attribute dropout 0.3 on θ** (training_scheme.md §6.1), applied to the class-name token embeddings
  during training only, as CSP does with its `attr_dropout`. Validation, checkpoints and T⁰ use no dropout.
- **Full 77-token context.** Cutting the padding would be mathematically identical (causal attention) but
  changes fp16 rounding enough to flip one val image at epoch 0. With 77 tokens the prompt path is bit-
  identical to `clip.encode_text`.

## Checks

- `[check]` lines at startup: the prompt path with zero deltas reproduces `encode_text` of the taxonomy
  prompts (cosine 1.000000 for both heads).
- **Epoch 0 equals 2b's and R5's epoch 0 exactly** (`history.csv` row 0: val loss 12.79334, crop 0.4769,
  disease 0.0209).
- The prompt learner asserts that every prompt starts with the template prefix and that each class
  name's tokens sit right after it.

## Outputs

| File | Contents |
|---|---|
| `best.pt`, `last.pt` | `model`: the class vectors T_c, T_d that epoch's prompts produce (+ logit scale), which is what evaluate.py scores; `prompt_learner`: the context and name-token deltas |
| `history.csv` | epoch 0 (zero-shot) and per-epoch metrics; adds `train_kg_loss` (unweighted L_kg) and `drift_crop`, `drift_disease` |
| `config.yaml` | exact config used |
| `eval/metrics.json` | same format as R5: test_seen / test_unseen (generalized valid-pair protocol, 333 pairs), HM, AUSUC, leakage |

R4 has no adapters, so its disease feature is N(v). `leakage.r4_f_d` therefore equals `leakage.base_N_v`
(guide.md: "For the base and R4 the probed feature is N(v)").

## Combining with R5 / R6 later

`R4Net.text()` returns this step's T_c, T_d, and the image side is the 2b one. For R4 + R5 and
All (R4 + R5 + R6), swap `features()` for R5's adapters (and add R6's discriminator on f_d). The prompt
learner and L_kg stay as they are.
