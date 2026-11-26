"""All (R4 + R5 + R6): R4 + R5 plus R6's conditional crop-invariance GRL on the disease feature f_d.

  f_d       = N(v + A_d(v))                     R5 disease feature (from R4R5Net)
  D_phi(R_lambda(f_d))                          crop discriminator behind a GRL, as in stage3/stage3c
  L_inv     = CE over the crops that carry the image's disease, only diseases on >= 2 crops
  loss      = crop CE + disease CE + λ_kg · L_kg + L_inv

The GRL, DANN ramp and crop mask are stage3/stage3c's. The discriminator is training-only, so `frozen()`
(the checkpoint's "model") is R4R5Net's R5Net snapshot and evaluation is exactly R5's.
`conditional=False` is the unconditional-DANN ablation (all images, softmax over all 55 crops).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage3.stage3d.model import R4R5Net
from stage3.stage3c.model import grad_reverse


class AllNet(R4R5Net):
    def __init__(self, clip, tokenizer, prompts, crops, diseases, logit_scale, device, crop_mask: torch.Tensor,
                 head_dropout=0.3, n_ctx=4, attr_dropout=0.3, bottleneck_dim=256, adapter_dropout=0.1,
                 disc_hidden: int = 256, conditional: bool = True):
        super().__init__(clip, tokenizer, prompts, crops, diseases, logit_scale, device, head_dropout, n_ctx,
                         attr_dropout, bottleneck_dim, adapter_dropout)
        num_crops, num_diseases = len(crops), len(diseases)
        assert crop_mask.shape == (num_diseases, num_crops), crop_mask.shape
        dim = self.prompts.T0c.shape[1]
        self.conditional = conditional
        self.register_buffer("crop_mask", crop_mask.bool().to(device))
        self.register_buffer("multi_crop", crop_mask.sum(1).to(device) >= 2)
        self.discriminator = nn.Sequential(nn.Linear(dim, disc_hidden), nn.ReLU(),
                                           nn.Linear(disc_hidden, num_crops)).to(device)

    def param_groups(self, lr, weight_decay, lr_prompt=None):
        return super().param_groups(lr, weight_decay, lr_prompt) + [
            {"name": "discriminator", "params": list(self.discriminator.parameters()),
             "lr": lr, "weight_decay": weight_decay}]

    def forward_train(self, v, T, crop, disease, lam: float) -> dict:
        """Classification logits (as R4 + R5) plus the invariance loss and discriminator accuracy."""
        f = self.features(v)
        out = self.logits(f, T)
        keep = self.multi_crop[disease] if self.conditional else torch.ones_like(disease, dtype=torch.bool)
        n = int(keep.sum())
        if n == 0:
            out["inv_loss"] = f["disease"].sum() * 0.0
            out["inv_n"], out["inv_correct"] = 0, 0
            return out
        logits = self.discriminator(grad_reverse(f["disease"][keep], lam))
        if self.conditional:
            logits = logits.masked_fill(~self.crop_mask[disease[keep]], float("-inf"))
        target = crop[keep]
        out["inv_loss"] = F.cross_entropy(logits, target)
        out["inv_n"] = n
        out["inv_correct"] = int((logits.argmax(1) == target).sum())
        return out
