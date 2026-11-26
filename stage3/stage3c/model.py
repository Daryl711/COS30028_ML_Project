"""R5 + R6: R5 plus conditional crop-invariance on the disease feature (PROPOSED_METHOD.md §3.4, §4.4).

  f_d       = N(v + A_d(v))                      R5 disease feature (unchanged)
  h         = R_lambda(f_d)                      gradient reversal: identity forward, -lambda * grad backward
  D_phi(h)  = Linear(256 -> C)(ReLU(Linear(512 -> 256)(h)))     crop discriminator, C = 55
  q(c|h,d)  = softmax of D_phi(h) restricted to C(d), the crops that carry disease d (train split)
  L_inv     = mean over images with |C(d)| >= 2 of  -log q(c_i | h_i, d_i)

The discriminator minimises L_inv (learns to tell the crop apart *within* a disease); through the GRL
the disease adapter A_d receives -lambda * grad L_inv and learns to make that fail. Single-crop
diseases are excluded: there the disease determines the crop, so removing crop information would
remove disease information.

`conditional=False` is the ablation sub-row: every image, softmax over all 55 crops (plain DANN).
The classification path (R5Net.forward) is untouched, so evaluation is exactly R5's.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from stage3.stage3b.model import R5Net


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


def dann_lambda(progress: float, lambda_max: float) -> float:
    """DANN ramp: lambda(p) = lambda_max * (2 / (1 + exp(-10 p)) - 1), p in [0, 1]."""
    p = min(max(progress, 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class R5R6Net(R5Net):
    """R5Net + crop discriminator on f_d behind a GRL."""

    def __init__(self, text_crop, text_disease, logit_scale, crop_mask: torch.Tensor,
                 head_dropout=0.3, bottleneck_dim=256, adapter_dropout=0.1,
                 disc_hidden: int = 256, conditional: bool = True):
        super().__init__(text_crop, text_disease, logit_scale, head_dropout, bottleneck_dim, adapter_dropout)
        dim, num_crops = text_crop.shape[1], text_crop.shape[0]
        assert crop_mask.shape == (text_disease.shape[0], num_crops), crop_mask.shape
        self.conditional = conditional
        # crop_mask[d, c] = crop c carries disease d in the training split
        self.register_buffer("crop_mask", crop_mask.bool())
        self.register_buffer("multi_crop", crop_mask.sum(1) >= 2)
        self.discriminator = nn.Sequential(nn.Linear(dim, disc_hidden), nn.ReLU(),
                                           nn.Linear(disc_hidden, num_crops))

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        return super().param_groups(lr, weight_decay) + [
            {"name": "discriminator", "params": list(self.discriminator.parameters()),
             "lr": lr, "weight_decay": weight_decay}]

    def forward_train(self, v: torch.Tensor, crop: torch.Tensor, disease: torch.Tensor, lam: float) -> dict:
        """Classification logits (as R5) plus the invariance loss and discriminator accuracy."""
        v = v.float()
        f = self.features(v)
        Tc, Td = F.normalize(self.Tc, dim=-1), F.normalize(self.Td, dim=-1)
        out = {"crop": self.logit_scale * F.normalize(self.drop(f["crop"]), dim=-1) @ Tc.T,
               "disease": self.logit_scale * F.normalize(self.drop(f["disease"]), dim=-1) @ Td.T}

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


def crop_mask_from_train(train_df, num_diseases: int, num_crops: int) -> torch.Tensor:
    """[num_diseases, num_crops] bool: which crops carry each disease in the training split."""
    m = torch.zeros(num_diseases, num_crops, dtype=torch.bool)
    for d, c in zip(train_df.disease_label.astype(int), train_df.crop_label.astype(int)):
        m[d, c] = True
    return m
