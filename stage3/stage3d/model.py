"""R4 + R5: learnable prompts (R4) on the text side, separate adapters (R5) on the image side.

  v   = BioCLIP.encode_image(x)                       512-d, frozen (kept outside this module)
  f_c = N(v + A_c(v))                                 R5 crop feature
  f_d = N(v + A_d(v))                                 R5 disease feature
  T_c = N(E_t([V^c] [θ_crop] leaf, {taxonomy}))       R4 prompts through the frozen text encoder
  T_d = N(E_t([V^d] leaf with [θ_disease], {taxonomy}))
  crop logits    = s * cos(Dropout(f_c), T_c)
  disease logits = s * cos(Dropout(f_d), T_d)
  loss = crop CE + disease CE + λ_kg · L_kg           L_kg as in R4 (KgCoOp anchor to T⁰)

Both halves are reused unchanged: the prompt learner is stage3/s3_r4's R4Net (context, name tokens,
attribute dropout, T⁰, drift), the adapters are stage3/s3_r5's Adapter. At epoch 0 the token deltas are
zero and the adapters are the identity, so the model equals 2b's epoch 0 exactly.

`frozen()` returns an ordinary R5Net whose T_c, T_d are this epoch's prompt output and whose adapters are
copied over. That is what the checkpoint's "model" entry holds, so stage3/s3_r5/evaluate.py scores it
unchanged (no text encoder at prediction time).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage3.s3_r4.model import R4Net
from stage3.s3_r5.model import Adapter, R5Net


class R4R5Net(nn.Module):
    """Everything trainable in R4 + R5: prompt tokens (R4) and the two adapters (R5). Input is the raw v."""

    def __init__(self, clip, tokenizer, prompts: dict[str, list[str]], crops: list[str], diseases: list[str],
                 logit_scale: float, device: torch.device, head_dropout: float = 0.3, n_ctx: int = 4,
                 attr_dropout: float = 0.3, bottleneck_dim: int = 256, adapter_dropout: float = 0.1):
        super().__init__()
        self.prompts = R4Net(clip, tokenizer, prompts, crops, diseases, logit_scale, head_dropout, n_ctx,
                             device, attr_dropout)
        dim = self.prompts.T0c.shape[1]
        self.adapter_crop = Adapter(dim, bottleneck_dim, adapter_dropout).to(device)
        self.adapter_disease = Adapter(dim, bottleneck_dim, adapter_dropout).to(device)
        self.drop = nn.Dropout(head_dropout)
        self.head_dropout, self.bottleneck_dim, self.adapter_dropout = head_dropout, bottleneck_dim, adapter_dropout

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.prompts.logit_scale

    def param_groups(self, lr: float, weight_decay: float, lr_prompt: float | None = None) -> list[dict]:
        """One AdamW param group per trainable module (training_scheme.md §3). `lr_prompt` sets the prompt
        tokens' LR separately (the R4 token-LR sweep sub-row); None = the same LR as every new module."""
        groups = self.prompts.param_groups(lr if lr_prompt is None else lr_prompt, weight_decay)
        for name, mod in (("crop_adapter", self.adapter_crop), ("disease_adapter", self.adapter_disease)):
            groups.append({"name": name, "params": list(mod.parameters()), "lr": lr, "weight_decay": weight_decay})
        return groups

    def text(self) -> dict[str, torch.Tensor]:
        return self.prompts.text()

    def drift(self, T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.prompts.drift(T)

    def features(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        v = v.float()
        return {"crop": F.normalize(self.adapter_crop(v), dim=-1),
                "disease": F.normalize(self.adapter_disease(v), dim=-1)}

    def logits(self, f: dict[str, torch.Tensor], T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s = self.logit_scale
        return {"crop": s * F.normalize(self.drop(f["crop"]), dim=-1) @ T["crop"].float().T,
                "disease": s * F.normalize(self.drop(f["disease"]), dim=-1) @ T["disease"].float().T}

    def forward(self, v: torch.Tensor, T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Logits for image features v, given this step's text features T (from `text()`)."""
        return self.logits(self.features(v), T)

    def frozen(self) -> R5Net:
        """Snapshot of the current classifier as an R5Net (eval-mode prompts, adapters copied)."""
        T = self.prompts._eval_text()
        head = R5Net(T["crop"], T["disease"], self.logit_scale.item(), self.head_dropout,
                     self.bottleneck_dim, self.adapter_dropout).to(self.logit_scale.device)
        head.adapter_crop.load_state_dict(self.adapter_crop.state_dict())
        head.adapter_disease.load_state_dict(self.adapter_disease.state_dict())
        return head.eval()
