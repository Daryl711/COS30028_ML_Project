"""R5: 2b text heads + separate crop and disease adapters (guide.md Stage 3, PROPOSED_METHOD.md §3.2).

  v   = BioCLIP.encode_image(x)                 512-d, frozen (kept outside this module)
  f_c = N(v + A_c(v))                           crop feature
  f_d = N(v + A_d(v))                           disease feature
  crop logits    = s * cos(Dropout(f_c), T_c)    s = BioCLIP's pretrained logit scale, fixed
  disease logits = s * cos(Dropout(f_d), T_d)
  A_k: 512 -> 256 -> 512, GELU, dropout 0.1, up-projection zero-initialised
  T_c, T_d: free parameters initialised from the frozen BioCLIP text embedding of the taxonomy prompts

The zero-initialised up-projection makes each adapter the identity at epoch 0, so R5 starts exactly
at 2b (stage2/stage2b/basic_text_colab.ipynb `Net`), and with A_c = A_d = 0 the forward pass is the
2b forward pass.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Adapter(nn.Module):
    """Residual MLP: v -> v + up(drop(gelu(down(v)))), up-projection zero-initialised."""

    def __init__(self, dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v + self.up(self.drop(self.act(self.down(v))))


class R5Net(nn.Module):
    """Everything trainable in R5. Input is the raw BioCLIP image feature v."""

    def __init__(self, text_crop: torch.Tensor, text_disease: torch.Tensor, logit_scale: float,
                 head_dropout: float = 0.3, bottleneck_dim: int = 256, adapter_dropout: float = 0.1):
        super().__init__()
        dim = text_crop.shape[1]
        self.Tc = nn.Parameter(text_crop.clone().float())
        self.Td = nn.Parameter(text_disease.clone().float())
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))  # fixed, not trained
        self.adapter_crop = Adapter(dim, bottleneck_dim, adapter_dropout)
        self.adapter_disease = Adapter(dim, bottleneck_dim, adapter_dropout)
        self.drop = nn.Dropout(head_dropout)

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """One AdamW param group per trainable module (training_scheme.md §3)."""
        mods = (("crop_text", [self.Tc]), ("disease_text", [self.Td]),
                ("crop_adapter", list(self.adapter_crop.parameters())),
                ("disease_adapter", list(self.adapter_disease.parameters())))
        return [{"name": n, "params": p, "lr": lr, "weight_decay": weight_decay} for n, p in mods]

    def features(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"crop": F.normalize(self.adapter_crop(v), dim=-1),
                "disease": F.normalize(self.adapter_disease(v), dim=-1)}

    def forward(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.features(v.float())
        Tc, Td = F.normalize(self.Tc, dim=-1), F.normalize(self.Td, dim=-1)
        return {"crop": self.logit_scale * F.normalize(self.drop(f["crop"]), dim=-1) @ Tc.T,
                "disease": self.logit_scale * F.normalize(self.drop(f["disease"]), dim=-1) @ Td.T}


# ---------------------------------------------------------------- frozen BioCLIP
def load_bioclip(name: str, device: torch.device):
    """BioCLIP ViT-B/16 with every parameter frozen, in eval mode. Returns (model, tokenizer)."""
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(name)
    clip = clip.to(device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)
    blocks = clip.visual.transformer.resblocks
    width = int(blocks[0].ln_1.normalized_shape[0])
    if width != 768 or len(blocks) != 12:
        raise RuntimeError(f"expected BioCLIP ViT-B/16 (width 768, 12 layers); got {width}, {len(blocks)}")
    return clip, open_clip.get_tokenizer(name)


@torch.no_grad()
def encode_text(clip, tokenizer, prompts: list[str], device: torch.device, amp: bool) -> torch.Tensor:
    tokens = tokenizer(prompts)
    longest = int((tokens != 0).sum(1).max())
    if longest >= 77:
        raise ValueError("a prompt hit the 77-token limit")
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        return F.normalize(clip.encode_text(tokens.to(device)).float(), dim=-1)


@torch.no_grad()
def encode_images(clip, loader, device: torch.device, amp: bool) -> torch.Tensor:
    """Frozen image features for a deterministic (eval-transform) loader."""
    from tqdm import tqdm
    out = []
    for b in tqdm(loader, leave=False, desc="features"):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out.append(clip.encode_image(b["image"].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def logits_from_features(net: R5Net, V: torch.Tensor, device: torch.device, batch: int = 1024):
    net.eval()
    crop, dis = [], []
    for i in range(0, len(V), batch):
        o = net(V[i:i + batch].to(device))
        crop.append(o["crop"].float().cpu())
        dis.append(o["disease"].float().cpu())
    return torch.cat(crop).numpy(), torch.cat(dis).numpy()
