"""BioCLIP ViT-B/16 (frozen) with two post-encoder disentangling adapters (R5).

Architecture (per PROPOSED_METHOD.md §3.2):
  v = BioCLIP.encode_image(x)              # 512-dim, backbone frozen
  f_c = normalize(v + A_c(v))             # crop adapter
  f_d = normalize(v + A_d(v))             # disease adapter
  A_k : 512 → bottleneck_dim → 512, GELU, dropout 0.1, last layer zero-initialized
  crop logits   = head_c(f_c)
  disease logits = head_d(f_d)

Zero-initialization of the up-projection means each adapter starts as the identity
map (A_k(v) = 0), so the model starts exactly at the frozen-BioCLIP baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DisentanglingAdapter(nn.Module):
    """Residual MLP adapter: f = normalize(v + up(dropout(gelu(down(v))))).

    The up-projection is zero-initialized so the adapter starts as the identity.
    """
    def __init__(self, dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v + self.up(self.drop(self.act(self.down(v)))), dim=-1)


def _visual_dimension(visual: nn.Module) -> int:
    projection = getattr(visual, "proj", None)
    if projection is None:
        raise RuntimeError("BioCLIP visual tower has no pretrained visual projection.")
    return int(projection.shape[-1])


class BioCLIPAdapterClassifier(nn.Module):
    def __init__(self, num_crops: int, num_diseases: int,
                 pretrained_name: str = "hf-hub:imageomics/bioclip",
                 bottleneck_dim: int = 256, adapter_dropout: float = 0.1,
                 dropout: float = 0.0, model: nn.Module | None = None):
        super().__init__()
        if model is None:
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError("Install open_clip_torch to load BioCLIP.") from exc
            model, _, _ = open_clip.create_model_and_transforms(pretrained_name)
        self.clip = model
        visual = self.clip.visual
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise RuntimeError("Unsupported BioCLIP visual tower: expected OpenCLIP transformer.resblocks.")
        width = int(blocks[0].ln_1.normalized_shape[0])
        if width != 768 or len(blocks) != 12:
            raise RuntimeError(
                f"Expected BioCLIP ViT-B/16 width=768, layers=12; got width={width}, layers={len(blocks)}")
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.feature_dim = _visual_dimension(visual)  # 512 for ViT-B/16
        self.adapters = nn.ModuleDict({
            "crop":    DisentanglingAdapter(self.feature_dim, bottleneck_dim, adapter_dropout),
            "disease": DisentanglingAdapter(self.feature_dim, bottleneck_dim, adapter_dropout),
        })
        self.heads = nn.ModuleDict({
            "crop":    nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_crops)),
            "disease": nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_diseases)),
        })

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(images)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        v = self.forward_features(images)
        return {name: self.heads[name](self.adapters[name](v))
                for name in ("crop", "disease")}

    def verify_trainability(self) -> None:
        assert all(not p.requires_grad for p in self.clip.parameters()), \
            "BioCLIP backbone must be fully frozen"
        assert all(p.requires_grad for a in self.adapters.values() for p in a.parameters()), \
            "All adapter parameters must be trainable"
        assert all(p.requires_grad for p in self.heads.parameters()), \
            "All head parameters must be trainable"


def build_model(num_crops: int, num_diseases: int, **kwargs) -> BioCLIPAdapterClassifier:
    kwargs.pop("adapter_blocks", None)  # R5 has no transformer-block adapters
    return BioCLIPAdapterClassifier(num_crops, num_diseases, **kwargs)
