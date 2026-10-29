"""EfficientNet backbone with one linear head per concept, plus the multi-task loss."""
from __future__ import annotations

import timm
import torch
from torch import nn
def heads_for_mode(mode: str, num_crops: int, num_diseases: int) -> dict[str, int]:
    if mode == "multitask":
        return {"crop": num_crops, "disease": num_diseases}
    if mode == "crop":
        return {"crop": num_crops}
    if mode == "disease":
        return {"disease": num_diseases}
    raise ValueError(f"unknown mode '{mode}' (use multitask | crop | disease)")


class EfficientNetHeads(nn.Module):
    """Shared EfficientNet backbone -> pooled feature -> {head_name: logits}.

    multitask: heads = {crop, disease}  (Fig. 1 right)
    independent: two separate models, heads = {crop} and {disease}  (Fig. 1 left)
    """

    def __init__(self, backbone: str, heads: dict[str, int], pretrained: bool = True,
                 dropout: float = 0.3, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=drop_path_rate)
        dim = self.backbone.num_features
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n))
            for name, n in heads.items()
        })

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        out = {name: head(feats) for name, head in self.heads.items()}
        if return_features:
            out["features"] = feats
        return out


def build_model(cfg: dict, heads: dict[str, int], pretrained: bool | None = None) -> EfficientNetHeads:
    m = cfg["model"]
    return EfficientNetHeads(
        backbone=m["backbone"],
        heads=heads,
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        dropout=m.get("dropout", 0.3),
        drop_path_rate=m.get("drop_path_rate", 0.2),
    )


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: dict[str, float], label_smoothing: float = 0.0):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: dict, targets: dict):
        parts = {k: self.ce(outputs[k], targets[k]) for k in self.weights}
        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, {k: v.detach().item() for k, v in parts.items()}
