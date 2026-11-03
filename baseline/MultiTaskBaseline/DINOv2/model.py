"""DINOv2 ViT-S/14 backbone with one linear head per concept, plus the multi-task loss.

Same two-head design as the EfficientNet baselines. What changes:
  - the backbone is a Vision Transformer pretrained by DINOv2 (self-supervised, no labels)
  - the model is built for 224 px input (timm's default for DINOv2 is 518 px)
  - the optimizer uses layer-wise learning-rate decay, so early layers change less than late ones
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from timm.optim import param_groups_layer_decay


def heads_for_mode(mode: str, num_crops: int, num_diseases: int) -> dict[str, int]:
    if mode == "multitask":
        return {"crop": num_crops, "disease": num_diseases}
    if mode == "crop":
        return {"crop": num_crops}
    if mode == "disease":
        return {"disease": num_diseases}
    raise ValueError(f"unknown mode '{mode}' (use multitask | crop | disease)")


class DinoV2Heads(nn.Module):
    """Shared DINOv2 ViT backbone -> CLS token feature -> {head_name: logits}.

    multitask: heads = {crop, disease}  (Fig. 1 right)
    independent: two separate models, heads = {crop} and {disease}  (Fig. 1 left)
    """

    def __init__(self, backbone: str, heads: dict[str, int], img_size: int = 224,
                 pretrained: bool = True, dropout: float = 0.3, drop_path_rate: float = 0.1):
        super().__init__()
        # img_size must be a multiple of the patch size (14). timm resizes the pretrained
        # position embeddings from 518 px (37x37 patches) to this size when loading weights.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          img_size=img_size, drop_path_rate=drop_path_rate)
        dim = self.backbone.num_features  # 384 for ViT-S
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n))
            for name, n in heads.items()
        })

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)  # CLS token after the final norm
        out = {name: head(feats) for name, head in self.heads.items()}
        if return_features:
            out["features"] = feats
        return out


def build_model(cfg: dict, heads: dict[str, int], pretrained: bool | None = None) -> DinoV2Heads:
    m = cfg["model"]
    return DinoV2Heads(
        backbone=m["backbone"],
        heads=heads,
        img_size=cfg["data"]["img_size"],
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        dropout=m.get("dropout", 0.3),
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


def build_optimizer(model: DinoV2Heads, tc: dict) -> torch.optim.Optimizer:
    """AdamW with layer-wise learning-rate decay for the backbone.

    The last transformer block gets lr_backbone, the one before it lr_backbone * decay, and so on
    down to the patch embedding. Early layers hold the general DINOv2 features, so they move least.
    Biases, norms, position embedding and CLS token get no weight decay (standard for ViTs).
    The heads are new, so they get their own larger learning rate.
    """
    groups = param_groups_layer_decay(
        model.backbone,
        weight_decay=tc["weight_decay"],
        no_weight_decay_list=model.backbone.no_weight_decay(),
        layer_decay=tc.get("layer_decay", 0.75),
    )
    for g in groups:
        g["lr"] = tc["lr_backbone"] * g.pop("lr_scale")
        g["name"] = "backbone"
    groups.append({"name": "heads", "params": list(model.heads.parameters()),
                   "lr": tc["lr_head"], "weight_decay": tc["weight_decay"]})
    return torch.optim.AdamW(groups)


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: dict[str, float], label_smoothing: float = 0.0):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: dict, targets: dict):
        parts = {k: self.ce(outputs[k], targets[k]) for k in self.weights}
        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, {k: v.detach().item() for k, v in parts.items()}
