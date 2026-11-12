"""Small checkpointing and parameter-count helpers."""
from pathlib import Path
import torch


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_metric": best_metric, "config": config}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None: scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
