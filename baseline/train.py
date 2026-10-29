"""Train an EfficientNet baseline on DPD-Small seen data.

  python baseline/train.py                                                   # multi-task
  python baseline/train.py --mode crop    --experiment indep_crop_effb0      # independent: crop
  python baseline/train.py --mode disease --experiment indep_disease_effb0   # independent: disease
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset import (PlantDiseaseDataset, audit, build_label_space, build_transforms,
                     read_split, split_train_val)
from model import MultiTaskLoss, build_model, heads_for_mode


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    """Linear warmup then cosine decay, stepped once per batch."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--mode", choices=["multitask", "crop", "disease"])
    p.add_argument("--experiment")
    p.add_argument("--epochs", type=int)
    p.add_argument("--backbone")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, amp, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    heads = list(criterion.weights)
    total_loss, n = 0.0, 0
    correct = {h: 0 for h in heads}
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        x = batch["image"].to(device, non_blocking=True)
        targets = {h: batch[h].to(device, non_blocking=True) for h in heads}
        with torch.set_grad_enabled(train), \
                torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss, _ = criterion(out, targets)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        for h in heads:
            correct[h] += (out[h].argmax(1) == targets[h]).sum().item()
    return {"loss": total_loss / max(n, 1), **{f"{h}_acc": correct[h] / max(n, 1) for h in heads}}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key in ("mode", "experiment"):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone

    seed_everything(cfg["seed"])
    device = get_device()
    tc, dc = cfg["train"], cfg["data"]
    amp = bool(tc.get("amp", True)) and device.type == "cuda"

    root = dc["root"]
    train_full = read_split(os.path.join(root, dc["train_csv"]))
    seen_df = read_split(os.path.join(root, dc["test_seen_csv"]))
    unseen_df = read_split(os.path.join(root, dc["test_unseen_csv"]))
    ls = build_label_space(train_full, seen_df, unseen_df)
    audit(train_full, seen_df, unseen_df, ls)

    # Model selection uses a val split carved from train. Test sets are never touched here.
    train_df, val_df = split_train_val(train_full, dc["val_fraction"], cfg["seed"])
    print(f"[data] train={len(train_df)} val={len(val_df)}")
    image_dir = os.path.join(root, dc["image_dir"])
    loader_kw = dict(batch_size=tc["batch_size"], num_workers=dc["num_workers"],
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(PlantDiseaseDataset(train_df, image_dir, build_transforms(dc["img_size"], True)),
                              shuffle=True, drop_last=len(train_df) > tc["batch_size"], **loader_kw)
    val_loader = None
    if len(val_df):
        val_loader = DataLoader(PlantDiseaseDataset(val_df, image_dir, build_transforms(dc["img_size"], False)),
                                shuffle=False, **loader_kw)

    heads = heads_for_mode(cfg["mode"], ls.num_crops, ls.num_diseases)
    model = build_model(cfg, heads).to(device)
    print(f"[model] {cfg['model']['backbone']} heads={heads} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": tc["lr_backbone"]},
        {"params": model.heads.parameters(), "lr": tc["lr_head"]},
    ], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, tc["warmup_epochs"] * steps_per_epoch,
                                   tc["epochs"] * steps_per_epoch)
    criterion = MultiTaskLoss({h: tc["loss_weights"][h] for h in heads}, tc["label_smoothing"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    out_dir = os.path.join(cfg["output_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    history_path = os.path.join(out_dir, "history.csv")

    best_score, best_epoch, history = -1.0, -1, []
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, amp, optimizer, scheduler, scaler)
        va = run_epoch(model, val_loader, criterion, device, amp) if val_loader else tr
        score = float(np.mean([va[f"{h}_acc"] for h in heads]))  # mean top-1 over heads

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, "val_score": score,
               "lr_backbone": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0}
        history.append(row)
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

        ckpt = {"model": model.state_dict(), "heads": heads, "config": cfg, "epoch": epoch,
                "val": va, "num_crops": ls.num_crops, "num_diseases": ls.num_diseases}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
        accs = " ".join(f"{h}={va[f'{h}_acc']:.4f}" for h in heads)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} {accs}"
              f"{' *' if improved else ''} | {row['time_s']:.0f}s")

        if tc.get("patience", 0) and epoch - best_epoch >= tc["patience"]:
            print(f"[train] early stop: no val improvement for {tc['patience']} epochs")
            break

    print(f"[train] best val score {best_score:.4f} at epoch {best_epoch} -> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
