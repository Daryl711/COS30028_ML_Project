"""Train Stage 3 run R5 (2b + separate adapters A_c, A_d). Run from the repo root:

  python -m stage3.stage3b.train
  python -m stage3.stage3b.train --epochs 1 --experiment stage3b_smoke     # quick check

training_scheme.md §1-§5 throughout; the only change from 2b is the two adapters.
BioCLIP is frozen, so val features are computed once (eval transform is deterministic) and reused.
Train images are never cached: they are augmented online every step.
"""
import argparse
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (PlantDiseaseDataset, build_transforms, class_names, read_split,
                      split_train_val, taxonomy_prompts)
from .metrics import build_pairs, log_probs
from .model import R5Net, encode_images, encode_text, load_bioclip, logits_from_features

HERE = os.path.dirname(os.path.abspath(__file__))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    """Linear warm-up then cosine decay to min_ratio of the peak, stepped per optimizer step."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def load_data(cfg: dict):
    """Train/val split, test frames, class names and taxonomy prompts, with the 2b sanity checks."""
    dc = cfg["data"]
    root = dc["root"]
    train_full = read_split(os.path.join(root, dc["train_csv"]))
    seen = read_split(os.path.join(root, dc["test_seen_csv"]))
    unseen = read_split(os.path.join(root, dc["test_unseen_csv"]))
    nc, nd = int(train_full.crop_label.max()) + 1, int(train_full.disease_label.max()) + 1
    assert (nc, nd) == (55, 175), (nc, nd)
    train_df, val_df = split_train_val(train_full, dc["val_fraction"], cfg["seed"])
    pairs, num_seen = build_pairs(train_full, unseen)
    assert (num_seen, len(pairs) - num_seen) == (323, 10), (num_seen, len(pairs))
    crops, diseases = class_names(train_full)
    prompts = taxonomy_prompts(os.path.join(root, dc["text_dir"]), crops, diseases)
    return dict(train_full=train_full, train=train_df, val=val_df, seen=seen, unseen=unseen,
                crops=crops, diseases=diseases, prompts=prompts)


def build_net(cfg, clip, tokenizer, prompts, device, amp) -> R5Net:
    m = cfg["model"]
    text_c = encode_text(clip, tokenizer, prompts["crop"], device, amp)
    text_d = encode_text(clip, tokenizer, prompts["disease"], device, amp)
    return R5Net(text_c, text_d, clip.logit_scale.exp().item(), m["head_dropout"],
                 m["bottleneck_dim"], m["adapter_dropout"]).to(device)


def val_stats(net, V, val_df, device) -> dict:
    lc, ld = logits_from_features(net, V, device)
    yc, yd = val_df.crop_label.to_numpy(), val_df.disease_label.to_numpy()
    r = np.arange(len(yc))
    crop_acc, dis_acc = float((lc.argmax(1) == yc).mean()), float((ld.argmax(1) == yd).mean())
    return {"val_loss": float(-(log_probs(lc)[r, yc] + log_probs(ld)[r, yd]).mean()),
            "val_crop_acc": crop_acc, "val_disease_acc": dis_acc, "val_score": (crop_acc + dis_acc) / 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    p.add_argument("--experiment")
    p.add_argument("--epochs", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.experiment:
        cfg["experiment"] = args.experiment
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    tc, dc = cfg["train"], cfg["data"]

    seed_everything(cfg["seed"])
    device = get_device()
    amp = bool(tc["amp"]) and device.type == "cuda"
    if device.type != "cuda":
        print("[warn] no CUDA device: training on CPU will be very slow")

    data = load_data(cfg)
    train_df, val_df = data["train"], data["val"]
    print(f"[data] train={len(train_df)} val={len(val_df)} test_seen={len(data['seen'])} "
          f"test_unseen={len(data['unseen'])}")
    print(f"[text] {data['prompts']['crop'][0]!r}\n[text] {data['prompts']['disease'][0]!r}")

    clip, tokenizer = load_bioclip(cfg["model"]["backbone"], device)
    net = build_net(cfg, clip, tokenizer, data["prompts"], device, amp)
    print(f"[model] logit scale {net.logit_scale.item():.2f} | trainable params "
          f"{sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    image_dir = os.path.join(dc["root"], dc["image_dir"])
    loader_kw = dict(num_workers=dc["num_workers"], pin_memory=device.type == "cuda",
                     persistent_workers=dc["num_workers"] > 0)
    V_val = encode_images(clip, DataLoader(PlantDiseaseDataset(val_df, image_dir, build_transforms(dc["img_size"], False)),
                                           batch_size=128, shuffle=False, num_workers=dc["num_workers"]), device, amp)
    train_loader = DataLoader(PlantDiseaseDataset(train_df, image_dir, build_transforms(dc["img_size"], True)),
                              batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kw)

    optimizer = torch.optim.AdamW(net.param_groups(tc["lr_new"], tc["weight_decay"]))
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    spe = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, tc["warmup_epochs"] * spe, tc["epochs"] * spe, tc["min_lr_ratio"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    out_dir = os.path.join(cfg["output_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Epoch 0: adapters are the identity, so this equals 2b's epoch 0 (BioCLIP zero-shot, taxonomy text).
    v0 = val_stats(net, V_val, val_df, device)
    history = [{"epoch": 0, **v0}]
    print(f"epoch  0 (zero-shot = 2b epoch 0) | val crop {v0['val_crop_acc']:.4f} "
          f"dis {v0['val_disease_acc']:.4f} score {v0['val_score']:.4f}")

    def ckpt(epoch, va):
        return {"model": net.state_dict(), "epoch": epoch, "val": va, "config": cfg,
                "crops": data["crops"], "diseases": data["diseases"]}

    ls = tc["label_smoothing"]
    best_score, best_epoch = -1.0, -1
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        net.train()
        tl = n = ca = da = 0
        for b in tqdm(train_loader, leave=False, desc=f"{cfg['experiment']} ep{epoch}"):
            x = b["image"].to(device, non_blocking=True)
            yc, yd = b["crop"].to(device), b["disease"].to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                v = clip.encode_image(x)
            o = net(v.float())                                  # trainable part in fp32, as in 2b
            lc, ld = o["crop"].float(), o["disease"].float()
            loss = 1.0 * F.cross_entropy(lc, yc, label_smoothing=ls) + 1.0 * F.cross_entropy(ld, yd, label_smoothing=ls)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = len(yc)
            tl += loss.item() * bs
            n += bs
            ca += (lc.argmax(1) == yc).sum().item()
            da += (ld.argmax(1) == yd).sum().item()

        va = val_stats(net, V_val, val_df, device)
        history.append({"epoch": epoch, "train_loss": tl / n, "train_crop_acc": ca / n, "train_disease_acc": da / n,
                        **va, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        torch.save(ckpt(epoch, va), os.path.join(out_dir, "last.pt"))
        improved = va["val_score"] > best_score
        if improved:
            best_score, best_epoch = va["val_score"], epoch
            torch.save(ckpt(epoch, va), os.path.join(out_dir, "best.pt"))
        print(f"epoch {epoch:2d} | train loss {tl / n:.4f} crop {ca / n:.3f} dis {da / n:.3f} | "
              f"val crop {va['val_crop_acc']:.4f} dis {va['val_disease_acc']:.4f} score {va['val_score']:.4f}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
        if tc["patience"] and epoch - best_epoch >= tc["patience"]:
            print(f"[train] early stop: no val improvement for {tc['patience']} epochs")
            break

    print(f"[train] best val score {best_score:.4f} at epoch {best_epoch} -> {out_dir}/best.pt")
    print(f"[next] python -m stage3.stage3b.evaluate --checkpoint {out_dir}/best.pt")


if __name__ == "__main__":
    main()
