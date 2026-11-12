"""Three-seed DPD compositional protocol: dev fit by HM, full refit, one test evaluation."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import Config
    from .dataset import (PlantDiseaseDataset, build_transforms,
                          compositional_validation_split, read_split)
    from .metrics import ausuc, harmonic_mean, split_metrics
    from .model import build_model
    from .utils import parameter_counts, save_checkpoint
except ImportError:
    from config import Config
    from dataset import (PlantDiseaseDataset, build_transforms,
                        compositional_validation_split, read_split)
    from metrics import ausuc, harmonic_mean, split_metrics
    from model import build_model
    from utils import parameter_counts, save_checkpoint


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_warmup(optimizer, total_steps, warmup_fraction, min_ratio=0.01):
    warm = int(total_steps * warmup_fraction)
    def scale(step):
        if warm and step < warm: return (step + 1) / warm
        progress = min(1.0, max(0.0, (step - warm) / max(1, total_steps - warm)))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=bool(enabled and device.type == "cuda"))


def make_model(cfg, n_crops, n_diseases, device):
    model = build_model(n_crops, n_diseases, pretrained_name=cfg.model.pretrained_name,
                        bottleneck_dim=cfg.model.bottleneck_dim,
                        adapter_dropout=cfg.model.adapter_dropout,
                        dropout=cfg.model.dropout).to(device)
    model.verify_trainability()
    # Cheap startup smoke check on every fresh dev/final model instance.
    model.train(); model.zero_grad(set_to_none=True)
    out = model(torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size, device=device))
    assert out["crop"].shape == (2, n_crops)
    assert out["disease"].shape == (2, n_diseases)
    (out["crop"].sum() + out["disease"].sum()).backward()
    adapter_params = [p for a in model.adapters.values() for p in a.parameters()]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter_params)
    assert not any(p.grad is not None for p in model.clip.parameters())
    model.zero_grad(set_to_none=True)
    print(f"[check] dummy batch passed: crop={tuple(out['crop'].shape)}, disease={tuple(out['disease'].shape)}")
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(
        p.numel() for p in model.parameters())
    print(f"[model] BioCLIP ViT-B/16 (frozen); adapters=crop+disease "
          f"(512->{cfg.model.bottleneck_dim}->512); "
          f"heads=crop({n_crops})+disease({n_diseases}); trainable={trainable:,}/{total:,}")
    return model


def make_optimizer(model, cfg):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)


def train_epoch(model, loader, device, optimizer, scheduler, mixed_precision, desc):
    model.train(); criterion = nn.CrossEntropyLoss()
    total_loss = 0.0; count = 0
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        crop = batch["crop"].to(device, non_blocking=True)
        disease = batch["disease"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, mixed_precision):
            out = model(images)
            loss = criterion(out["crop"].float(), crop) + criterion(out["disease"].float(), disease)
        loss.backward()
        optimizer.step(); scheduler.step()
        total_loss += loss.item() * len(crop); count += len(crop)
    return total_loss / max(1, count)


@torch.no_grad()
def collect_logits(model, loader, device, mixed_precision):
    model.eval(); crops, diseases, targets_crop, targets_disease = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with autocast(device, mixed_precision):
            out = model(images)
        crops.append(out["crop"].float().cpu().numpy())
        diseases.append(out["disease"].float().cpu().numpy())
        targets_crop.append(batch["crop"].numpy())
        targets_disease.append(batch["disease"].numpy())
    # targets are retained in their dataframe; these lists ensure one-to-one loader traversal.
    return np.concatenate(crops), np.concatenate(diseases)


def eval_split(model, frame, loader, device, cfg, n_diseases, unseen_pairs=()):
    crop_logits, disease_logits = collect_logits(model, loader, device, cfg.train.mixed_precision == "bf16")
    result, scores, targets = split_metrics(crop_logits, disease_logits, frame, n_diseases, unseen_pairs)
    return result, scores, targets


def loader_for(frame, image_dir, cfg, train, views=1, batch_size=None, shuffle=False):
    clean = build_transforms(cfg.data.image_size, False)
    aug = build_transforms(cfg.data.image_size, True) if train and views > 1 else None
    ds = PlantDiseaseDataset(frame, image_dir, transform=clean, train_transform=aug,
                             views=views, base_seed=cfg.data.view_base_seed)
    return DataLoader(ds, batch_size=batch_size or cfg.train.batch_size, shuffle=shuffle,
                      num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available())


def fit_dev(seed, cfg, train_df, val_seen, val_unseen, image_dir, n_crops, n_diseases, device, out_dir):
    seed_all(seed)
    model = make_model(cfg, n_crops, n_diseases, device)
    n_trainable, n_total = parameter_counts(model)
    print(f"[seed {seed}] dev model trainable={n_trainable:,}/{n_total:,}")
    opt = make_optimizer(model, cfg)
    train_loader = loader_for(train_df, image_dir, cfg, True,
                              cfg.data.train_views, shuffle=True)
    val_seen_loader = loader_for(val_seen, image_dir, cfg, False, batch_size=cfg.train.batch_size * 2)
    val_unseen_loader = loader_for(val_unseen, image_dir, cfg, False, batch_size=cfg.train.batch_size * 2)
    sched = cosine_warmup(opt, max(1, len(train_loader) * cfg.train.epochs),
                          cfg.train.warmup_fraction, cfg.train.min_lr_ratio)
    best_hm, best_epoch, stale = -1.0, 0, 0
    best_path = os.path.join(out_dir, f"seed_{seed}", "dev_best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    seen_pairs = set(zip(train_df.crop_label.tolist(), train_df.disease_label.tolist()))
    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, opt, sched, cfg.train.mixed_precision == "bf16",
                              f"seed {seed} dev {epoch}/{cfg.train.epochs}")
        seen_metrics, _, _ = eval_split(model, val_seen, val_seen_loader, device, cfg, n_diseases)
        unseen_metrics, _, _ = eval_split(model, val_unseen, val_unseen_loader, device, cfg, n_diseases,
                                          set(zip(val_unseen.crop_label.tolist(), val_unseen.disease_label.tolist())))
        hm = harmonic_mean(seen_metrics["top1"], unseen_metrics["top1"])
        print(f"[seed {seed}] dev epoch={epoch} loss={tr_loss:.4f} val_seen={seen_metrics['top1']:.4f} "
              f"val_unseen={unseen_metrics['top1']:.4f} HM={hm:.4f}")
        if hm > best_hm:
            best_hm, best_epoch, stale = hm, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_hm": hm,
                        "seen_pairs": sorted(seen_pairs)}, best_path)
        else:
            stale += 1
        if stale >= cfg.train.patience:
            print(f"[seed {seed}] dev early stop after {stale} epochs without HM improvement")
            break
    return best_epoch, best_hm


def final_fit_and_test(seed, epochs, cfg, full_train, test_seen, test_unseen, image_dir,
                       n_crops, n_diseases, device, out_dir):
    seed_all(seed)
    # Fresh BioCLIP initialization and fresh adapters/head; no dev checkpoint is reused.
    model = make_model(cfg, n_crops, n_diseases, device)
    opt = make_optimizer(model, cfg)
    train_loader = loader_for(full_train, image_dir, cfg, True,
                              cfg.data.train_views, shuffle=True)
    sched = cosine_warmup(opt, max(1, len(train_loader) * epochs),
                          cfg.train.warmup_fraction, cfg.train.min_lr_ratio)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, device, opt, sched, cfg.train.mixed_precision == "bf16",
                           f"seed {seed} final {epoch}/{epochs}")
        print(f"[seed {seed}] final epoch={epoch}/{epochs} loss={loss:.4f}")

    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs,
                "num_crops": n_crops, "num_diseases": n_diseases,
                "config": asdict(cfg)}, os.path.join(seed_dir, "final.pt"))
    seen_loader = loader_for(test_seen, image_dir, cfg, False, batch_size=cfg.train.batch_size * 2)
    unseen_loader = loader_for(test_unseen, image_dir, cfg, False, batch_size=cfg.train.batch_size * 2)
    full_seen_pairs = set(zip(full_train.crop_label.tolist(), full_train.disease_label.tolist()))
    seen_m, seen_scores, seen_gt = eval_split(model, test_seen, seen_loader, device, cfg, n_diseases)
    unseen_pairs = set(zip(test_unseen.crop_label.tolist(), test_unseen.disease_label.tolist()))
    unseen_m, unseen_scores, unseen_gt = eval_split(model, test_unseen, unseen_loader, device, cfg,
                                                    n_diseases, unseen_pairs)
    test_ausuc = ausuc(seen_scores, seen_gt, unseen_scores, unseen_gt, n_crops, n_diseases,
                       full_seen_pairs)
    result = {"seed": seed, "selected_epochs": epochs, "test_seen": seen_m,
              "test_unseen": unseen_m, "test_hm_top1": harmonic_mean(seen_m["top1"], unseen_m["top1"]),
              "test_hm_top5": harmonic_mean(seen_m["top5"], unseen_m["top5"]), "ausuc": test_ausuc}
    with open(os.path.join(seed_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", help="Overrides the configured three seeds")
    args = parser.parse_args()
    cfg = Config(); seeds = tuple(args.seeds) if args.seeds else cfg.train.seeds
    if len(seeds) != 3:
        raise ValueError(f"The protocol requires exactly 3 seeds, got {len(seeds)}")
    device = get_device()
    if cfg.train.mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 training, but this CUDA device does not support BF16.")
    dc = cfg.data
    full_train = read_split(os.path.join(dc.root, dc.train_csv))
    test_seen = read_split(os.path.join(dc.root, dc.test_seen_csv))
    test_unseen = read_split(os.path.join(dc.root, dc.test_unseen_csv))
    dev_train, val_seen, val_unseen, val_unseen_pairs = compositional_validation_split(
        full_train, dc.validation_seed, dc.val_unseen_compositions, dc.val_fraction)
    print(f"[split] train={len(dev_train)} val_seen={len(val_seen)} val_unseen={len(val_unseen)} "
          f"heldout_pairs={len(val_unseen_pairs)}; full_train={len(full_train)} "
          f"test_seen={len(test_seen)} test_unseen={len(test_unseen)}")
    expected = (6739, 723, 538)
    actual = (len(dev_train), len(val_seen), len(val_unseen))
    if actual != expected:
        raise ValueError(f"Validation split differs from protocol expectation: got {actual}, expected {expected}. "
                         "Check the seed-0 composition-selection convention and source CSV version.")
    all_frames = [full_train, test_seen, test_unseen]
    n_crops = max(int(f.crop_label.max()) for f in all_frames) + 1
    n_diseases = max(int(f.disease_label.max()) for f in all_frames) + 1
    if n_crops != 55 or n_diseases != 175:
        raise ValueError(f"Expected full label space 55 crops x 175 diseases, got {n_crops} x {n_diseases}")
    image_dir = os.path.join(dc.root, dc.image_dir)
    out_dir = cfg.train.output_dir
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for seed in seeds:
        best_epoch, best_hm = fit_dev(seed, cfg, dev_train, val_seen, val_unseen,
                                      image_dir, n_crops, n_diseases, device, out_dir)
        print(f"[seed {seed}] selected epoch={best_epoch} validation HM={best_hm:.4f}; starting fresh final fit")
        records.append(final_fit_and_test(seed, best_epoch, cfg, full_train, test_seen, test_unseen,
                                          image_dir, n_crops, n_diseases, device, out_dir))
    summary = {}
    for key_path in (("test_seen", "top1"), ("test_seen", "top5"), ("test_unseen", "top1"),
                     ("test_unseen", "top5"), ("test_unseen", "crop_acc"),
                     ("test_unseen", "disease_acc"), ("test_unseen", "class_balanced_acc"),
                     ("ausuc", "auc"), ("test_hm_top1",), ("test_hm_top5",)):
        vals = []
        for row in records:
            item = row
            for key in key_path: item = item[key]
            vals.append(float(item))
        label = ".".join(key_path)
        summary[label] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "seeds": vals}
    report = {"split": {"train": len(dev_train), "val_seen": len(val_seen),
                        "val_unseen": len(val_unseen), "full_train": len(full_train)},
              "results": records, "mean_std": summary}
    with open(os.path.join(out_dir, "three_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
