"""Evaluate a trained adapter checkpoint on test_seen / test_unseen.

  python -m adapter.evaluate --checkpoint outputs/multitask_bioclip_adapter/best.pt

Uses the config stored in the checkpoint. Writes <checkpoint dir>/eval/metrics.json (or --out).
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PlantDiseaseDataset, build_transforms, read_split
from .metrics import ausuc, build_pairs, harmonic_mean, split_metrics
from .train import build_model, get_device


@torch.no_grad()
def collect_logits(model, df, image_dir, img_size, batch_size, num_workers, device, amp):
    loader = DataLoader(PlantDiseaseDataset(df, image_dir, build_transforms(img_size, False)),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    model.eval()
    crops, diseases = [], []
    for batch in tqdm(loader, leave=False, desc="eval"):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch["image"].to(device, non_blocking=True))
        crops.append(out["crop"].float().cpu().numpy())
        diseases.append(out["disease"].float().cpu().numpy())
    return np.concatenate(crops), np.concatenate(diseases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dc = cfg["data"]
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    n_crops, n_dis = ckpt["num_crops"], ckpt["num_diseases"]

    train = read_split(os.path.join(dc["root"], dc["train_csv"]))
    seen = read_split(os.path.join(dc["root"], dc["test_seen_csv"]))
    unseen = read_split(os.path.join(dc["root"], dc["test_unseen_csv"]))
    image_dir = os.path.join(dc["root"], dc["image_dir"])

    model = build_model(cfg, n_crops, n_dis).to(device)
    model.load_state_dict(ckpt["model"])

    kw = dict(image_dir=image_dir, img_size=dc["img_size"], batch_size=args.batch_size,
              num_workers=dc["num_workers"], device=device, amp=amp)
    # Generalized protocol over valid pairs: candidates are train (seen) pairs + test_unseen pairs.
    pairs, num_seen = build_pairs(train, unseen)
    u_pairs = pairs[num_seen:]
    s_m, s_sc, s_gt = split_metrics(*collect_logits(model, seen, **kw), seen, pairs)
    u_m, u_sc, u_gt = split_metrics(*collect_logits(model, unseen, **kw), unseen, pairs, u_pairs)
    for name, m in (("test_seen", s_m), ("test_unseen", u_m)):
        if m["n_skipped"]:
            print(f"[warn] {name}: skipped {m['n_skipped']} images whose pair is not a candidate")
    result = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
              "protocol": "generalized_valid_pairs",
              "num_pairs": len(pairs), "num_seen_pairs": num_seen, "num_unseen_pairs": len(u_pairs),
              "test_seen": s_m, "test_unseen": u_m,
              "test_hm_top1": harmonic_mean(s_m["top1"], u_m["top1"]),
              "test_hm_top5": harmonic_mean(s_m["top5"], u_m["top5"]),
              "ausuc": ausuc(s_sc, s_gt, u_sc, u_gt, num_seen)}

    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
    print("saved ->", out)


if __name__ == "__main__":
    main()
