"""Evaluate an R5 checkpoint on test_seen / test_unseen, plus the crop-leakage probe. Repo root:

  python -m stage3.s3_r5.evaluate --checkpoint outputs/s3_r5/best.pt

Generalized valid-pair protocol (training_scheme.md §5), same code as the Stage 2 notebooks.
Writes <checkpoint dir>/eval/metrics.json (or --out).

Leakage probe (PROPOSED_METHOD.md §6.5, guide.md "What to record"): on val images whose disease
occurs on >= 2 crops, a 5-fold logistic-regression probe predicts the crop from
  - f_d = N(v + A_d(v)), the R5 disease feature  -> "leakage.r5_f_d"
  - N(v), the Stage 3 base (2b) disease feature  -> "leakage.base_N_v"
Only val (carved from train) is probed; the test sets are scored once, never used for a choice.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PlantDiseaseDataset, build_transforms, crops_of_disease, read_split, split_train_val
from .metrics import ausuc, build_pairs, harmonic_mean, leakage_probe, split_metrics
from .model import R5Net, encode_images, load_bioclip, logits_from_features
from .train import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dc, m = cfg["data"], cfg["model"]
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    root = dc["root"]
    train_full = read_split(os.path.join(root, dc["train_csv"]))
    seen = read_split(os.path.join(root, dc["test_seen_csv"]))
    unseen = read_split(os.path.join(root, dc["test_unseen_csv"]))
    train_df, val_df = split_train_val(train_full, dc["val_fraction"], cfg["seed"])
    image_dir = os.path.join(root, dc["image_dir"])

    clip, _ = load_bioclip(m["backbone"], device)
    sd = ckpt["model"]
    net = R5Net(sd["Tc"], sd["Td"], float(sd["logit_scale"]), m["head_dropout"],
                m["bottleneck_dim"], m["adapter_dropout"]).to(device)
    net.load_state_dict(sd)
    net.eval()

    def feats(df):
        loader = DataLoader(PlantDiseaseDataset(df, image_dir, build_transforms(dc["img_size"], False)),
                            batch_size=args.batch_size, shuffle=False, num_workers=dc["num_workers"],
                            pin_memory=device.type == "cuda")
        return encode_images(clip, loader, device, amp)

    pairs, num_seen = build_pairs(train_full, unseen)
    u_pairs = pairs[num_seen:]
    s_m, s_sc, s_gt = split_metrics(*logits_from_features(net, feats(seen), device), seen, pairs)
    u_m, u_sc, u_gt = split_metrics(*logits_from_features(net, feats(unseen), device), unseen, pairs, u_pairs)
    for name, mt in (("test_seen", s_m), ("test_unseen", u_m)):
        if mt["n_skipped"]:
            print(f"[warn] {name}: skipped {mt['n_skipped']} images whose pair is not a candidate")

    # crop-leakage probe on val
    C = crops_of_disease(train_df)
    multi = sorted(d for d, cs in C.items() if len(cs) >= 2)
    V_val = feats(val_df)
    with torch.no_grad():
        f_d = net.features(V_val.to(device))["disease"].cpu().numpy()
    yc, yd = val_df.crop_label.to_numpy(), val_df.disease_label.to_numpy()
    leakage = {"num_multi_crop_diseases": len(multi),
               "r5_f_d": leakage_probe(f_d, yc, yd, C, cfg["seed"]),
               "base_N_v": leakage_probe(F.normalize(V_val, dim=-1).numpy(), yc, yd, C, cfg["seed"])}

    result = {"checkpoint": args.checkpoint, "experiment": cfg["experiment"], "epoch": ckpt.get("epoch"),
              "val": ckpt.get("val"), "protocol": "generalized_valid_pairs",
              "num_pairs": len(pairs), "num_seen_pairs": num_seen, "num_unseen_pairs": len(u_pairs),
              "test_seen": s_m, "test_unseen": u_m,
              "test_hm_top1": harmonic_mean(s_m["top1"], u_m["top1"]),
              "test_hm_top5": harmonic_mean(s_m["top5"], u_m["top5"]),
              "ausuc": ausuc(s_sc, s_gt, u_sc, u_gt, num_seen),
              "leakage": leakage}

    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
    print("saved ->", out)


if __name__ == "__main__":
    main()
