"""Evaluate a saved adapter checkpoint on test_seen / test_unseen.
python -m adapter.evaluate --ckpt outputs/bioclip_adapter/seed_0/dev_best.pt

"""
import argparse, json, os
import torch
from .config import Config
from .dataset import read_split
from .metrics import ausuc, harmonic_mean
from .train import eval_split, get_device, loader_for, make_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
args = p.parse_args()

cfg, device, dc = Config(), get_device(), Config().data
train = read_split(os.path.join(dc.root, dc.train_csv))
seen = read_split(os.path.join(dc.root, dc.test_seen_csv))
unseen = read_split(os.path.join(dc.root, dc.test_unseen_csv))
n_crops, n_dis = 55, 175
image_dir = os.path.join(dc.root, dc.image_dir)

model = make_model(cfg, n_crops, n_dis, device)
model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])

bs = cfg.train.batch_size * 2
s_m, s_sc, s_gt = eval_split(model, seen, loader_for(seen, image_dir, cfg, False, batch_size=bs), device, cfg, n_dis)
u_pairs = set(zip(unseen.crop_label, unseen.disease_label))
u_m, u_sc, u_gt = eval_split(model, unseen, loader_for(unseen, image_dir, cfg, False, batch_size=bs),
                             device, cfg, n_dis, u_pairs)
result = {"checkpoint": args.ckpt, "test_seen": s_m, "test_unseen": u_m,
          "test_hm_top1": harmonic_mean(s_m["top1"], u_m["top1"]),
          "test_hm_top5": harmonic_mean(s_m["top5"], u_m["top5"]),
          "ausuc": ausuc(s_sc, s_gt, u_sc, u_gt, n_crops, n_dis,
                         set(zip(train.crop_label, train.disease_label)))}
out = os.path.join(os.path.dirname(args.ckpt), "test_metrics.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=float)
print(json.dumps(result, indent=2, default=float)); print("saved ->", out)
