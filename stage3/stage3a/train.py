"""Train Stage 3 run R4 (2b + learnable prompts: CoOp context, CSP class tokens, KgCoOp anchor). Repo root:

  python -m stage3.stage3a.train                                        # lambda_kg 8 -> outputs/stage3a/
  python -m stage3.stage3a.train --lambda_kg 0 --experiment stage3a_nokg  # sub-row without L_kg
  python -m stage3.stage3a.train --epochs 1 --experiment stage3a_smoke    # quick check

training_scheme.md §1-§5 throughout, with the same data, split, augmentation, schedule, model selection
and evaluation as R5 (stage3/stage3b/train.py, whose helpers this reuses). The only change from 2b is
where T_c, T_d come from: the frozen text encoder runs every step on the learnable prompts, and
L += lambda_kg * L_kg.
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage3.stage3b.dataset import PlantDiseaseDataset, build_transforms
from stage3.stage3b.model import encode_images, encode_text, load_bioclip
from stage3.stage3b.train import cosine_with_warmup, get_device, load_data, seed_everything, val_stats

from .model import R4Net

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    p.add_argument("--experiment")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lambda_kg", type=float)
    p.add_argument("--lr_new", type=float, help="token LR (sweep sub-row only; default 1e-3, guide.md)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.experiment:
        cfg["experiment"] = args.experiment
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lambda_kg is not None:
        cfg["r4"]["lambda_kg"] = args.lambda_kg
    if args.lr_new is not None:
        cfg["train"]["lr_new"] = args.lr_new
    tc, dc, m = cfg["train"], cfg["data"], cfg["model"]
    lam_kg = float(cfg["r4"]["lambda_kg"])

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

    clip, tokenizer = load_bioclip(m["backbone"], device)
    net = R4Net(clip, tokenizer, data["prompts"], data["crops"], data["diseases"],
                clip.logit_scale.exp().item(), m["head_dropout"], m["n_ctx"], device, m["attr_dropout"])
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[model] logit scale {net.logit_scale.item():.2f} | trainable params {n_train:,} "
          f"(context 2 x {m['n_ctx']} tokens, name tokens {net.crop_prompts.num_name_tokens} crop + "
          f"{net.disease_prompts.num_name_tokens} disease) | lambda_kg {lam_kg} | lr_new {tc['lr_new']:g}")

    # Epoch-0 check: the prompt path with zero deltas must reproduce the 2b / R5 text classifier.
    for key, t0 in (("crop", net.T0c), ("disease", net.T0d)):
        ref = encode_text(clip, tokenizer, data["prompts"][key], device, amp)
        cos = (t0 * ref).sum(-1)
        print(f"[check] {key}: min cosine(prompt path at init, encode_text) = {cos.min().item():.6f}")
        assert cos.min() > 0.999, "prompt learner does not reproduce the taxonomy text embedding at init"

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

    # Epoch 0: every delta is zero, so this equals 2b's (and R5's) epoch 0.
    v0 = val_stats(net.frozen(), V_val, val_df, device)
    history = [{"epoch": 0, **v0}]
    print(f"epoch  0 (zero-shot = 2b epoch 0) | val crop {v0['val_crop_acc']:.4f} "
          f"dis {v0['val_disease_acc']:.4f} score {v0['val_score']:.4f}")

    def ckpt(epoch, va, head):
        # "model" = the classifier this epoch produces (what evaluate.py scores);
        # "prompt_learner" = the learnable tokens that produced it.
        return {"model": head.state_dict(), "prompt_learner": net.state_dict(), "epoch": epoch, "val": va,
                "config": cfg, "crops": data["crops"], "diseases": data["diseases"]}

    ls = tc["label_smoothing"]
    best_score, best_epoch = -1.0, -1
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        net.train()
        tl = tkg = dcrop = ddis = n = ca = da = 0
        for b in tqdm(train_loader, leave=False, desc=f"{cfg['experiment']} ep{epoch}"):
            x = b["image"].to(device, non_blocking=True)
            yc, yd = b["crop"].to(device), b["disease"].to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                v = clip.encode_image(x)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                T = net.text()                              # 230 prompts through the frozen text encoder
            o = net(v.float(), T)                           # heads in fp32, as in 2b
            lc, ld = o["crop"].float(), o["disease"].float()
            cls_loss = F.cross_entropy(lc, yc, label_smoothing=ls) + F.cross_entropy(ld, yd, label_smoothing=ls)
            drift = net.drift(T)
            kg_loss = drift["crop"] + drift["disease"]
            loss = cls_loss + lam_kg * kg_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = len(yc)
            tl += cls_loss.item() * bs
            tkg += kg_loss.item() * bs
            dcrop += drift["crop"].item() * bs
            ddis += drift["disease"].item() * bs
            n += bs
            ca += (lc.argmax(1) == yc).sum().item()
            da += (ld.argmax(1) == yd).sum().item()

        head = net.frozen()
        va = val_stats(head, V_val, val_df, device)
        history.append({"epoch": epoch, "train_loss": tl / n, "train_kg_loss": tkg / n,
                        "drift_crop": dcrop / n, "drift_disease": ddis / n,
                        "train_crop_acc": ca / n, "train_disease_acc": da / n,
                        **va, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        torch.save(ckpt(epoch, va, head), os.path.join(out_dir, "last.pt"))
        improved = va["val_score"] > best_score
        if improved:
            best_score, best_epoch = va["val_score"], epoch
            torch.save(ckpt(epoch, va, head), os.path.join(out_dir, "best.pt"))
        print(f"epoch {epoch:2d} | train cls {tl / n:.4f} crop {ca / n:.3f} dis {da / n:.3f} | "
              f"L_kg {tkg / n:.4f} (crop {dcrop / n:.4f} dis {ddis / n:.4f}) | "
              f"val crop {va['val_crop_acc']:.4f} dis {va['val_disease_acc']:.4f} score {va['val_score']:.4f}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
        if tc["patience"] and epoch - best_epoch >= tc["patience"]:
            print(f"[train] early stop: no val improvement for {tc['patience']} epochs")
            break

    print(f"[train] best val score {best_score:.4f} at epoch {best_epoch} -> {out_dir}/best.pt")
    print(f"[next] python -m stage3.stage3a.evaluate --checkpoint {out_dir}/best.pt")


if __name__ == "__main__":
    main()
