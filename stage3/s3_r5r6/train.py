"""Train Stage 3 run R5 + R6. Run from the repo root:

  python -m stage3.s3_r5r6.train
  python -m stage3.s3_r5r6.train --epochs 1 --experiment s3_r5r6_smoke          # quick check
  python -m stage3.s3_r5r6.train --lambda_max 0.3 --experiment s3_r5r6_lam0.3   # sweep run
  python -m stage3.s3_r5r6.train --unconditional --experiment s3_r5r6_uncond    # ablation sub-row

Identical to stage3/s3_r5/train.py (same data, split, schedule, selection, early stopping) except:
  loss = crop CE + disease CE + L_inv, where L_inv reaches A_d through a GRL whose strength follows the
  DANN ramp over training progress p = step / total_steps.
Model selection is unchanged: best val_score (mean of val crop and disease top-1).
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

from stage3.s3_r5.dataset import PlantDiseaseDataset, build_transforms
from stage3.s3_r5.model import encode_images, encode_text, load_bioclip
from stage3.s3_r5.train import cosine_with_warmup, get_device, load_data, seed_everything, val_stats

from .model import R5R6Net, crop_mask_from_train, dann_lambda

HERE = os.path.dirname(os.path.abspath(__file__))


def build_net(cfg, clip, tokenizer, prompts, train_df, device, amp) -> R5R6Net:
    m, r6 = cfg["model"], cfg["r6"]
    text_c = encode_text(clip, tokenizer, prompts["crop"], device, amp)
    text_d = encode_text(clip, tokenizer, prompts["disease"], device, amp)
    mask = crop_mask_from_train(train_df, text_d.shape[0], text_c.shape[0])
    return R5R6Net(text_c, text_d, clip.logit_scale.exp().item(), mask, m["head_dropout"],
                   m["bottleneck_dim"], m["adapter_dropout"], m["disc_hidden"],
                   bool(r6["conditional"])).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    p.add_argument("--experiment")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lambda_max", type=float)
    p.add_argument("--unconditional", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.experiment:
        cfg["experiment"] = args.experiment
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lambda_max is not None:
        cfg["r6"]["lambda_max"] = args.lambda_max
    if args.unconditional:
        cfg["r6"]["conditional"] = False
    tc, dc, r6 = cfg["train"], cfg["data"], cfg["r6"]

    seed_everything(cfg["seed"])
    device = get_device()
    amp = bool(tc["amp"]) and device.type == "cuda"
    if device.type != "cuda":
        print("[warn] no CUDA device: training on CPU will be very slow")

    data = load_data(cfg)
    train_df, val_df = data["train"], data["val"]
    print(f"[data] train={len(train_df)} val={len(val_df)} test_seen={len(data['seen'])} "
          f"test_unseen={len(data['unseen'])}")

    clip, tokenizer = load_bioclip(cfg["model"]["backbone"], device)
    net = build_net(cfg, clip, tokenizer, data["prompts"], train_df, device, amp)
    n_multi = int(net.multi_crop.sum())
    frac = float(net.multi_crop[torch.as_tensor(train_df.disease_label.to_numpy().copy(), device=device)].float().mean())
    print(f"[model] logit scale {net.logit_scale.item():.2f} | trainable params "
          f"{sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print(f"[r6] {'conditional' if r6['conditional'] else 'UNCONDITIONAL'} | lambda_max {r6['lambda_max']} | "
          f"{n_multi} multi-crop diseases | {frac:.1%} of train images get L_inv"
          f"{'' if r6['conditional'] else ' (unconditional: all images)'}")
    assert n_multi == 42, n_multi

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
    total_steps = tc["epochs"] * spe
    scheduler = cosine_with_warmup(optimizer, tc["warmup_epochs"] * spe, total_steps, tc["min_lr_ratio"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    out_dir = os.path.join(cfg["output_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Epoch 0: adapters are the identity, so this equals 2b's epoch 0 and R5's epoch 0.
    v0 = val_stats(net, V_val, val_df, device)
    history = [{"epoch": 0, **v0}]
    print(f"epoch  0 (zero-shot = 2b epoch 0) | val crop {v0['val_crop_acc']:.4f} "
          f"dis {v0['val_disease_acc']:.4f} score {v0['val_score']:.4f}")

    def ckpt(epoch, va):
        return {"model": net.state_dict(), "epoch": epoch, "val": va, "config": cfg,
                "crops": data["crops"], "diseases": data["diseases"]}

    ls = tc["label_smoothing"]
    best_score, best_epoch, step = -1.0, -1, 0
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        net.train()
        tl = til = n = ca = da = inv_n = inv_c = 0
        for b in tqdm(train_loader, leave=False, desc=f"{cfg['experiment']} ep{epoch}"):
            x = b["image"].to(device, non_blocking=True)
            yc, yd = b["crop"].to(device), b["disease"].to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                v = clip.encode_image(x)
            lam = dann_lambda(step / total_steps, r6["lambda_max"])
            o = net.forward_train(v.float(), yc, yd, lam)       # trainable part in fp32, as in R5
            lc, ld = o["crop"].float(), o["disease"].float()
            cls_loss = F.cross_entropy(lc, yc, label_smoothing=ls) + F.cross_entropy(ld, yd, label_smoothing=ls)
            loss = cls_loss + o["inv_loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            bs = len(yc)
            tl += cls_loss.item() * bs
            til += o["inv_loss"].item() * o["inv_n"]
            n += bs
            inv_n += o["inv_n"]
            inv_c += o["inv_correct"]
            ca += (lc.argmax(1) == yc).sum().item()
            da += (ld.argmax(1) == yd).sum().item()

        va = val_stats(net, V_val, val_df, device)
        history.append({"epoch": epoch, "train_loss": tl / n, "train_crop_acc": ca / n, "train_disease_acc": da / n,
                        "train_inv_loss": til / max(1, inv_n), "train_disc_acc": inv_c / max(1, inv_n),
                        "lambda": lam, **va, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        torch.save(ckpt(epoch, va), os.path.join(out_dir, "last.pt"))
        improved = va["val_score"] > best_score
        if improved:
            best_score, best_epoch = va["val_score"], epoch
            torch.save(ckpt(epoch, va), os.path.join(out_dir, "best.pt"))
        print(f"epoch {epoch:2d} | train cls {tl / n:.4f} crop {ca / n:.3f} dis {da / n:.3f} | "
              f"inv {til / max(1, inv_n):.4f} disc acc {inv_c / max(1, inv_n):.3f} lam {lam:.3f} | "
              f"val crop {va['val_crop_acc']:.4f} dis {va['val_disease_acc']:.4f} score {va['val_score']:.4f}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
        if tc["patience"] and epoch - best_epoch >= tc["patience"]:
            print(f"[train] early stop: no val improvement for {tc['patience']} epochs")
            break

    print(f"[train] best val score {best_score:.4f} at epoch {best_epoch} -> {out_dir}/best.pt")
    print(f"[next] python -m stage3.s3_r5r6.evaluate --checkpoint {out_dir}/best.pt")


if __name__ == "__main__":
    main()
