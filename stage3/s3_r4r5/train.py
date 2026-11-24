"""Train Stage 3 run R4 + R5 (learnable prompts + separate adapters). Repo root:

  python -m stage3.s3_r4r5.train                                       # -> outputs/s3_r4r5/
  python -m stage3.s3_r4r5.train --epochs 1 --experiment s3_r4r5_smoke  # quick check
  python -m stage3.s3_r4r5.train --lr_prompt 1e-4 --experiment s3_r4r5_lrp1e-4   # token-LR sub-row

This loop also trains All (R4 + R5 + R6): stage3/s3_all/train.py calls `main()` with its own config, and
an `r6:` section in the config switches on the discriminator and L_inv.

training_scheme.md §1-§5 throughout; same data, split, augmentation, schedule, model selection and evaluator
as R4 / R5 (helpers reused from stage3/s3_r5/train.py). Per step:
  T = prompts through the frozen text encoder (R4), f_c, f_d = adapters (R5)
  loss = crop CE + disease CE + lambda_kg * L_kg  [+ L_inv through a GRL with the DANN ramp, if R6]
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
from stage3.s3_r5r6.model import crop_mask_from_train, dann_lambda

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv, default_config):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    p.add_argument("--experiment")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lambda_kg", type=float)
    p.add_argument("--lr_prompt", type=float, help="prompt-token LR only (sweep sub-row; default = lr_new)")
    p.add_argument("--lambda_max", type=float, help="R6 only")
    p.add_argument("--unconditional", action="store_true", help="R6 only: plain-DANN ablation")
    return p.parse_args(argv)


def build_net(cfg, clip, tokenizer, data, device):
    m = cfg["model"]
    kw = dict(head_dropout=m["head_dropout"], n_ctx=m["n_ctx"], attr_dropout=m["attr_dropout"],
              bottleneck_dim=m["bottleneck_dim"], adapter_dropout=m["adapter_dropout"])
    scale = clip.logit_scale.exp().item()
    if cfg.get("r6"):
        from stage3.s3_all.model import AllNet
        mask = crop_mask_from_train(data["train"], len(data["diseases"]), len(data["crops"]))
        return AllNet(clip, tokenizer, data["prompts"], data["crops"], data["diseases"], scale, device, mask,
                      disc_hidden=m["disc_hidden"], conditional=bool(cfg["r6"]["conditional"]), **kw)
    from .model import R4R5Net
    return R4R5Net(clip, tokenizer, data["prompts"], data["crops"], data["diseases"], scale, device, **kw)


def main(argv=None, default_config=os.path.join(HERE, "config.yaml")):
    args = parse_args(argv, default_config)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    r6 = cfg.get("r6")
    if (args.lambda_max is not None or args.unconditional) and not r6:
        raise SystemExit("--lambda_max / --unconditional need a config with an r6 section (stage3.s3_all)")
    if args.experiment:
        cfg["experiment"] = args.experiment
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lambda_kg is not None:
        cfg["r4"]["lambda_kg"] = args.lambda_kg
    if args.lr_prompt is not None:
        cfg["train"]["lr_prompt"] = args.lr_prompt
    if args.lambda_max is not None:
        r6["lambda_max"] = args.lambda_max
    if args.unconditional:
        r6["conditional"] = False
    tc, dc, m = cfg["train"], cfg["data"], cfg["model"]
    lam_kg = float(cfg["r4"]["lambda_kg"])
    lr_prompt = tc.get("lr_prompt")
    lr_prompt = None if lr_prompt is None else float(lr_prompt)

    seed_everything(cfg["seed"])
    device = get_device()
    amp = bool(tc["amp"]) and device.type == "cuda"
    if device.type != "cuda":
        print("[warn] no CUDA device: training on CPU will be very slow")

    data = load_data(cfg)
    train_df, val_df = data["train"], data["val"]
    print(f"[data] train={len(train_df)} val={len(val_df)} test_seen={len(data['seen'])} "
          f"test_unseen={len(data['unseen'])}")

    clip, tokenizer = load_bioclip(m["backbone"], device)
    net = build_net(cfg, clip, tokenizer, data, device)
    pl = net.prompts
    print(f"[model] {'All (R4 + R5 + R6)' if r6 else 'R4 + R5'} | logit scale {net.logit_scale.item():.2f} | "
          f"trainable params {sum(p.numel() for p in net.parameters() if p.requires_grad):,} "
          f"(context 2 x {m['n_ctx']}, name tokens {pl.crop_prompts.num_name_tokens} crop + "
          f"{pl.disease_prompts.num_name_tokens} disease, adapters 2 x 512-{m['bottleneck_dim']}-512)")
    print(f"[r4] lambda_kg {lam_kg} | lr tokens {tc['lr_new'] if lr_prompt is None else lr_prompt:g}, "
          f"other modules {tc['lr_new']:g}")
    if r6:
        n_multi = int(net.multi_crop.sum())
        frac = float(net.multi_crop[torch.as_tensor(train_df.disease_label.to_numpy().copy(), device=device)]
                     .float().mean())
        print(f"[r6] {'conditional' if r6['conditional'] else 'UNCONDITIONAL'} | lambda_max {r6['lambda_max']} | "
              f"{n_multi} multi-crop diseases | {frac:.1%} of train images get L_inv"
              f"{'' if r6['conditional'] else ' (unconditional: all images)'}")
        assert n_multi == 42, n_multi

    # Epoch-0 check: the prompt path with zero deltas must reproduce the 2b / R5 text classifier.
    for key, t0 in (("crop", pl.T0c), ("disease", pl.T0d)):
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

    optimizer = torch.optim.AdamW(net.param_groups(tc["lr_new"], tc["weight_decay"], lr_prompt))
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    spe = len(train_loader)
    total_steps = tc["epochs"] * spe
    scheduler = cosine_with_warmup(optimizer, tc["warmup_epochs"] * spe, total_steps, tc["min_lr_ratio"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    out_dir = os.path.join(cfg["output_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Epoch 0: token deltas are zero and adapters are the identity, so this equals 2b's epoch 0.
    v0 = val_stats(net.frozen(), V_val, val_df, device)
    history = [{"epoch": 0, **v0}]
    print(f"epoch  0 (zero-shot = 2b epoch 0) | val crop {v0['val_crop_acc']:.4f} "
          f"dis {v0['val_disease_acc']:.4f} score {v0['val_score']:.4f}")

    def ckpt(epoch, va, head):
        # "model" = the R5Net classifier this epoch produces (what evaluate.py scores);
        # "net" = every trainable part (prompt deltas, adapters, discriminator if R6).
        return {"model": head.state_dict(), "net": net.state_dict(), "epoch": epoch, "val": va,
                "config": cfg, "crops": data["crops"], "diseases": data["diseases"]}

    ls = tc["label_smoothing"]
    best_score, best_epoch, step, lam = -1.0, -1, 0, 0.0
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        net.train()
        tl = tkg = dcrop = ddis = n = ca = da = til = inv_n = inv_c = 0
        for b in tqdm(train_loader, leave=False, desc=f"{cfg['experiment']} ep{epoch}"):
            x = b["image"].to(device, non_blocking=True)
            yc, yd = b["crop"].to(device), b["disease"].to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                v = clip.encode_image(x)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                T = net.text()                              # 230 prompts through the frozen text encoder
            if r6:
                lam = dann_lambda(step / total_steps, r6["lambda_max"])
                o = net.forward_train(v.float(), T, yc, yd, lam)    # trainable image side in fp32, as in R5
            else:
                o = net(v.float(), T)
            lc, ld = o["crop"].float(), o["disease"].float()
            cls_loss = F.cross_entropy(lc, yc, label_smoothing=ls) + F.cross_entropy(ld, yd, label_smoothing=ls)
            drift = net.drift(T)
            kg_loss = drift["crop"] + drift["disease"]
            loss = cls_loss + lam_kg * kg_loss + (o["inv_loss"] if r6 else 0.0)
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
            tkg += kg_loss.item() * bs
            dcrop += drift["crop"].item() * bs
            ddis += drift["disease"].item() * bs
            n += bs
            ca += (lc.argmax(1) == yc).sum().item()
            da += (ld.argmax(1) == yd).sum().item()
            if r6:
                til += o["inv_loss"].item() * o["inv_n"]
                inv_n += o["inv_n"]
                inv_c += o["inv_correct"]

        head = net.frozen()
        va = val_stats(head, V_val, val_df, device)
        row = {"epoch": epoch, "train_loss": tl / n, "train_kg_loss": tkg / n,
               "drift_crop": dcrop / n, "drift_disease": ddis / n,
               "train_crop_acc": ca / n, "train_disease_acc": da / n}
        if r6:
            row.update({"train_inv_loss": til / max(1, inv_n), "train_disc_acc": inv_c / max(1, inv_n),
                        "lambda": lam})
        history.append({**row, **va, "lr": optimizer.param_groups[-1]["lr"],
                        "lr_prompt": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        torch.save(ckpt(epoch, va, head), os.path.join(out_dir, "last.pt"))
        improved = va["val_score"] > best_score
        if improved:
            best_score, best_epoch = va["val_score"], epoch
            torch.save(ckpt(epoch, va, head), os.path.join(out_dir, "best.pt"))
        r6_s = (f" | inv {til / max(1, inv_n):.4f} disc acc {inv_c / max(1, inv_n):.3f} lam {lam:.3f}"
                if r6 else "")
        print(f"epoch {epoch:2d} | train cls {tl / n:.4f} crop {ca / n:.3f} dis {da / n:.3f} | "
              f"L_kg {tkg / n:.4f} (crop {dcrop / n:.4f} dis {ddis / n:.4f}){r6_s} | "
              f"val crop {va['val_crop_acc']:.4f} dis {va['val_disease_acc']:.4f} score {va['val_score']:.4f}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
        if tc["patience"] and epoch - best_epoch >= tc["patience"]:
            print(f"[train] early stop: no val improvement for {tc['patience']} epochs")
            break

    pkg = "s3_all" if r6 else "s3_r4r5"
    print(f"[train] best val score {best_score:.4f} at epoch {best_epoch} -> {out_dir}/best.pt")
    print(f"[next] python -m stage3.{pkg}.evaluate --checkpoint {out_dir}/best.pt")


if __name__ == "__main__":
    main()
