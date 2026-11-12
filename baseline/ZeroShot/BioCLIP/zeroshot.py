"""Zero-shot BioCLIP on DPD-Small test_seen and test_unseen. No training.

Each class and each crop-disease pair is turned into text prompts ("a photo of a tomato leaf with
early blight."). An image is given the class whose text embedding is closest to its image embedding.
Because the text can describe any pair, unseen pairs are scored the same way as seen ones.

  python baseline/ZeroShot/BioCLIP/zeroshot.py
  python baseline/ZeroShot/BioCLIP/zeroshot.py --model hf-hub:imageomics/bioclip-2 --out outputs/zeroshot_bioclip2/eval
  # plain OpenAI CLIP, to see how much the biology training helps (OpenAI weights need the quickgelu variant):
  python baseline/ZeroShot/BioCLIP/zeroshot.py --model ViT-B-16-quickgelu --pretrained openai --out outputs/zeroshot_clip/eval

Writes metrics.json (same keys as the baselines' evaluate.py), predictions.csv, prompts.json and
calibration_curve.png.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PlantDiseaseDataset, audit, build_label_space, read_split
from metrics import calibration_sweep, composition_scores, harmonic_mean, topk_accuracy

# Prompt templates. Each class embedding is the average over its templates.
PAIR = ["a photo of {a_crop} leaf with {disease}.",
        "a photo of {a_crop} plant with {disease}.",
        "a close-up photo of {disease} on {a_crop} leaf."]
HEALTHY_PAIR = ["a photo of a healthy {crop} leaf.",
                "a photo of a healthy {crop} plant.",
                "a close-up photo of a healthy {crop} leaf."]
CROP = ["a photo of {a_crop} leaf.",
        "a photo of {a_crop} plant.",
        "a close-up photo of {a_crop} leaf."]
DISEASE = ["a photo of a leaf with {disease}.",
           "a photo of a plant with {disease}.",
           "a close-up photo of {disease} on a leaf."]
HEALTHY_DISEASE = ["a photo of a healthy leaf.",
                   "a photo of a healthy plant.",
                   "a close-up photo of a healthy leaf."]

# Dataset names that read badly as text: shorthand and typos. Everything else just loses its underscores.
CROP_ALIASES = {"paddy": "rice", "pepper_bell": "bell pepper", "chilli": "chili pepper"}
DISEASE_ALIASES = {"crinckle": "crinkle", "segatoka": "sigatoka", "xamthomonas": "xanthomonas",
                   "verticulium_wilt": "verticillium wilt", "monillia_laxa": "monilinia laxa",
                   "aphis_spp": "aphids", "leafblast": "leaf blast", "erosomyia_sp": "erosomyia"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hf-hub:imageomics/bioclip", help="open_clip model name")
    p.add_argument("--pretrained", default=None, help="open_clip weights tag (not needed for hf-hub models)")
    p.add_argument("--root", default="dataset")
    p.add_argument("--image_dir", default="DPD_Small_dataset")
    p.add_argument("--train_csv", default="DPD_Small_train.csv")
    p.add_argument("--test_seen_csv", default="DPD_Small_test_seen.csv")
    p.add_argument("--test_unseen_csv", default="DPD_Small_test_unseen.csv")
    p.add_argument("--out", default="outputs/zeroshot_bioclip/eval")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# Progress bars only in a terminal (not in notebooks or captured logs).
SHOW_PROGRESS = sys.stderr.isatty()


def text_name(name: str, aliases: dict) -> str:
    return aliases.get(name, name.replace("_", " "))


def with_article(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def build_prompts(ls):
    """Prompt lists for every crop id, disease id and candidate pair (in label-space order)."""
    crop = {c: text_name(n, CROP_ALIASES) for c, n in ls.crop_names.items()}
    disease = {d: text_name(n, DISEASE_ALIASES) for d, n in ls.disease_names.items()}
    healthy = {d for d, n in ls.disease_names.items() if n == "healthy"}

    def pair_prompts(c, d):
        templates = HEALTHY_PAIR if d in healthy else PAIR
        return [t.format(crop=crop[c], a_crop=with_article(crop[c]), disease=disease[d]) for t in templates]

    return {
        "crop": {c: [t.format(a_crop=with_article(crop[c])) for t in CROP] for c in crop},
        "disease": {d: HEALTHY_DISEASE if d in healthy else [t.format(disease=disease[d]) for t in DISEASE]
                    for d in disease},
        "pair": [pair_prompts(c, d) for c, d in ls.pairs],
    }


@torch.no_grad()
def encode_texts(model, tokenizer, prompt_lists, device):
    """One unit-length embedding per class: the normalised mean of its prompts' embeddings."""
    out = []
    for prompts in tqdm(prompt_lists, desc="text", leave=False, disable=not SHOW_PROGRESS):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            e = model.encode_text(tokenizer(prompts).to(device)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        out.append(e / e.norm())
    return torch.stack(out)


def class_matrix(model, tokenizer, prompts_by_id, n, device):
    """Text embeddings indexed by label id (0..n-1). Ids with no name get a zero row and are masked out."""
    ids = sorted(prompts_by_id)
    emb = encode_texts(model, tokenizer, [prompts_by_id[i] for i in ids], device)
    full = torch.zeros(n, emb.shape[1], device=device)
    full[ids] = emb
    valid = torch.zeros(n, dtype=torch.bool, device=device)
    valid[ids] = True
    return full, valid


@torch.no_grad()
def encode_images(model, loader, device):
    feats = []
    for batch in tqdm(loader, desc="images", leave=False, disable=not SHOW_PROGRESS):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.encode_image(x).float()
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats)


def gt_pair_indices(df, ls):
    idx = ls.pair_index()
    return np.array([idx.get(p, -1) for p in zip(df["crop_label"].tolist(), df["disease_label"].tolist())])


def save_calibration_plot(cal, out_dir, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(cal["seen_acc"], cal["unseen_acc"], lw=1.5)
    ax.scatter([cal["seen_at_best"]], [cal["unseen_at_best"]], color="red", zorder=3,
               label=f"best HM {cal['best_hm']:.3f}")
    ax.set_xlabel("seen top-1 accuracy")
    ax.set_ylabel("unseen top-1 accuracy")
    ax.set_title(f"{title}: seen-unseen trade-off (AUC {cal['auc']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = read_split(os.path.join(args.root, args.train_csv))
    seen_df = read_split(os.path.join(args.root, args.test_seen_csv))
    unseen_df = read_split(os.path.join(args.root, args.test_unseen_csv))
    # Same label space as the trained baselines: seen pairs come from train, so the numbers compare.
    ls = build_label_space(train_df, seen_df, unseen_df)
    audit(train_df, seen_df, unseen_df, ls)

    print(f"[model] {args.model} ({args.pretrained or 'default weights'}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    scale = model.logit_scale.exp().item()

    prompts = build_prompts(ls)
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump({"crop": {ls.crop_names[c]: p for c, p in prompts["crop"].items()},
                   "disease": {ls.disease_names[d]: p for d, p in prompts["disease"].items()},
                   "pair": {ls.pair_name(i): p for i, p in enumerate(prompts["pair"])}}, f, indent=2)

    crop_txt, crop_ok = class_matrix(model, tokenizer, prompts["crop"], ls.num_crops, device)
    dis_txt, dis_ok = class_matrix(model, tokenizer, prompts["disease"], ls.num_diseases, device)
    pair_txt = encode_texts(model, tokenizer, prompts["pair"], device)

    image_dir = os.path.join(args.root, args.image_dir)
    results, scores_by_split, gt_by_split, pred_rows = {}, {}, {}, []
    for split, df in (("seen", seen_df), ("unseen", unseen_df)):
        gt = gt_pair_indices(df, ls)
        keep = gt >= 0
        df, gt = df[keep].reset_index(drop=True), gt[keep]
        loader = DataLoader(PlantDiseaseDataset(df, image_dir, preprocess), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)
        img = encode_images(model, loader, device)

        crop_logits = (scale * img @ crop_txt.T).masked_fill(~crop_ok, -1e4).cpu().numpy()
        dis_logits = (scale * img @ dis_txt.T).masked_fill(~dis_ok, -1e4).cpu().numpy()
        # Main score: one prompt per crop-disease pair, e.g. "a photo of a tomato leaf with early blight."
        scores = (scale * img @ pair_txt.T).cpu().numpy()
        # Alternative: score crop and disease separately and add them, as the two-head baselines do.
        factorized = composition_scores(crop_logits, dis_logits, ls.pairs)
        scores_by_split[split], gt_by_split[split] = scores, gt

        r = {
            "n_images": int(len(df)),
            "crop": topk_accuracy(crop_logits, df["crop_label"].to_numpy()),
            "disease": topk_accuracy(dis_logits, df["disease_label"].to_numpy()),
            # generalized: choose among ALL seen + unseen compositions (main protocol)
            "composition": topk_accuracy(scores, gt),
            "composition_factorized": topk_accuracy(factorized, gt),
        }
        if split == "seen":
            r["composition_closed_world"] = topk_accuracy(scores[:, : ls.num_seen], gt)
        else:
            r["composition_closed_world"] = topk_accuracy(scores[:, ls.num_seen:], gt - ls.num_seen)
        results[split] = r

        top5 = np.argsort(-scores, axis=1)[:, :5]
        for i in range(len(df)):
            pred_rows.append({"split": split, "image": df["image"][i], "true": ls.pair_name(gt[i]),
                              **{f"pred_{k + 1}": ls.pair_name(top5[i, k]) for k in range(5)}})

    results["harmonic_mean"] = {
        k: harmonic_mean(results["seen"]["composition"][k], results["unseen"]["composition"][k])
        for k in ("top1", "top5")
    }
    results["harmonic_mean_factorized"] = {
        k: harmonic_mean(results["seen"]["composition_factorized"][k],
                         results["unseen"]["composition_factorized"][k])
        for k in ("top1", "top5")
    }
    cal = calibration_sweep(scores_by_split["seen"], gt_by_split["seen"],
                            scores_by_split["unseen"], gt_by_split["unseen"], ls.num_seen)
    results["calibrated"] = {k: cal[k] for k in ("best_hm", "best_bias", "seen_at_best", "unseen_at_best", "auc")}
    order = np.argsort(cal["seen_acc"], kind="stable")  # the trade-off curve, for plotting several models together
    results["calibration_curve"] = {"seen_acc": cal["seen_acc"][order].round(5).tolist(),
                                    "unseen_acc": cal["unseen_acc"][order].round(5).tolist()}
    results["model"] = {"name": args.model, "pretrained": args.pretrained, "zero_shot": True}

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    save_calibration_plot(cal, args.out, "zero-shot")

    print("\n                       Top-1    Top-5")
    for split in ("seen", "unseen"):
        for key in ("crop", "disease", "composition", "composition_factorized", "composition_closed_world"):
            r = results[split][key]
            print(f"{split:>6} {key:<25} {r['top1']:.4f}   {r['top5']:.4f}")
    for key in ("harmonic_mean", "harmonic_mean_factorized"):
        hm = results[key]
        print(f"{key:<32} {hm['top1']:.4f}   {hm['top5']:.4f}")
    c = results["calibrated"]
    print(f"calibrated best HM {c['best_hm']:.4f} (seen {c['seen_at_best']:.4f}, unseen "
          f"{c['unseen_at_best']:.4f}, bias {c['best_bias']:.3f}) | AUC {c['auc']:.4f}")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
