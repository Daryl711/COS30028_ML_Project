"""DPD-Small data: CSV reading, label space, split, Dataset and transforms.

Each CSV has 7 columns, read by position:
  image | crop_label | disease_label | comp_label | crop_name | disease_name | comp_name
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
COLUMNS = ["image", "crop_label", "disease_label", "comp_label",
           "crop_name", "disease_name", "comp_name"]
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _is_int(x) -> bool:
    try:
        int(str(x).strip())
        return True
    except ValueError:
        return False


def read_split(csv_path: str) -> pd.DataFrame:
    """Read one metadata CSV, with or without a header row."""
    df = pd.read_csv(csv_path)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{csv_path}: expected 7 columns, found {df.shape[1]}")
    # No header row -> pandas used the first data row as column names.
    if str(df.columns[0]).lower().endswith(_IMG_EXT) or _is_int(df.columns[1]):
        df = pd.read_csv(csv_path, header=None)
    df.columns = COLUMNS
    df["image"] = df["image"].astype(str)
    df["crop_label"] = df["crop_label"].astype(int)
    df["disease_label"] = df["disease_label"].astype(int)
    df["comp_label"] = pd.to_numeric(df["comp_label"], errors="coerce")
    return df.reset_index(drop=True)


def _pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["crop_label"].tolist(), df["disease_label"].tolist()))


@dataclass
class LabelSpace:
    """Crop/disease vocab plus the candidate crop-disease compositions.

    Pairs are indexed seen-first, so pair index < num_seen means a seen composition.
    """
    num_crops: int
    num_diseases: int
    seen_pairs: list[tuple[int, int]]
    unseen_pairs: list[tuple[int, int]]
    crop_names: dict[int, str]
    disease_names: dict[int, str]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return self.seen_pairs + self.unseen_pairs

    @property
    def num_seen(self) -> int:
        return len(self.seen_pairs)

    def pair_index(self) -> dict[tuple[int, int], int]:
        return {p: i for i, p in enumerate(self.pairs)}

    def pair_name(self, i: int) -> str:
        c, d = self.pairs[i]
        return f"{self.crop_names.get(c, c)} | {self.disease_names.get(d, d)}"


def build_label_space(train_df, seen_df, unseen_df) -> LabelSpace:
    all_df = pd.concat([train_df, seen_df, unseen_df], ignore_index=True)
    seen_pairs = sorted(_pairs(train_df))
    unseen_pairs = sorted(_pairs(unseen_df) - set(seen_pairs))
    return LabelSpace(
        num_crops=int(all_df["crop_label"].max()) + 1,
        num_diseases=int(all_df["disease_label"].max()) + 1,
        seen_pairs=seen_pairs,
        unseen_pairs=unseen_pairs,
        crop_names=dict(zip(all_df["crop_label"].tolist(), all_df["crop_name"].astype(str))),
        disease_names=dict(zip(all_df["disease_label"].tolist(), all_df["disease_name"].astype(str))),
    )


def audit(train_df, seen_df, unseen_df, ls: LabelSpace) -> None:
    """Print sanity checks about the split. Read this output before trusting any results."""
    print(f"[data] train={len(train_df)} test_seen={len(seen_df)} test_unseen={len(unseen_df)}")
    print(f"[data] crops={ls.num_crops} diseases={ls.num_diseases} "
          f"seen pairs={ls.num_seen} unseen pairs={len(ls.unseen_pairs)}")
    extra = _pairs(seen_df) - set(ls.seen_pairs)
    if extra:
        print(f"[warn] {len(extra)} test_seen pairs never appear in train; those images are skipped in eval")
    overlap = _pairs(unseen_df) & set(ls.seen_pairs)
    if overlap:
        print(f"[warn] {len(overlap)} test_unseen pairs also appear in train (not truly unseen)")
    tc, td = set(train_df["crop_label"]), set(train_df["disease_label"])
    new_c = set(unseen_df["crop_label"]) - tc
    new_d = set(unseen_df["disease_label"]) - td
    if new_c or new_d:
        print(f"[warn] unseen test has crops {sorted(new_c)} / diseases {sorted(new_d)} never seen in "
              "training; a two-head CNN cannot predict these")
    else:
        print("[data] every unseen crop and disease appears in train -> compositional zero-shot setting")
    counts = train_df.groupby(["crop_label", "disease_label"]).size()
    print(f"[data] images per seen pair: min={counts.min()} median={int(counts.median())} max={counts.max()}")


def split_train_val(df: pd.DataFrame, val_fraction: float, seed: int):
    """Hold out val_fraction of each composition, always keeping >=1 image per pair in train."""
    if val_fraction <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for _, g in df.groupby(["crop_label", "disease_label"]):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = min(int(len(idx) * val_fraction), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
    is_val = df.index.isin(val_idx)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class PlantDiseaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.paths = [os.path.join(image_dir, p) for p in df["image"]]
        self.crops = df["crop_label"].to_numpy()
        self.diseases = df["disease_label"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "crop": int(self.crops[i]),
                "disease": int(self.diseases[i]), "index": i}


# ---------------------------------------------------------------- transforms
# Train-time augmentation targets lab-vs-field variation (lighting, viewpoint, occlusion, scale).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(int(img_size / 0.875)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
