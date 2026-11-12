"""DPD-Small CSV data, compositional validation split, transforms, and fixed views."""
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

COLUMNS = ["image", "crop_label", "disease_label", "comp_label", "crop_name", "disease_name", "comp_name"]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def read_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{csv_path}: expected 7 columns, found {df.shape[1]}")
    first = str(df.columns[0]).lower()
    if first.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
        df = pd.read_csv(csv_path, header=None)
    df.columns = COLUMNS
    df["image"] = df["image"].astype(str)
    for col in ("crop_label", "disease_label"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    df["crop_name"] = df["crop_name"].astype(str)
    df["disease_name"] = df["disease_name"].astype(str)
    # Set before any validation filtering/resetting so an image keeps its ID across subsets.
    df["_image_id"] = np.arange(len(df), dtype=np.int64)
    return df.reset_index(drop=True)



def compositional_validation_split(
    df: pd.DataFrame,
    seed: int = 0,
    n_unseen: int = 10,
    seen_fraction: float = 0.1
):
    """Reproduce the new pseudo-unseen compositional split."""

    # Count images belonging to each composition
    n = df.groupby("comp_label").size()

    # Number of different crops per disease
    crops_of = df.groupby("disease_label")["crop_label"].nunique()

    # Number of different diseases per crop
    dis_of = df.groupby("crop_label")["disease_label"].nunique()

    # Get one representative row per composition
    first = df.drop_duplicates("comp_label").set_index("comp_label")

    # Identify eligible compositions
    cand = [
        c for c in n.index
        if 15 <= n[c] <= 150
        and str(first.loc[c, "disease_name"]).lower() != "healthy"
        and crops_of[first.loc[c, "disease_label"]] >= 2
        and dis_of[first.loc[c, "crop_label"]] >= 2
    ]

    if len(cand) < n_unseen:
        raise ValueError(
            f"Not enough eligible compositions: {len(cand)}"
        )

    # Select pseudo-unseen compositions
    rng = np.random.default_rng(seed)
    val_unseen_comps = rng.choice(
        cand, n_unseen, replace=False
    ).tolist()

    # Remaining data after removing unseen compositions
    rest = df[~df["comp_label"].isin(val_unseen_comps)]

    # Select validation-seen samples independently per composition
    vs = [
        i
        for _, group in rest.groupby("comp_label")
        if len(group) > 1
        for i in group.sample(
            frac=seen_fraction,
            random_state=seed
        ).index
    ]

    idx_vs = np.array(sorted(vs))
    idx_vu = np.where(
        df["comp_label"].isin(val_unseen_comps)
    )[0]

    idx_tr = np.array(
        sorted(set(rest.index) - set(vs))
    )

    # Construct final splits
    train = df.loc[idx_tr].reset_index(drop=True)
    val_seen = df.loc[idx_vs].reset_index(drop=True)
    val_unseen = df.loc[idx_vu].reset_index(drop=True)

    # Verify that both components of every unseen pair remain represented
    train_crops = set(train["crop_label"])
    train_diseases = set(train["disease_label"])

    for comp in val_unseen_comps:
        crop = first.loc[comp, "crop_label"]
        disease = first.loc[comp, "disease_label"]

        if crop not in train_crops:
            raise AssertionError(
                f"Unseen crop {crop} is missing from training"
            )

        if disease not in train_diseases:
            raise AssertionError(
                f"Unseen disease {disease} is missing from training"
            )

    print(
        f"{len(cand)} eligible compositions | "
        f"train {len(train)} | "
        f"val_seen {len(val_seen)} | "
        f"val_unseen {len(val_unseen)}"
    )

    print("Selected unseen compositions:", val_unseen_comps)

    return train, val_seen, val_unseen, set(val_unseen_comps)



def get_augmentation_seed(base_seed: int, epoch_independent_image_id: int, view_id: int) -> int:
    """Return the same augmentation seed for an image/view on every epoch and run."""
    if not 0 <= view_id < 100:
        raise ValueError("view_id must be in [0, 99] for the image_id * 100 seed scheme")
    return int(base_seed + epoch_independent_image_id * 100 + view_id)


class PlantDiseaseDataset(Dataset):
    """CSV rows with fixed clean/augmented views generated deterministically on demand."""
    def __init__(self, frame: pd.DataFrame, image_dir: str, transform=None,
                 train_transform=None, views: int = 1, base_seed: int = 0):
        self.paths = [os.path.join(image_dir, p) for p in frame["image"]]
        self.crops = frame["crop_label"].to_numpy(dtype=np.int64)
        self.diseases = frame["disease_label"].to_numpy(dtype=np.int64)
        self.image_ids = (frame["_image_id"].to_numpy(dtype=np.int64)
                          if "_image_id" in frame else np.arange(len(frame), dtype=np.int64))
        self.transform, self.train_transform = transform, train_transform
        self.views, self.base_seed = int(views), int(base_seed)
        if not 1 <= self.views <= 100:
            raise ValueError("views must be in [1, 100] with this deterministic seed scheme")

    def __len__(self):
        return len(self.paths) * self.views

    def __getitem__(self, index):
        image_index, view_index = divmod(index, self.views)
        with Image.open(self.paths[image_index]) as im:
            image = im.convert("RGB")
        transform = self.transform if view_index == 0 or self.train_transform is None else self.train_transform
        image_id = int(self.image_ids[image_index])
        if transform is not None and view_index > 0 and self.train_transform is not None:
            seed = get_augmentation_seed(self.base_seed, image_id, view_index)
            py_state = random.getstate()
            np_state = np.random.get_state()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                image = transform(image)
            random.setstate(py_state)
            np.random.set_state(np_state)
        elif transform is not None:
            image = transform(image)
        return {"image": image, "crop": int(self.crops[image_index]),
                "disease": int(self.diseases[image_index]), "index": image_id,
                "view": view_index}


def build_transforms(image_size: int, train: bool):
    if train:
        return T.Compose([T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                          T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.2),
                          T.TrivialAugmentWide(), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
                          T.RandomErasing(p=0.25)])
    return T.Compose([T.Resize(int(image_size / 0.875)), T.CenterCrop(image_size), T.ToTensor(),
                      T.Normalize(CLIP_MEAN, CLIP_STD)])
