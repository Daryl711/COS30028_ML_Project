"""DPD-Small data and class text for Stage 3 (training_scheme.md §1-§2).

Same CSV reading, split, transforms and taxonomy prompts as the Stage 2 notebook
(stage2/stage2b/basic_text_colab.ipynb), so R5 starts from exactly the 2b setup.

Each CSV has 7 columns, read by position:
  image | crop_label | disease_label | comp_label | crop_name | disease_name | comp_name
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

COLUMNS = ["image", "crop_label", "disease_label", "comp_label",
           "crop_name", "disease_name", "comp_name"]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def read_split(csv_path: str) -> pd.DataFrame:
    """Read one metadata CSV, with or without a header row."""
    df = pd.read_csv(csv_path, header=None, names=COLUMNS, dtype=str)
    if not df.crop_label.iloc[0].strip().isdigit():  # tolerate a header row
        df = df.iloc[1:]
    df = df.reset_index(drop=True)
    for c in ("crop_label", "disease_label"):
        df[c] = df[c].astype(int)
    return df


def split_train_val(df: pd.DataFrame, val_fraction: float, seed: int):
    """Hold out val_fraction of each composition, always keeping >= 1 image per pair in train."""
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for _, g in df.groupby(["crop_label", "disease_label"]):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        val_idx.extend(idx[:min(int(len(idx) * val_fraction), len(idx) - 1)].tolist())
    is_val = df.index.isin(val_idx)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def crops_of_disease(train_df: pd.DataFrame) -> dict[int, set[int]]:
    """C(d): the crops each disease occurs on in the training split."""
    return {int(d): set(g.crop_label.astype(int).tolist()) for d, g in train_df.groupby("disease_label")}


class PlantDiseaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform):
        self.paths = [os.path.join(image_dir, p) for p in df["image"]]
        self.crops = df["crop_label"].to_numpy()
        self.diseases = df["disease_label"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB")
        return {"image": self.transform(img), "crop": int(self.crops[i]), "disease": int(self.diseases[i])}


def build_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(int(img_size / 0.875)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ---------------------------------------------------------------- class text (guide.md run 2b)
def _sp(s: str) -> str:
    return s.replace("_", " ").strip()


def _read_taxonomy(path: str) -> dict[str, str]:
    # cp1252: two rows of the DPD taxonomy files are not valid UTF-8
    with open(path, encoding="cp1252", newline="") as f:
        return {r[0]: _sp(r[1]) for r in csv.reader(f) if r}


def class_names(train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Crop and disease names ordered by label index."""
    crops = train_df.drop_duplicates("crop_label").set_index("crop_label").crop_name.sort_index().tolist()
    diseases = train_df.drop_duplicates("disease_label").set_index("disease_label").disease_name.sort_index().tolist()
    return crops, diseases


def taxonomy_prompts(text_dir: str, crops: list[str], diseases: list[str]) -> dict[str, list[str]]:
    """'a photo of a {crop} leaf, {taxonomy}' / 'a photo of a leaf with {disease}, {taxonomy}'."""
    tax_c = _read_taxonomy(os.path.join(text_dir, "plant_descriptions.csv"))
    tax_d = _read_taxonomy(os.path.join(text_dir, "diseases_descriptions.csv"))
    missing = (set(crops) - set(tax_c)) | (set(diseases) - set(tax_d))
    if missing:
        raise ValueError(f"taxonomy CSVs have no entry for: {sorted(missing)}")
    return {"crop": [f"a photo of a {_sp(c)} leaf, {tax_c[c]}" for c in crops],
            "disease": [f"a photo of a leaf with {_sp(d)}, {tax_d[d]}" for d in diseases]}
