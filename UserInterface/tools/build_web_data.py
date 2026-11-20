"""
build_web_data.py - Builds the data files the static web page needs.

Reads the three DPD-Small CSVs and writes, inside UserInterface/web/:
  * data/classes.json  crop names, disease names, the 333 compositions
                       (seen / unseen, training count, gallery thumbnails)
                       and the sample test images
  * thumbs/            small copies of up to GALLERY_PER_CLASS training
                       images per seen composition (the "matching training
                       images" gallery)
  * examples/          a few seen and unseen test images for one-click demos

None of this depends on a model, so it can be built before any model exists.

Usage (from the repo root):
    python UserInterface/tools/build_web_data.py
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "dataset"
IMAGES = DATASET / "DPD_Small_dataset"
WEB = REPO / "UserInterface" / "web"

GALLERY_PER_CLASS = 6   # training thumbnails kept per seen composition
THUMB_SIZE = 220        # longest side of a gallery thumbnail, in pixels
EXAMPLE_SIZE = 512      # longest side of a sample image
N_EXAMPLES = 4          # sample images per split (seen and unseen)

# CSV columns (no header row):
# 0 file, 1 crop id, 2 disease id, 3 composition id, 4 crop name, 5 disease name, 6 composition name


def read_csv(name):
    with open(DATASET / name, encoding="utf-8", newline="") as f:
        return [r for r in csv.reader(f) if r]


def pretty(name):
    """'pepper_bell' -> 'Pepper bell'"""
    s = name.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def spread(items, k):
    """Pick k items spread evenly across a list (deterministic)."""
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def save_resized(src, dst, size):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(dst, "JPEG", quality=80, optimize=True)


def main():
    train = read_csv("DPD_Small_train.csv")
    test_seen = read_csv("DPD_Small_test_seen.csv")
    test_unseen = read_csv("DPD_Small_test_unseen.csv")

    crops, diseases, comps = {}, {}, {}
    for r in train + test_seen + test_unseen:
        c, d, k = int(r[1]), int(r[2]), int(r[3])
        crops[c], diseases[d] = r[4], r[5]
        comps[k] = (c, d, r[6])

    n_crops, n_dis, n_comp = max(crops) + 1, max(diseases) + 1, max(comps) + 1
    assert len(crops) == n_crops and len(diseases) == n_dis and len(comps) == n_comp, "label ids are not contiguous"

    train_files = defaultdict(list)
    for r in train:
        train_files[int(r[3])].append(r[0])

    # Fresh output folders, so removed classes or settings leave no stale files
    for sub in ("thumbs", "examples"):
        shutil.rmtree(WEB / sub, ignore_errors=True)
        (WEB / sub).mkdir(parents=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)

    compositions = []
    n_thumbs = 0
    for k in range(n_comp):
        c, d, name = comps[k]
        files = sorted(train_files.get(k, []))
        thumbs = []
        for fn in spread(files, GALLERY_PER_CLASS):
            out = f"thumbs/{Path(fn).stem}.jpg"
            save_resized(IMAGES / fn, WEB / out, THUMB_SIZE)
            thumbs.append(out)
        n_thumbs += len(thumbs)
        compositions.append({
            "id": k,
            "crop": c,
            "disease": d,
            "key": name,
            "seen": bool(files),        # unseen compositions have no training images
            "n_train": len(files),
            "thumbs": thumbs,
        })

    # Sample images: one per composition, from different compositions
    examples = []
    for split, rows in (("seen", test_seen), ("unseen", test_unseen)):
        by_comp = defaultdict(list)
        for r in rows:
            by_comp[int(r[3])].append(r[0])
        for k in spread(sorted(by_comp), N_EXAMPLES):
            fn = sorted(by_comp[k])[0]
            out = f"examples/{Path(fn).stem}.jpg"
            save_resized(IMAGES / fn, WEB / out, EXAMPLE_SIZE)
            examples.append({"file": out, "truth": k, "split": split})

    data = {
        "crops": [pretty(crops[i]) for i in range(n_crops)],
        "diseases": [pretty(diseases[i]) for i in range(n_dis)],
        "compositions": compositions,
        "examples": examples,
    }
    (WEB / "data" / "classes.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    n_unseen = sum(not c["seen"] for c in compositions)
    print(f"{n_crops} crops, {n_dis} diseases, {n_comp} compositions ({n_comp - n_unseen} seen, {n_unseen} unseen)")
    print(f"{n_thumbs} gallery thumbnails, {len(examples)} sample images written to {WEB}")


if __name__ == "__main__":
    main()
