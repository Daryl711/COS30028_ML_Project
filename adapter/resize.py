import os
from multiprocessing import Pool
from PIL import Image

SRC = os.path.join("dataset", "DPD_Small_dataset")
DST = os.path.join("dataset", "DPD_Small_256")
SHORT = 256  # short side in pixels

def process(rel):
    src, dst = os.path.join(SRC, rel), os.path.join(DST, rel)
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        im.draft("RGB", (SHORT * 2, SHORT * 2))  # fast JPEG downscale on decode
        im = im.convert("RGB")
        w, h = im.size
        scale = SHORT / min(w, h)
        if scale < 1:  # never upscale
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        im.save(dst, quality=95)

if __name__ == "__main__":
    files = []
    for root, _, names in os.walk(SRC):
        for n in names:
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
                files.append(os.path.relpath(os.path.join(root, n), SRC))
    print(f"{len(files)} images")
    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(process, files, chunksize=32), 1):
            if i % 500 == 0:
                print(i, "/", len(files))
    print("done")