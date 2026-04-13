from pathlib import Path
from PIL import Image

#!/usr/bin/env python3

TARGET_W, TARGET_H = 480, 288

def resize_image(src: Path, dst: Path, size=(TARGET_W, TARGET_H)):
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    with Image.open(src) as im:
        im = im.convert("RGBA") if im.mode in ("LA", "P") else im.convert("RGB")
        im = im.resize(size, resample=Image.LANCZOS)
        im.save(dst, format="PNG")

if __name__ == "__main__":
    base = Path(__file__).parent
    resize_image(base / "left.jpg",  base / "left.png")
    resize_image(base / "right.jpg", base / "right.png")
    print("Saved left.png and right.png")