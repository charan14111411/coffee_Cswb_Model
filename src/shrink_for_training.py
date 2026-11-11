"""Resize work/clean and work/tiles images IN PLACE to short side 768.

Run on Colab after clean/split/tiles, before training: the model trains at
384x384, so this loses nothing while making data loading ~10x faster.
(Do NOT run locally - the local work/clean keeps full resolution.)
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHORT_SIDE = 768

n = 0
for d in [ROOT / "work" / "clean", ROOT / "work" / "tiles"]:
    for p in sorted(d.rglob("*.jpg")):
        img = Image.open(p).convert("RGB")
        s = SHORT_SIDE / min(img.size)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            img.save(p, "JPEG", quality=90)
            n += 1
print(f"resized {n} images to short side {SHORT_SIDE}")
