"""Zip everything Colab needs into one file: cwsb_package.zip.

Images are resized so the SHORT side is 768 px (the model trains at 384:
no practical information loss, ~20x smaller upload). Full-resolution
originals stay in work/clean locally. CSVs and train.py ride along as-is.

Upload the zip to the top level of Google Drive, then run
colab/CWSB_train_colab.ipynb.
"""
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cwsb_package.zip"
SHORT_SIDE = 768

DIRS = [ROOT / "work" / "clean", ROOT / "work" / "splits", ROOT / "work" / "tiles"]
FILES = [ROOT / "src" / "train.py"]


def shrink(p: Path) -> bytes:
    img = Image.open(p).convert("RGB")
    s = SHORT_SIDE / min(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in DIRS:
            for p in sorted(q for q in d.rglob("*") if q.is_file()):
                rel = str(p.relative_to(ROOT))
                if p.suffix.lower() == ".jpg":
                    z.writestr(rel, shrink(p))
                    n += 1
                    if n % 100 == 0:
                        print(f"  {n} images...")
                else:
                    z.write(p, rel)
        for p in FILES:
            z.write(p, p.relative_to(ROOT))
    print(f"-> {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB, {n} images at short side {SHORT_SIDE})")


if __name__ == "__main__":
    main()
