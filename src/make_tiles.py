"""Phase 2a - Multi-crop tiling for the tiny healthy class.

TRAIN-split healthy photos only (never val/test): each big photo (3000-6000 px)
is cut into overlapping square tiles covering different stem/canopy regions -
genuinely different pixel regions, not synthetic copies. 28 photos -> ~170
extra real samples. Tiles inherit their photo's plant group, so no leakage.
"""
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "work" / "clean"
SPLIT = ROOT / "work" / "splits" / "split.csv"
OUT = ROOT / "work" / "tiles" / "Healthy"

GRID = (3, 2)  # cols x rows of overlapping tiles + 1 center tile


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SPLIT, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["label"] == "Healthy" and r["split"] == "train"]

    manifest = []
    for r in rows:
        img = Image.open(CLEAN / r["file"])
        w, h = img.size
        side = int(min(w, h) * 0.55)  # tile side: 55% of short edge -> overlap
        cols, rows_n = GRID
        xs = [int(i * (w - side) / max(cols - 1, 1)) for i in range(cols)]
        ys = [int(j * (h - side) / max(rows_n - 1, 1)) for j in range(rows_n)]
        boxes = [(x, y) for y in ys for x in xs]
        boxes.append(((w - side) // 2, (h - side) // 2))  # center
        stem = Path(r["file"]).stem
        for k, (x, y) in enumerate(boxes):
            name = f"{stem}_tile{k}.jpg"
            img.crop((x, y, x + side, y + side)).save(OUT / name, "JPEG", quality=95)
            manifest.append({"file": f"Healthy/{name}", "label": "Healthy",
                             "group": r["group"], "split": "train"})

    with open(ROOT / "work" / "tiles" / "tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "group", "split"])
        w.writeheader()
        w.writerows(manifest)
    print(f"{len(rows)} train healthy photos -> {len(manifest)} tiles in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
