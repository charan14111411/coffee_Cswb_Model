"""STEP 4 - cut every photo into overlapping pieces.

Why this is the whole point:

    whole photo 3456 px -> 384 px  = 9.0x shrink -> a 100 px bore hole = 11 px
    one piece   1728 px -> 384 px  = 4.5x shrink -> the same hole      = 22 px

The network's finest view is a 12 x 12 grid. On a whole photo one grid cell
covers 288 px of the original, so a 100 px hole is a THIRD of one cell - it can
never be isolated. On a piece, one cell covers 144 px, so the hole fills a cell.

Note: if the photos came from cwsb_package.zip they are already 768 px on the
short side, so a piece is 384 px and needs no shrink at all. The detail that
reaches the network is identical either way.

Pieces are cut ONCE and reused by every fold.
Reads work/clean. Writes only version_rbn/work/pieces/.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rbn_common as C

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CLEAN = ROOT / "work" / "clean"
FOLDS = HERE / "work" / "folds.csv"
OUT = HERE / "work" / "pieces"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    if not FOLDS.exists():
        print(f"missing {FOLDS} - run make_folds.py first")
        return 1

    with open(FOLDS, newline="") as f:
        rows = list(csv.DictReader(f))
    photos = {}
    for r in rows:
        photos.setdefault(r["file"], {"label": r["label"], "group": r["group"]})
    todo = sorted(photos)
    if args.limit:
        todo = todo[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    man, per, sizes = [], [], Counter()

    for i, fname in enumerate(todo, 1):
        src = CLEAN / fname
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        side, boxes, pcs = C.pieces_of(img)
        sizes[side] += 1
        stem = Path(fname).stem
        meta = photos[fname]
        dest = OUT / meta["label"]
        dest.mkdir(parents=True, exist_ok=True)

        for k, (p, (x, y)) in enumerate(zip(pcs, boxes)):
            if p.size != (C.IMG, C.IMG):
                p = p.resize((C.IMG, C.IMG), Image.LANCZOS)
            name = f"{stem}__p{k:02d}.jpg"
            p.save(dest / name, "JPEG", quality=args.quality)
            man.append({"file": f"{meta['label']}/{name}", "photo": fname,
                        "label": meta["label"], "group": meta["group"],
                        "piece": k, "x": x, "y": y, "side": side})
        per.append(len(boxes))
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} photos -> {len(man)} pieces")

    with open(OUT.parent / "pieces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "photo", "label", "group",
                                          "piece", "x", "y", "side"])
        w.writeheader()
        w.writerows(man)

    byc = Counter(m["label"] for m in man)
    print(f"\n{len(per)} photos -> {len(man)} pieces "
          f"(avg {sum(per) / max(len(per), 1):.1f} per photo, "
          f"min {min(per)}, max {max(per)})")
    for k, v in byc.items():
        print(f"  {k:10s} {v:6d} pieces")
    print(f"  piece sides seen: {dict(sizes.most_common(4))}")
    print(f"-> {OUT.parent / 'pieces.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
