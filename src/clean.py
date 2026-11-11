"""Phase 1a - Clean Data_final into work/clean.

Per image: apply EXIF rotation, strip ALL metadata (GPS!), save plain JPEG q95
at full resolution, and record quality scores (blur / clipped / dark), SHA-256
and a perceptual dhash. Then report exact and near duplicates.

Data_final/ is read-only; everything is written to work/clean/.
"""
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Data_final"
DST = ROOT / "work" / "clean"
CLASSES = {"Healthy": "Healthy", "Unhealthy(Infected)": "Unhealthy"}

# Quality flags (tuned for big phone photos)
BLUR_MIN = 60.0        # Laplacian variance below this => blurry
CLIPPED_MAX = 0.20     # >20% near-white pixels => burnt out (sun)
DARK_MAX = 0.40        # >40% near-black pixels => too dark
NEAR_DUP_HAMMING = 6   # dhash distance <= this => near duplicate


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    rows = []
    for src_cls, dst_cls in CLASSES.items():
        out_dir = DST / dst_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((SRC / src_cls).glob("*.jpg"))
        print(f"{src_cls}: {len(files)} images")
        for i, p in enumerate(files):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            out = out_dir / p.name
            img.save(out, "JPEG", quality=95, exif=b"")  # strips GPS/EXIF

            arr = np.asarray(img)
            g_full = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            scale = 1024 / max(g_full.shape[1], 1)
            g = cv2.resize(g_full, (1024, max(int(g_full.shape[0] * scale), 1)))
            blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
            clipped = float((g > 250).mean())
            dark = float((g < 15).mean())

            flags = []
            if blur < BLUR_MIN:
                flags.append("blurry")
            if clipped > CLIPPED_MAX:
                flags.append("clipped")
            if dark > DARK_MAX:
                flags.append("dark")

            rows.append({
                "file": f"{dst_cls}/{p.name}",
                "label": dst_cls,
                "width": img.width,
                "height": img.height,
                "blur": round(blur, 2),
                "clipped": round(clipped, 4),
                "dark": round(dark, 4),
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "dhash": f"{dhash(g):016x}",
                "flags": "|".join(flags),
            })
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}")

    with open(DST / "images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Duplicate report
    dupes = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["sha256"] == b["sha256"]:
                dupes.append((a["file"], b["file"], "exact", 0))
            else:
                d = hamming(int(a["dhash"], 16), int(b["dhash"], 16))
                if d <= NEAR_DUP_HAMMING:
                    dupes.append((a["file"], b["file"], "near", d))
    with open(DST / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "file_b", "kind", "hamming"])
        w.writerows(dupes)

    n_flag = sum(1 for r in rows if r["flags"])
    cross = [d for d in dupes if d[0].split("/")[0] != d[1].split("/")[0]]
    print(f"\nDone: {len(rows)} images cleaned -> {DST}")
    print(f"Flagged (excluded from training): {n_flag}")
    for r in rows:
        if r["flags"]:
            print(f"  {r['file']}: {r['flags']} (blur={r['blur']}, clipped={r['clipped']}, dark={r['dark']})")
    print(f"Duplicate pairs: {len(dupes)} ({len(cross)} CROSS-CLASS - label errors, must review)")
    for d in dupes:
        print(f"  {d[2]} d={d[3]}: {d[0]} <-> {d[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
