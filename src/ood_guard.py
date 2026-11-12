"""Out-of-distribution guard: does this photo look like the training data?

The classifier must answer "healthy" or "infected" for ANY image - it cannot
say "this is not my kind of photo". That is how 19 close-up branch photos all
came back as WSB-likely at 0.99.

This guard works on the model's own 768-number description of an image
(the layer just before the final decision). For a new photo we measure how
similar it is to its nearest neighbours among the TRAINING photos. Photos
taken the way the training photos were taken land close to them; a different
distance, subject or camera lands far away.

  python src/ood_guard.py --fit                  build the reference set
  python src/ood_guard.py --check test_new       score a folder
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_report_pdf as M  # noqa: E402
import train as T  # noqa: E402

REF = ROOT / "models" / "v1" / "ood_reference.npz"
K = 5  # average similarity to this many nearest training photos
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@torch.no_grad()
def features(model, paths, bs=8):
    """768-number description per image, L2-normalised.

    Images are loaded one batch at a time: these photos are up to 6000x4000,
    so holding them all in memory at once exhausts RAM.
    """
    out = []
    for i in range(0, len(paths), bs):
        xs = []
        for p in paths[i:i + bs]:
            with Image.open(p) as im:
                pil = ImageOps.exif_transpose(im).convert("RGB")
            xs.append(M.eval_tf(pil))
            del pil
        f = model.forward_features(torch.stack(xs))
        f = model.forward_head(f, pre_logits=True)   # [N, 768]
        out.append(torch.nn.functional.normalize(f, dim=1).numpy())
        if (i + bs) % 80 == 0:
            print(f"  {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


def similarity(feats, ref):
    """Mean cosine similarity to the K nearest training photos (1 = identical)."""
    sims = feats @ ref.T                      # both are normalised
    k = min(K, ref.shape[0])
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def fit(model):
    rows = T.load_rows("train", with_tiles=False)
    paths = [ROOT / "work" / "clean" / r["file"] for r in rows]
    print(f"describing {len(paths)} training photos...")
    ref = features(model, paths)

    # calibrate on validation + test, which ARE the right kind of photo
    ind_rows = T.load_rows("val") + T.load_rows("test")
    print(f"describing {len(ind_rows)} held-out photos...")
    ind = features(model, [ROOT / "work" / "clean" / r["file"] for r in ind_rows])
    s = similarity(ind, ref)
    cut = float(np.quantile(s, 0.05))          # 95% of good photos sit above this
    np.savez(REF, ref=ref, cut=cut)
    print(f"in-distribution similarity: min {s.min():.3f}  median {np.median(s):.3f}  max {s.max():.3f}")
    print(f"cut-off (5th percentile)  : {cut:.3f}")
    print(f"-> {REF}")


def check(model, folder):
    d = np.load(REF)
    ref, cut = d["ref"], float(d["cut"])
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        print(f"no images in {folder}")
        return
    feats = features(model, paths)
    sims = similarity(feats, ref)
    print(f"cut-off {cut:.3f}  ({len(paths)} images in {folder})\n")
    rows = []
    for p, s in zip(paths, sims):
        ok = s >= cut
        print(f"  {p.name:28s} similarity {s:.3f}  "
              f"{'assessable' if ok else 'OUT OF DISTRIBUTION - cannot assess'}")
        rows.append({"file": p.name, "similarity": round(float(s), 4),
                     "assessable": bool(ok)})
    n_bad = sum(not r["assessable"] for r in rows)
    print(f"\n{n_bad} of {len(rows)} would be refused with "
          f"'photo not suitable - retake as instructed'")
    out = ROOT / "reports" / f"ood_{Path(folder).name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--check", metavar="FOLDER")
    args = ap.parse_args()
    model = M.load_model()
    if args.fit:
        fit(model)
    if args.check:
        check(model, args.check)


if __name__ == "__main__":
    main()
