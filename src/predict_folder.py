"""Score any folder of photos with the trained model - no labels needed.

This mirrors what the production API will do (Phase 6): a quality gate first,
then the model, then the frozen thresholds. Use it on new photos to see what
the app would tell a farmer.

  python src/predict_folder.py test_new
  python src/predict_folder.py test_new --pdf        # with heatmaps
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "models" / "v1" / "config.json").read_text())
T_LOW, T_HIGH, IMG = CFG["T_low"], CFG["T_high"], CFG["img"]

# Quality gate (same numbers as cleaning and the planned API)
BLUR_MIN, CLIPPED_MAX, DARK_MAX = 60.0, 0.20, 0.40
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def quality(pil):
    arr = np.asarray(pil)
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    s = 1024 / max(g.shape[1], 1)
    if s < 1:
        g = cv2.resize(g, (1024, max(int(g.shape[0] * s), 1)))
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    clipped = float((g > 250).mean())
    dark = float((g < 15).mean())
    if blur < BLUR_MIN:
        return blur, clipped, dark, "photo is blurry - hold steady and retake"
    if clipped > CLIPPED_MAX:
        return blur, clipped, dark, "too much sun - shade the plant with your body and retake"
    if dark > DARK_MAX:
        return blur, clipped, dark, "too dark - step back or use the flash"
    return blur, clipped, dark, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF with heatmaps")
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    folder = (ROOT / args.folder) if not Path(args.folder).is_absolute() else Path(args.folder)
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        print(f"no images found in {folder}")
        return
    print(f"{len(paths)} images in {folder}")
    print(f"thresholds: no signs < {T_LOW} | inconclusive | >= {T_HIGH} WSB likely\n")

    import make_report_pdf as M  # reuse the model loader and Grad-CAM
    model = M.load_model()

    rows, cache = [], {}
    for p in paths:
        try:
            pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        except Exception as e:
            print(f"  {p.name}: cannot open ({e})")
            continue
        blur, clipped, dark, reject = quality(pil)
        if reject:
            score, verdict, cam = None, "INCONCLUSIVE - " + reject, None
        else:
            score, cam = M.score_and_cam(model, pil)
            verdict = M.verdict_of(score)
        rows.append({"file": str(p.relative_to(folder)), "score": "" if score is None else round(score, 4),
                     "verdict": verdict, "blur": round(blur, 1),
                     "clipped": round(clipped, 4), "dark": round(dark, 4),
                     "size": f"{pil.width}x{pil.height}"})
        if cam is not None:
            cache[str(p.relative_to(folder))] = (np.array(M.disp_tf(pil)), cam, score, verdict)
        s = "gate" if score is None else f"{score:.3f}"
        print(f"  {p.name:42s} {s:>7s}  {verdict}")

    out_csv = ROOT / "reports" / f"predictions_{folder.name}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    tally = Counter(r["verdict"].split(" - ")[0] for r in rows)
    print(f"\n{dict(tally)}")
    print(f"-> {out_csv}")

    if args.pdf and cache:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        out_pdf = ROOT / "reports" / f"predictions_{folder.name}.pdf"
        with PdfPages(out_pdf) as pdf:
            for name, (disp, cam, score, verdict) in cache.items():
                col = ("#b3261e" if verdict.startswith("WSB") else
                       "#2e7d32" if verdict.startswith("No") else "#a05a00")
                fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
                ax[0].imshow(disp); ax[0].set_title("photo as the model saw it", fontsize=10)
                ax[1].imshow(disp); ax[1].imshow(cam, cmap="jet", alpha=0.45)
                ax[1].set_title("what drove the decision (red = most)", fontsize=10)
                for a in ax:
                    a.axis("off")
                fig.suptitle(f"{name}    score {score:.3f}  ->  {verdict}", color=col, fontsize=12)
                pdf.savefig(fig, dpi=args.dpi); plt.close(fig)
        print(f"-> {out_pdf}  ({out_pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
