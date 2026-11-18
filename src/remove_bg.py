"""Label-free background removal for stem close-ups.

No annotations, no extra model download: OpenCV GrabCut seeded with a centre
rectangle, which works because the capture protocol puts the stem in the
middle of the frame.

THE RULE: whatever this does to an image must happen in BOTH training and
serving. The model is trained on the output of `apply()`, so the API must
call the same `apply()` before predicting. Masking only at prediction time
makes things worse - measured, see reports/mask_experiment_images_16_11.csv.

It is deliberately conservative: when GrabCut collapses (finds nothing) or
keeps nearly everything, the mask is rejected and the original image is
returned unchanged, with `ok=False`. Whole-plant canopy photos fail this way
about 80% of the time - there is no figure/ground to separate - so this is
for stem close-ups only.

  python src/remove_bg.py images_16_11 --check          # CAN I use removal on this data?
  python src/remove_bg.py images_16_11 --out work/nobg/images_16_11
  python src/remove_bg.py images_16_11 --out work/nobg/x --fill blur --preview
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

WORK = 480          # GrabCut runs at this long side, then the mask is upscaled
MARGIN = 0.12       # seed rectangle inset from each edge
ITERS = 5
KEEP_MIN, KEEP_MAX = 0.08, 0.85   # outside this the mask is not trusted
FEATHER = 7         # soft edge, avoids a hard cut-out look


def mask_for(rgb):
    """Foreground mask (float 0..1, image size). Uses no labels."""
    h, w = rgb.shape[:2]
    s = WORK / max(h, w)
    small = cv2.resize(rgb, (max(int(w * s), 1), max(int(h * s), 1)))
    sh, sw = small.shape[:2]
    rect = (int(sw * MARGIN), int(sh * MARGIN),
            int(sw * (1 - 2 * MARGIN)), int(sh * (1 - 2 * MARGIN)))

    m = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), m, rect,
                    bgd, fgd, ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)

    # keep only the largest blob: drops specks of stray foreground
    n, lab, stats, _ = cv2.connectedComponentsWithStats((fg > 0.5).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == biggest).astype(np.float32)

    fg = cv2.GaussianBlur(fg, (FEATHER, FEATHER), 0)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)


def apply(pil, fill="black"):
    """(PIL image, ok, kept_fraction). ok=False -> original returned unchanged.

    fill: 'black' | 'grey' | 'blur'  - what replaces the background.
          'blur' keeps the scene's colours but destroys its detail, which is
          gentler than a hard cut-out and closer to what the camera sees.
    """
    rgb = np.asarray(pil.convert("RGB"))
    m = mask_for(rgb)
    if m is None:
        return pil, False, 0.0
    kept = float((m > 0.5).mean())
    if not (KEEP_MIN < kept < KEEP_MAX):
        return pil, False, kept            # untrustworthy mask: change nothing

    if fill == "blur":
        k = max(int(min(rgb.shape[:2]) * 0.05) | 1, 9)
        back = cv2.GaussianBlur(rgb, (k, k), 0)
    elif fill == "grey":
        back = np.full_like(rgb, 128)
    else:
        back = np.zeros_like(rgb)

    out = rgb * m[..., None] + back * (1.0 - m[..., None])
    return Image.fromarray(out.astype(np.uint8)), True, kept


GOOD, RISKY = 0.90, 0.75     # share of photos that must process cleanly


def check(paths, sample=40):
    """Decide whether a dataset may be trained with REMOVE_BACKGROUND = True.

    Answers the only question that matters before a retrain: will this run
    produce a fully processed dataset, or a half-processed one?
    """
    random.seed(42)
    picked = random.sample(paths, min(sample, len(paths)))
    ok_n, keeps = 0, []
    for p in picked:
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        pil.thumbnail((900, 900))
        _, ok, kept = apply(pil)
        ok_n += bool(ok)
        if ok:
            keeps.append(kept)
    rate = ok_n / len(picked)

    print(f"checked {len(picked)} of {len(paths)} photos")
    print(f"background could be removed from {ok_n} ({rate:.0%})")
    if keeps:
        print(f"stem area kept: median {np.median(keeps):.0%}")
    print()
    if rate >= GOOD:
        print("VERDICT: YES - you can train this data with REMOVE_BACKGROUND = True.")
    elif rate >= RISKY:
        print(f"VERDICT: RISKY - {1 - rate:.0%} of photos would pass through untouched,")
        print("         giving a half-processed dataset. Either train with")
        print("         REMOVE_BACKGROUND = False, or remove the photos that fail.")
    else:
        print(f"VERDICT: NO - keep REMOVE_BACKGROUND = False. Only {rate:.0%} of these")
        print("         photos have a separable stem. These look like whole-plant")
        print("         canopy shots, where the background is more coffee plant and")
        print("         there is nothing to cut away.")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--check", action="store_true",
                    help="only report whether this data can be trained with removal on")
    ap.add_argument("--out", help="where the processed copies go (not needed with --check)")
    ap.add_argument("--fill", choices=["black", "grey", "blur"], default="black")
    ap.add_argument("--preview", action="store_true",
                    help="also write a before/after contact sheet PDF")
    args = ap.parse_args()

    src = Path(args.folder)
    src = src if src.is_absolute() else ROOT / src
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        print(f"no images found in {src}")
        return

    if args.check:
        print(f"{len(paths)} images in {src}\n")
        check(paths)
        return

    if not args.out:
        ap.error("--out is required unless you pass --check")
    dst = Path(args.out)
    dst = dst if dst.is_absolute() else ROOT / dst
    dst.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} images in {src}  ->  {dst}   (fill: {args.fill})")

    done, skipped, keeps, shots = 0, [], [], []
    for p in paths:
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        out, ok, kept = apply(pil, args.fill)
        out.save(dst / (p.stem + ".jpg"), quality=95)
        if ok:
            done += 1
            keeps.append(kept)
            if args.preview and len(shots) < 24:
                shots.append((p.name, np.asarray(pil.copy().resize((320, 320))),
                              np.asarray(out.copy().resize((320, 320))), kept))
        else:
            skipped.append((p.name, kept))

    print(f"\nbackground removed : {done}/{len(paths)}")
    print(f"left unchanged     : {len(skipped)}/{len(paths)}  (mask not trustworthy)")
    if keeps:
        print(f"kept area          : median {np.median(keeps):.0%}  "
              f"range {min(keeps):.0%}-{max(keeps):.0%}")
    for name, kept in skipped[:10]:
        print(f"  unchanged: {name}  (grabcut kept {kept:.0%})")

    if args.preview and shots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = dst / "_preview.pdf"
        with PdfPages(pdf_path) as pdf:
            for i in range(0, len(shots), 3):
                chunk = shots[i:i + 3]
                fig, axes = plt.subplots(len(chunk), 2, figsize=(7, 3.4 * len(chunk)),
                                         squeeze=False)
                for r, (name, before, after, kept) in enumerate(chunk):
                    axes[r][0].imshow(before); axes[r][0].set_title(name, fontsize=8)
                    axes[r][1].imshow(after)
                    axes[r][1].set_title(f"background removed (kept {kept:.0%})", fontsize=8)
                    for a in axes[r]:
                        a.axis("off")
                fig.tight_layout(); pdf.savefig(fig, dpi=85); plt.close(fig)
        print(f"-> {pdf_path}")


if __name__ == "__main__":
    main()
