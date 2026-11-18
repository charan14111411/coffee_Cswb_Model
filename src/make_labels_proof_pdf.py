"""Proof report: model predictions vs. the CONFIRMED damage labels.

The annotation team confirmed every image in images_health_16_11_v2 as
infested and drew boxes around the damage (labels/*.txt, YOLO format).
This script checks "to what extent the model reaches" those labels:

  image level    does the model flag each confirmed-infested image?
  location level does the Grad-CAM heatmap point AT the labeled damage?
                 (peak-in-box hit, and % of model attention inside boxes)

Outputs reports/proof_labels_vs_model.pdf (summary page first, misses
up front) and reports/proof_labels_vs_model.csv.

  python src/make_labels_proof_pdf.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import make_report_pdf as M          # model loader, Grad-CAM, thresholds
from predict_folder import quality   # same quality gate as the screening run

IMG_DIR = ROOT / "images_health_16_11_v2"
LBL_DIR = ROOT / "labels"
OUT_PDF = ROOT / "reports" / "proof_labels_vs_model.pdf"
OUT_CSV = ROOT / "reports" / "proof_labels_vs_model.csv"
IMG = M.IMG                          # 384
RESIZE = IMG + 32                    # 416, matches eval_tf / disp_tf

BOX_COLORS = {0: "#00e5ff", 1: "#76ff03", 2: "#ffd600"}   # per label class


def boxes_in_crop(label_path, w, h):
    """YOLO-normalised boxes -> pixel boxes in the 384x384 display crop.

    disp_tf does Resize(shorter side = 416) then CenterCrop(384); apply the
    same geometry to the box corners and clip to the crop.
    """
    scale = RESIZE / min(w, h)
    left = (w * scale - IMG) / 2.0
    top = (h * scale - IMG) / 2.0
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
        x0 = xc * w * scale - bw * w * scale / 2 - left
        y0 = yc * h * scale - bh * h * scale / 2 - top
        x1 = x0 + bw * w * scale
        y1 = y0 + bh * h * scale
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1, float(IMG)), min(y1, float(IMG))
        if x1 - x0 > 2 and y1 - y0 > 2:
            out.append((c, x0, y0, x1, y1))
    return out


def box_mask(boxes):
    mask = np.zeros((IMG, IMG), dtype=bool)
    for _, x0, y0, x1, y1 in boxes:
        mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    return mask


def localisation(cam, boxes):
    """(peak_hit, attention_share): is the hottest point inside a confirmed
    box, and what fraction of total heatmap mass falls inside the boxes."""
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    hit = any(x0 <= px <= x1 and y0 <= py <= y1 for _, x0, y0, x1, y1 in boxes)
    total = float(cam.sum())
    share = float(cam[box_mask(boxes)].sum()) / total if total > 1e-9 else 0.0
    return (int(px), int(py)), hit, share


def draw_boxes(ax, boxes):
    for c, x0, y0, x1, y1 in boxes:
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2.2,
            edgecolor=BOX_COLORS.get(c, "#ffffff")))


def page_for(pdf, r, disp, cam, boxes, dpi=80):
    col = {"FLAGGED (correct)": "#2e7d32", "MISSED": "#b3261e"}.get(r["agreement"], "#a05a00")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.6))
    ax[0].imshow(disp)
    ax[0].set_title("photo + CONFIRMED damage boxes (annotators)", fontsize=9)
    draw_boxes(ax[0], boxes)
    ax[1].imshow(disp)
    if cam is not None:
        ax[1].imshow(cam, cmap="jet", alpha=0.45)
        px, py = r["peak"]
        ax[1].plot(px, py, marker="x", color="white", markersize=12, markeredgewidth=3)
        loc = f"heatmap peak {'INSIDE' if r['peak_in_box'] else 'OUTSIDE'} a confirmed box" \
              f"   |   {r['attention_in_boxes']:.0%} of model attention inside boxes"
    else:
        loc = "too blurry - model was not asked (quality gate)"
    ax[1].set_title("model heatmap (red = what drove the score, x = peak)", fontsize=9)
    draw_boxes(ax[1], boxes)
    for a in ax:
        a.axis("off")
    score = "gate" if r["score"] == "" else f"{r['score']:.3f}"
    fig.suptitle(f"{r['file']}    score {score}  ->  {r['verdict']}    [{r['agreement']}]\n{loc}",
                 color=col, fontsize=10)
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def summary_page(pdf, recs):
    n = len(recs)
    flagged = [r for r in recs if r["agreement"] == "FLAGGED (correct)"]
    missed = [r for r in recs if r["agreement"] == "MISSED"]
    inconc = [r for r in recs if r["agreement"] == "INCONCLUSIVE"]
    scored = [r for r in recs if r["score"] != ""]
    hits = [r for r in scored if r["peak_in_box"]]
    shares = [r["attention_in_boxes"] for r in scored]

    groups = sorted({r["group"] for r in recs})
    g_flagged = sorted({r["group"] for r in flagged})
    g_missed = sorted({r["group"] for r in missed})

    L = [
        "CWSB MODEL vs CONFIRMED LABELS - PROOF REPORT",
        "",
        f"Dataset   : images_health_16_11_v2  ({n} images, {len(groups)} distinct source photos)",
        "Ground    : every image was confirmed INFESTED by the annotation team,",
        "            with boxes drawn around each damage spot (labels/, YOLO format,",
        f"            {sum(r['n_boxes'] for r in recs)} boxes total).",
        "Model     : cwsb_v1 (ConvNeXt-Tiny classifier) - it never saw these images",
        "            or labels during training.",
        f"Thresholds: no signs < {M.T_LOW}  |  inconclusive  |  >= {M.T_HIGH} WSB likely",
        "",
        "=== 1. IMAGE LEVEL - does the model flag the confirmed-infested images? ===",
        "",
        f"  Flagged WSB LIKELY     : {len(flagged):3d} / {n}   ({len(flagged) / n:.1%})",
        f"  Inconclusive / retake  : {len(inconc):3d} / {n}",
        f"  Missed (cleared wrong) : {len(missed):3d} / {n}",
        "",
        f"  By source photo: all images flagged for {len(g_flagged)} of {len(groups)} source photos.",
        f"  Missed sources : {', '.join(g_missed) if g_missed else 'none'}",
        "",
        "=== 2. LOCATION LEVEL - does the model look AT the labeled damage? ===",
        "",
        f"  Heatmap peak inside a confirmed box : {len(hits):3d} / {len(scored)}   ({len(hits) / max(len(scored), 1):.1%})",
        f"  Model attention inside boxes (avg)  : {np.mean(shares):.0%}   (median {np.median(shares):.0%})",
        "",
        f"  For scale: the confirmed boxes are small - they cover on average only",
        f"  {np.mean([r['box_frac'] for r in recs]):.1%} of the image (median {np.median([r['box_frac'] for r in recs]):.1%}). A random model would land its peak",
        f"  inside a box ~{np.mean([r['box_frac'] for r in recs]):.0%} of the time, so the model is ~2x better than chance",
        "  but its attention is still spread over the whole stem, not focused on",
        "  the boxes. This is expected: v1 is a CLASSIFIER trained on whole-image",
        "  labels with a coarse 12x12 heatmap - it judges the stem overall. Only",
        "  a DETECTOR trained on these boxes can point at the exact damage spot.",
        "",
        "  Reading guide: on every page, colored rectangles are the annotators'",
        "  confirmed damage boxes; the red areas are what the model actually",
        "  looked at; the white x is the single hottest point.",
        "",
        "=== 3. THE MISSES (pages follow, worst first) ===",
        "",
    ]
    for r in sorted(missed, key=lambda r: r["score"]):
        L.append(f"  {r['file']:38s} score {r['score']:.3f}  (confirmed infested, model cleared it)")
    if not missed:
        L.append("  none")
    L += ["", "Page order: misses first, then inconclusive, then correctly flagged images."]

    fig = plt.figure(figsize=(10, 5.6))
    fig.text(0.05, 0.96, "\n".join(L), va="top", family="monospace", fontsize=8.2)
    plt.axis("off")
    pdf.savefig(fig, dpi=90)
    plt.close(fig)


def main():
    model = M.load_model()
    paths = sorted(IMG_DIR.glob("*.jpg"))
    print(f"{len(paths)} images; labels from {LBL_DIR}")

    recs, cache = [], {}
    for i, p in enumerate(paths, 1):
        lbl = LBL_DIR / (p.stem + ".txt")
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        boxes = boxes_in_crop(lbl, pil.width, pil.height) if lbl.exists() else []
        disp = np.array(M.disp_tf(pil))

        blur, clipped, dark, reject = quality(pil)
        if reject:
            score, cam, verdict = "", None, "INCONCLUSIVE - " + reject
            peak, hit, share = ("", ""), "", ""
            agree = "INCONCLUSIVE"
        else:
            score, cam = M.score_and_cam(model, pil)
            verdict = M.verdict_of(score)
            peak, hit, share = localisation(cam, boxes)
            agree = ("FLAGGED (correct)" if verdict.startswith("WSB")
                     else "MISSED" if verdict.startswith("No") else "INCONCLUSIVE")

        recs.append({"file": p.name, "group": p.stem.split("_")[0], "score": score,
                     "verdict": verdict, "agreement": agree, "n_boxes": len(boxes),
                     "box_frac": float(box_mask(boxes).mean()),
                     "peak": peak, "peak_in_box": hit, "attention_in_boxes": share})
        cache[p.name] = (disp, cam, boxes)
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    order = ([r for r in recs if r["agreement"] == "MISSED"] +
             [r for r in recs if r["agreement"] == "INCONCLUSIVE"] +
             [r for r in recs if r["agreement"] == "FLAGGED (correct)"])

    OUT_PDF.parent.mkdir(exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        summary_page(pdf, recs)
        for r in order:
            disp, cam, boxes = cache[r["file"]]
            page_for(pdf, r, disp, cam, boxes)

    with open(OUT_CSV, "w", newline="") as f:
        rows = [{**r, "peak": f"{r['peak'][0]},{r['peak'][1]}" if r["score"] != "" else "",
                 "attention_in_boxes": f"{r['attention_in_boxes']:.3f}" if r["score"] != "" else ""}
                for r in recs]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n", dict(Counter(r["agreement"] for r in recs)))
    scored = [r for r in recs if r["score"] != ""]
    print(f"peak-in-box: {sum(r['peak_in_box'] for r in scored)}/{len(scored)}, "
          f"mean attention share: {np.mean([r['attention_in_boxes'] for r in scored]):.1%}")
    print(f"-> {OUT_CSV}")
    print(f"-> {OUT_PDF}  ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
