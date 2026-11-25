"""STEP 9 - look with your own eyes.

One PDF page per photo:

   +------------------------+   +--------------------+
   | whole photo            |   | that piece, zoomed |
   |      +------+          |   |                    |
   |      |yellow| <- the   |   |   big and clear    |
   |      +------+   piece  |   |                    |
   +------------------------+   +--------------------+

If the yellow box sits on bark damage, the model is real.
If it sits on soil or leaves, the numbers lied to you.

Wrong answers are placed FIRST, so you cannot skip them.

Writes only version_rbn/reports/.

    python version_rbn/src/look.py --fold 0
    python version_rbn/src/look.py --folder some/photos
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rbn_common as C

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
MODELS = ROOT / "models" / "version_rbn"
REPORTS = HERE / "reports"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def verdict(s, lo, hi):
    return "wsb_likely" if s >= hi else ("no_visible_signs" if s < lo else "inconclusive")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--folder", default=None, help="score any folder instead of the test split")
    ap.add_argument("--max-pages", type=int, default=60)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else MODELS / f"rbn_fold{args.fold}.pt"
    cfg_path = MODELS / f"config_fold{args.fold}.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"T_low": 0.5, "T_high": 0.9}
    lo, hi = float(cfg["T_low"]), float(cfg["T_high"])

    model, ck, dev = C.load_model(ckpt)
    print(f"model      : {ckpt}")
    print(f"scoring    : MAX over pieces   thresholds {lo:.2f} / {hi:.2f}")

    # ---- which photos ----
    if args.folder:
        base = Path(args.folder)
        items = [(p, None) for p in sorted(base.rglob("*")) if p.suffix.lower() in EXTS]
        title = base.name
    else:
        with open(HERE / "work" / "folds.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if int(r["fold"]) == args.fold and r["split"] == "test"]
        items = [(ROOT / "work/clean" / r["file"], r["label"]) for r in rows]
        title = f"fold{args.fold}_test"
    print(f"photos     : {len(items)}\n")

    recs = []
    for p, truth in items:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        best, scores, side, boxes = C.score_pieces(model, im, dev)
        k = int(scores.argmax())
        v = verdict(best, lo, hi)
        mark = ("-" if truth is None else
                "inconclusive" if v == "inconclusive" else
                "CORRECT" if (v == "wsb_likely") == (truth == "Unhealthy") else "WRONG")
        recs.append({"file": p.name, "truth": truth, "score": best, "verdict": v,
                     "mark": mark, "n": len(boxes), "box": boxes[k], "side": side, "im": im})
        print(f"  {p.name:<46s} {best:.3f}  {v:<17s} {mark}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / f"{title}_rbn.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "score", "verdict", "mark", "pieces", "box_x", "box_y", "side"])
        for r in recs:
            w.writerow([r["file"], r["truth"] or "", f"{r['score']:.4f}", r["verdict"],
                        r["mark"], r["n"], r["box"][0], r["box"][1], r["side"]])

    order = ([r for r in recs if r["mark"] == "WRONG"] +
             [r for r in recs if r["mark"] == "inconclusive"] +
             [r for r in recs if r["mark"] not in ("WRONG", "inconclusive")])
    pdf_path = REPORTS / f"{title}_rbn.pdf"
    with PdfPages(pdf_path) as pdf:
        dec = [r for r in recs if r["mark"] in ("CORRECT", "WRONG")]
        L = [f"version_rbn - fold {args.fold}", "",
             f"scoring     : highest of {recs[0]['n'] if recs else 0} pieces per photo",
             f"thresholds  : < {lo:.2f} no signs | inconclusive | >= {hi:.2f} wsb likely",
             f"photos      : {len(recs)}", ""]
        if dec:
            L += [f"correct     : {sum(r['mark'] == 'CORRECT' for r in dec)} of {len(dec)} decided",
                  f"wrong       : {sum(r['mark'] == 'WRONG' for r in dec)}",
                  f"inconclusive: {sum(r['mark'] == 'inconclusive' for r in recs)}", ""]
        L += ["Wrong answers are the first pages. Look at the yellow box:",
              "on bark damage = real. On soil or leaves = the number lied."]
        fig = plt.figure(figsize=(10, 5.4))
        fig.text(0.05, 0.95, "\n".join(L), va="top", family="monospace", fontsize=9)
        plt.axis("off"); pdf.savefig(fig, dpi=90); plt.close(fig)

        for r in order[: args.max_pages]:
            im = r["im"]
            th = im.copy(); th.thumbnail((900, 900))
            sc = th.width / im.width
            bx, by = r["box"]
            fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
            ax[0].imshow(th)
            ax[0].add_patch(mpatches.Rectangle((bx * sc, by * sc), r["side"] * sc,
                                               r["side"] * sc, fill=False,
                                               edgecolor="yellow", linewidth=2.5))
            ax[0].set_title("photo + the piece that decided", fontsize=10)
            crop = im.crop((bx, by, bx + r["side"], by + r["side"]))
            crop.thumbnail((700, 700))
            ax[1].imshow(crop)
            ax[1].set_title("that piece, zoomed in", fontsize=10)
            for a in ax:
                a.axis("off")
            col = {"CORRECT": "#2e7d32", "WRONG": "#b3261e",
                   "inconclusive": "#a05a00", "-": "#333333"}[r["mark"]]
            t = f'{r["file"]}   score {r["score"]:.3f}  ->  {r["verdict"]}'
            if r["truth"]:
                t += f'   true: {r["truth"]}   [{r["mark"]}]'
            fig.suptitle(t, color=col, fontsize=11)
            pdf.savefig(fig, dpi=85); plt.close(fig)

    print(f"\n-> {csv_path}")
    print(f"-> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
