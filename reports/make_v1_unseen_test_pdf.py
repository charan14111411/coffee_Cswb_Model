"""Test PDF for the outer-layer model v1 on photos it never used for training or validation.

Three sets:
  A  v1's held-out test split (work/splits/split.csv, split == test), whole-plant photos
  B  images_health_16_11_v2: infested stem close-ups (added later, for v2)
  C  test_new: healthy close-ups (exact duplicates removed)

Uses the v1 code in src/make_report_pdf.py (model loader, preprocessing, Grad-CAM, thresholds).
Writes reports/cwsb_v1_unseen_test_report.pdf and a CSV.   Run:  python reports/make_v1_unseen_test_pdf.py
"""
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_report_pdf as M  # noqa: E402

OUT = ROOT / "reports" / "cwsb_v1_unseen_test_report.pdf"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
T_LOW, T_HIGH = M.T_LOW, M.T_HIGH

sets = []
rows = [r for r in csv.DictReader(open(ROOT / "work" / "splits" / "split.csv")) if r["split"] == "test"]
sets.append(("A", "v1 held-out test split (whole-plant photos)",
             [(ROOT / "work" / "clean" / r["file"], r["label"]) for r in rows]))
sets.append(("B", "Infested stem close-ups (images_health_16_11_v2)",
             [(p, "Unhealthy") for p in sorted((ROOT / "images_health_16_11_v2").iterdir()) if p.suffix.lower() in EXT]))
seen, healthy = set(), []
for p in sorted((ROOT / "test_new").rglob("*")):
    if p.is_file() and p.suffix.lower() in EXT:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            healthy.append((p, "Healthy"))
sets.append(("C", "Healthy close-ups (test_new)", healthy))


def result_of(s):
    return "Unhealthy" if s >= T_HIGH else "Healthy" if s < T_LOW else "Unsure"


model = M.load_model()
recs = []
for code, title, items in sets:
    for i, (p, truth) in enumerate(items, 1):
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        score, cam = M.score_and_cam(model, pil)
        res = result_of(score)
        mark = "UNSURE" if res == "Unsure" else "CORRECT" if res == truth else "WRONG"
        small = np.array(M.disp_tf(pil).resize((256, 256)))
        heat = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
        recs.append({"set": code, "set_title": title, "file": p.name, "truth": truth, "score": score, "result": res,
                     "mark": mark, "photo": small, "cam": heat})
        if i % 40 == 0:
            print(f"  set {code}: {i}/{len(items)}", flush=True)


def tally(sub):
    inf = [r for r in sub if r["truth"] == "Unhealthy"]
    hea = [r for r in sub if r["truth"] == "Healthy"]
    c = lambda lst, res: sum(r["result"] == res for r in lst)  # noqa: E731
    return inf, hea, c


L = ["CWSB OUTER-LAYER MODEL v1 - TEST ON PHOTOS IT NEVER TRAINED ON", "",
     "Model      : v1 (ConvNeXt-Tiny classification), models/v1/contender_B.pt",
     f"Result rule: score below {T_LOW:.2f} = Healthy | {T_LOW:.2f} to {T_HIGH:.2f} = Unsure | {T_HIGH:.2f} or above = Unhealthy",
     f"Photos     : {len(recs)}. None was used to train or validate v1.", ""]
for code, title, items in sets:
    sub = [r for r in recs if r["set"] == code]
    inf, hea, c = tally(sub)
    L += [f"--- SET {code}: {title} ---", f"  Photos: {len(sub)}  ({len(inf)} infested, {len(hea)} healthy)"]
    if inf:
        L.append(f"  Infested photos : {c(inf, 'Unhealthy'):3d} called Unhealthy (correct), {c(inf, 'Unsure'):3d} Unsure, {c(inf, 'Healthy'):3d} called Healthy (missed)")
    if hea:
        L.append(f"  Healthy photos  : {c(hea, 'Healthy'):3d} called Healthy (correct), {c(hea, 'Unsure'):3d} Unsure, {c(hea, 'Unhealthy'):3d} called Unhealthy (false alarm)")
    L.append("")
inf, hea, c = tally(recs)
L += ["--- ALL SETS TOGETHER ---",
      f"  Infested photos : {len(inf)}.  Caught {c(inf, 'Unhealthy')} ({c(inf, 'Unhealthy') / max(len(inf), 1):.0%}), unsure {c(inf, 'Unsure')}, missed {c(inf, 'Healthy')}.",
      f"  Healthy photos  : {len(hea)}.  Cleared {c(hea, 'Healthy')} ({c(hea, 'Healthy') / max(len(hea), 1):.0%}), unsure {c(hea, 'Unsure')}, false alarm {c(hea, 'Unhealthy')}.",
      "",
      "--- HOW TO READ THIS ---", "",
      "  Set A is the fair test: whole-plant photos like the ones v1 was trained on,",
      "  from plants kept out of training. It has only 8 healthy photos.",
      "  Sets B and C are stem close-ups. v1 saw almost no close-ups in training and",
      "  no healthy close-ups at all, so results there show its limits.",
      "  The infested close-ups in set B are 47 source photos in several cropped or",
      "  rotated versions, so they are not 211 independent plants.",
      "",
      "  Each row of the pages shows the photo as the model saw it (centre square),",
      "  the heatmap of what pushed the score (red = most), and the result.",
      "  Within each set: wrong results first, then unsure, then correct."]

order = {"WRONG": 0, "UNSURE": 1, "CORRECT": 2}
COLR = {"WRONG": "#b3261e", "UNSURE": "#8a5a00", "CORRECT": "#1f7a43"}
with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.05, 0.96, "\n".join(L), va="top", family="monospace", fontsize=8.3)
    pdf.savefig(fig)
    plt.close(fig)
    for code, title, items in sets:
        sub = sorted([r for r in recs if r["set"] == code], key=lambda r: (order[r["mark"]], r["file"]))
        for s in range(0, len(sub), 4):
            part = sub[s:s + 4]
            fig, axes = plt.subplots(4, 3, figsize=(8.27, 11.69), gridspec_kw={"width_ratios": [1, 1, 0.95]})
            fig.suptitle(f"Set {code}: {title}   ({s + 1}-{s + len(part)} of {len(sub)})", fontsize=10, y=0.985)
            for ax in axes.ravel():
                ax.axis("off")
            for row, r in zip(axes, part):
                row[0].imshow(r["photo"])
                row[1].imshow(r["photo"])
                row[1].imshow(r["cam"], cmap="jet", alpha=0.45)
                row[2].text(0.02, 0.92, r["file"][:34], fontsize=7.5, va="top", family="monospace")
                row[2].text(0.02, 0.76, f"True label : {r['truth']}", fontsize=9.5, va="top")
                row[2].text(0.02, 0.62, f"Model says : {r['result']}", fontsize=9.5, va="top", weight="bold", color=COLR[r["mark"]])
                row[2].text(0.02, 0.48, f"Score      : {r['score']:.4f}", fontsize=9.5, va="top")
                row[2].text(0.02, 0.30, r["mark"], fontsize=13, va="top", weight="bold", color=COLR[r["mark"]])
            fig.subplots_adjust(left=0.02, right=0.99, top=0.955, bottom=0.01, wspace=0.03, hspace=0.05)
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

with open(OUT.with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set", "file", "true_label", "score", "model_result", "mark"])
    for r in recs:
        w.writerow([r["set"], r["file"], r["truth"], f"{r['score']:.4f}", r["result"], r["mark"]])
print("\n".join(L[5:22]))
print("->", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
