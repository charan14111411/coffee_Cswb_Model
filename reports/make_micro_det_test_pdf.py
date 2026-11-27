"""Test PDF for the microscope detection model cwsb_micro_det_v1.pt.

Runs the model on 321 microscope frames from four capture sessions that were
never uploaded to Roboflow, so the model did not see them in training or
validation. Writes reports/cwsb_micro_det_v1_test_report.pdf and a CSV.

"Reference" boxes are Claude's own earlier reading of these frames. They are
NOT expert labels. They are passed in as a JSON file:
    python reports/make_micro_det_test_pdf.py <reference.json>
"""
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RM = ROOT / "Resource_for_micro"
MODEL = ROOT / "cwsb_micro_det_v1" / "cwsb_micro_det_v1.pt"
OUT = ROOT / "reports" / "cwsb_micro_det_v1_test_report.pdf"
CONF, IMG, IOU_MATCH = 0.25, 512, 0.30
SESS = {
    "b": ("25 Nov 10:26 session", RM / "20261125_111814_838244" / "20261125_102437_492646"),
    "c": ("25 Nov 10:35 session", RM / "20261125_111814_838244" / "20261125_103531_886292"),
    "d": ("25 Nov 11:03 session", RM / "20261125_111814_838244" / "20261125_110256_102067"),
    "f": ("23 Nov session", RM / "Another source infested data" / "bark_044_region_01"),
}
OLD = {0: "hole", 1: "frass", 2: "egg", 3: "larva"}     # ids in the reference file; cracks are ignored
COL = {"egg": "#ff00ff", "frass": "#ff8c00", "hole": "#ff2020", "larva": "#1e90ff", "ridge": "#ffff00"}

ref_all = json.load(open(sys.argv[1]))
model = YOLO(str(MODEL))
NAMES = model.names


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


recs = []
for key, (title, folder) in SESS.items():
    files = sorted(folder.glob("bark_*.png"))
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=IMG, conf=CONF, verbose=False)):
            n = int(p.name.split("_")[1])
            a = ref_all.get(f"{key}{n}", {"status": "ok", "boxes": []})
            ref = [(OLD[b[0]], tuple(b[1:])) for b in a["boxes"] if b[0] in OLD]
            det = [(NAMES[int(c)], float(s), tuple(float(v) for v in xy))
                   for c, s, xy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
            used, found = set(), []
            for cls, rb in ref:
                best = max(((iou(rb, d[2]), j) for j, d in enumerate(det) if j not in used and d[0] == cls), default=(0, -1))
                ok = best[0] >= IOU_MATCH
                if ok:
                    used.add(best[1])
                found.append(ok)
            extra = [d for j, d in enumerate(det) if j not in used]
            recs.append({"session": title, "file": p.name, "path": p, "ref": ref, "det": det, "found": found,
                         "extra": extra, "blurred": a["status"] == "skip"})

# ---------------------------------------------------------------- numbers
per = {c: collections.Counter() for c in ("hole", "egg", "larva", "frass", "ridge")}
for r in recs:
    for (cls, _), ok in zip(r["ref"], r["found"]):
        per[cls]["ref"] += 1
        per[cls]["found"] += ok
    for d in r["extra"]:
        per[d[0]]["extra"] += 1
with_ref = [r for r in recs if r["ref"]]
no_ref = [r for r in recs if not r["ref"]]
alarm_no_ref = [r for r in no_ref if r["det"]]
silent = [r for r in no_ref if not r["det"]]
frames_hit = [r for r in with_ref if any(r["found"])]

L = ["CWSB MICROSCOPE DETECTION MODEL - TEST REPORT", "",
     f"Model       : cwsb_micro_det_v1.pt  (YOLO26 Nano, input {IMG}x{IMG})",
     f"Classes     : {', '.join(NAMES[i] for i in sorted(NAMES))}",
     f"Confidence  : {CONF}  (a box is drawn when the model is at least this sure)",
     "",
     f"Images tested: {len(recs)} microscope frames (640x480) from 4 capture sessions.",
     "None of these sessions was uploaded to Roboflow, so the model never saw",
     "these frames in training or validation.", ""]
for title in dict.fromkeys(r["session"] for r in recs):
    sub = [r for r in recs if r["session"] == title]
    L.append(f"  {title:22s} {len(sub):4d} frames, {sum(bool(r['ref']) for r in sub):3d} with a reference sign")
L += ["",
      "IMPORTANT: the 'reference' signs are Claude's own reading of the frames,",
      "made before this model existed. They are NOT expert-confirmed labels.",
      "A model box counts as a match when it has the same class and overlaps the",
      f"reference box (IoU >= {IOU_MATCH}).", "",
      "--- SIGNS FOUND (against the reference) ---", "",
      f"  {'class':8s} {'reference':>10s} {'found':>7s} {'missed':>7s}"]
for c in ("hole", "egg", "larva", "frass", "ridge"):
    n = per[c]["ref"]
    L.append(f"  {c:8s} {n:10d} {per[c]['found']:7d} {n - per[c]['found']:7d}" + ("" if n else "   (no reference example in these frames)"))
tot = sum(per[c]["ref"] for c in per)
totf = sum(per[c]["found"] for c in per)
L += ["", f"  All classes: {totf} of {tot} reference signs found ({totf / max(tot, 1):.0%}).",
      f"  Frames with a reference sign where the alarm would ring correctly: {len(frames_hit)} of {len(with_ref)}.", "",
      "--- FRAMES WITH NO REFERENCE SIGN ---", "",
      f"  Frames: {len(no_ref)}",
      f"  Model stayed silent: {len(silent)}",
      f"  Model drew at least one box: {len(alarm_no_ref)}  (possible false alarms; see the pages)",
      "  Boxes drawn on these frames, by class: " + (", ".join(f"{c} {n}" for c, n in collections.Counter(d[0] for r in alarm_no_ref for d in r["det"]).most_common()) or "none"),
      "", "--- HOW TO READ THE PAGES ---", "",
      "  Solid coloured box  = the model's detection, with class and confidence.",
      "  Dashed white box    = the reference sign (Claude's reading).",
      "  Section 1: frames with a reference sign.  Section 2: frames with no",
      "  reference sign where the model drew a box.  Section 3: frames where the",
      "  model stayed silent (small pictures).", "",
      "  Ridge cannot appear in a microscope frame and frass has no reference",
      "  example in these sessions, so those two classes are not tested here."]

with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, "\n".join(L), va="top", family="monospace", fontsize=8.6)
    pdf.savefig(fig)
    plt.close(fig)

    def pages(items, heading, cols, rows, small=False):
        per_page = cols * rows
        for s in range(0, len(items), per_page):
            part = items[s:s + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
            fig.suptitle(f"{heading}   ({s + 1}-{s + len(part)} of {len(items)})", fontsize=10, y=0.985)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, r in zip(axes.ravel(), part):
                ax.imshow(Image.open(r["path"]).convert("RGB"))
                for cls, rb in r["ref"]:
                    ax.add_patch(Rectangle((rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1], fill=False, ec="white", lw=1.6, ls="--"))
                for cls, s_, xy in r["det"]:
                    ax.add_patch(Rectangle((xy[0], xy[1]), xy[2] - xy[0], xy[3] - xy[1], fill=False, ec=COL[cls], lw=2))
                    ax.text(xy[0] + 3, max(xy[1] - 6, 12), f"{cls} {s_:.2f}", color="black", fontsize=6.5 if not small else 5,
                            bbox=dict(fc=COL[cls], ec="none", pad=1))
                if small:
                    ax.set_title(r["file"][5:8] + " " + r["session"][:12], fontsize=5, pad=1)
                else:
                    refs = ", ".join(f"{c} {'FOUND' if ok else 'MISSED'}" for (c, _), ok in zip(r["ref"], r["found"])) or "no reference sign"
                    extra = f" | other boxes: {', '.join(d[0] for d in r['extra'])}" if r["extra"] and r["ref"] else ""
                    blur = "  [blurred frame]" if r["blurred"] else ""
                    colr = "#b3261e" if (r["ref"] and not all(r["found"])) or (not r["ref"] and r["det"]) else "#1f7a43"
                    ax.set_title(f"{r['session']} / {r['file']}{blur}\n{refs}{extra}", fontsize=6.6, color=colr, pad=3)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.01, wspace=0.04, hspace=0.16 if not small else 0.12)
            pdf.savefig(fig, dpi=110)
            plt.close(fig)

    order1 = sorted(with_ref, key=lambda r: (all(r["found"]), r["session"], r["file"]))
    pages(order1, "1. Frames with a reference sign (missed first)", 2, 4)
    pages(sorted(alarm_no_ref, key=lambda r: -max(d[1] for d in r["det"])), "2. No reference sign, but the model drew a box", 2, 4)
    pages(silent, "3. No reference sign and the model stayed silent", 5, 8, small=True)

with open(OUT.with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["session", "file", "blurred", "reference_signs", "found", "model_detections"])
    for r in recs:
        w.writerow([r["session"], r["file"], r["blurred"], "; ".join(c for c, _ in r["ref"]),
                    "; ".join("yes" if ok else "no" for ok in r["found"]),
                    "; ".join(f"{c} {s:.2f}" for c, s, _ in r["det"])])
print({c: dict(v) for c, v in per.items()})
print("frames:", len(recs), "| with ref:", len(with_ref), "hit:", len(frames_hit), "| no ref:", len(no_ref), "alarm:", len(alarm_no_ref), "silent:", len(silent))
print("->", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
