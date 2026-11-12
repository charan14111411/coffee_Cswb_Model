"""Score every test2 image with the winning model and characterize the failures.

Answers: which images are wrong, what kind of error, and where they sit
relative to the decision thresholds. Writes reports/failure_analysis.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def load_model(device):
    ck = torch.load(ROOT / "models" / "v1" / "contender_B.pt",
                    map_location=device, weights_only=False)
    m = timm.create_model(ck["backbone"], pretrained=False, num_classes=2)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def score_files(model, paths, device, bs=8):
    out = []
    for i in range(0, len(paths), bs):
        batch = paths[i:i + bs]
        xs = []
        for p in batch:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            xs.append(T.eval_tf(img))
        p_ = torch.softmax(model(torch.stack(xs).to(device)), 1)[:, 1]
        out += p_.cpu().tolist()
        print(f"  {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    return np.array(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    print(f"device {device}")

    # --- thresholds from the validation split (same recipe as Colab) ---
    va = T.load_rows("val")
    print("scoring validation...")
    pva = score_files(model, [ROOT / "work" / "clean" / r["file"] for r in va], device)
    yva = np.array([1 if r["label"] == "Unhealthy" else 0 for r in va])
    best, lab = {}, {}
    for r, y, p in zip(va, yva, pva):
        best[r["group"]] = max(best.get(r["group"], 0.0), float(p))
        lab[r["group"]] = int(y)
    gk = list(best)
    vy = np.array([lab[k] for k in gk]); vp = np.array([best[k] for k in gk])
    inf = np.sort(vp[vy == 1])
    T_high = float(np.quantile(inf, 0.10))
    T_low = float(np.quantile(inf, 0.02))
    if T_low >= T_high:
        T_low = T_high * 0.5
    print(f"T_low={T_low:.4f} T_high={T_high:.4f}")
    print(f"val healthy photo scores: {np.sort(pva[yva == 0])}")

    # --- score test2 ---
    key = {r["sample"]: r for r in csv.DictReader(open(ROOT / "test2" / "answer_key2.csv"))}
    paths = sorted((ROOT / "test2").glob("t2_*.jpg"))
    print(f"scoring {len(paths)} test2 images...")
    ps = score_files(model, paths, device)

    rows = []
    for p, s in zip(paths, ps):
        k = key[p.name]
        verdict = ("WSB_LIKELY" if s >= T_high else
                   "NO_SIGNS" if s < T_low else "INCONCLUSIVE")
        truth = k["true_label"]
        if verdict == "INCONCLUSIVE":
            outcome = "inconclusive"
        elif (verdict == "WSB_LIKELY") == (truth == "Unhealthy"):
            outcome = "correct"
        else:
            outcome = "FALSE_ALARM" if truth == "Healthy" else "MISS"
        rows.append({"sample": p.name, "truth": truth, "score": round(float(s), 4),
                     "verdict": verdict, "outcome": outcome,
                     "lighting": k["lighting"], "source": k["source"],
                     "split": k["split"], "plant": k["plant"],
                     "blur": k["blur"], "clipped": k["clipped"],
                     "origin": k["original"]})

    with open(OUT / "failure_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # --- report ---
    from collections import Counter
    print("\n=== OUTCOMES ===", dict(Counter(r["outcome"] for r in rows)))
    hs = [r["score"] for r in rows if r["truth"] == "Healthy"]
    us = [r["score"] for r in rows if r["truth"] == "Unhealthy"]
    print(f"healthy scores  : min {min(hs):.3f} med {np.median(hs):.3f} max {max(hs):.3f}")
    print(f"infected scores : min {min(us):.3f} med {np.median(us):.3f} max {max(us):.3f}")
    print(f"infected below 0.9: {sum(1 for s in us if s < 0.9)} of {len(us)}")

    print("\n=== EVERY FAILURE ===")
    for r in sorted([x for x in rows if x["outcome"] in ("MISS", "FALSE_ALARM")],
                    key=lambda x: x["score"]):
        print(f"  {r['sample']}  {r['outcome']:12s} score={r['score']:.3f} "
              f"truth={r['truth']:9s} light={r['lighting']:17s} plant={r['plant']} "
              f"from={r['origin']}")
    print("\n=== INCONCLUSIVE ===")
    for r in [x for x in rows if x["outcome"] == "inconclusive"]:
        print(f"  {r['sample']} score={r['score']:.3f} truth={r['truth']}")

    print("\n=== BY CATEGORY (error rate) ===")
    for field in ("lighting", "source", "truth", "split"):
        for v in sorted({r[field] for r in rows}):
            sub = [r for r in rows if r[field] == v]
            bad = [r for r in sub if r["outcome"] in ("MISS", "FALSE_ALARM")]
            print(f"  {field:9s} {v:17s}: {len(bad):2d}/{len(sub):3d} wrong")

    # what a better T_low would do
    print("\n=== IF T_low WERE 0.50 (band widened) ===")
    nb = Counter()
    for r in rows:
        s = r["score"]
        v = "WSB_LIKELY" if s >= T_high else ("NO_SIGNS" if s < 0.50 else "INCONCLUSIVE")
        if v == "INCONCLUSIVE":
            nb["inconclusive"] += 1
        elif (v == "WSB_LIKELY") == (r["truth"] == "Unhealthy"):
            nb["correct"] += 1
        else:
            nb["FALSE_ALARM" if r["truth"] == "Healthy" else "MISS"] += 1
    print(" ", dict(nb))
    print(f"\n-> {OUT / 'failure_analysis.csv'}")


if __name__ == "__main__":
    main()
