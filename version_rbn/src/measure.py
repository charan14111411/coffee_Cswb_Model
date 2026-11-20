"""STEP 1 and STEP 7 - does the model read the stem, or the background?

Two tests, both on photos the model was never trained to handle specially.

TEST A - blackout
    full    the photo as it is
    centre  only the middle 50% (where the stem usually is)
    border  the middle 50% BLACKED OUT - only background left
    A model that reads the stem scores 'border' low.

TEST B - detail
    the photo shrunk to 64x64 (or 32, 16) and blown back up.
    At 64x64 a bore hole cannot exist. Any score left is colour and layout.

READ-ONLY. Opens the checkpoint, prints numbers, writes nothing.

    python version_rbn/src/measure.py --ckpt models/v1/contender_B.pt
    python version_rbn/src/measure.py --ckpt models/version_rbn/rbn_fold0.pt --mode pieces
"""
import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rbn_common as C

ROOT = Path(__file__).resolve().parents[2]

# the numbers v1 produced, so you always have something to compare against
V1_BASELINE = {"border": 0.78, "blur64": 0.42, "full": 0.92, "false_alarm": 0.04}
TARGET = {"border": 0.25, "blur64": 0.15, "full": 0.85, "false_alarm": 0.10}


def gather(source, n_unhealthy, seed):
    """Photo paths per class. 'clean' works in Colab, 'old_data' on your PC."""
    if source == "clean":
        with open(ROOT / "work/splits/split.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        pick = {c: [ROOT / "work/clean" / r["file"] for r in rows if r["label"] == c]
                for c in ("Unhealthy", "Healthy")}
    else:
        pick = {"Unhealthy": sorted((ROOT / "old_data/Unhealthy(Infected)").glob("*.jpg")),
                "Healthy": sorted((ROOT / "old_data/Healthy").glob("*.jpg"))}
    rng = random.Random(seed)
    return {"Unhealthy": rng.sample(pick["Unhealthy"], min(n_unhealthy, len(pick["Unhealthy"]))),
            "Healthy": pick["Healthy"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["whole", "pieces"], default="whole",
                    help="whole = v1 style, pieces = version_rbn style (max over pieces)")
    ap.add_argument("--source", choices=["clean", "old_data"], default="clean")
    ap.add_argument("--n-unhealthy", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, ck, dev = C.load_model(args.ckpt)
    print(f"checkpoint : {args.ckpt}")
    print(f"backbone   : {ck.get('backbone')}   device {dev}")
    print(f"scoring    : {'MAX over pieces (version_rbn)' if args.mode == 'pieces' else 'whole frame (v1)'}")
    print(f"photos     : from {args.source}\n")

    sets = gather(args.source, args.n_unhealthy, args.seed)

    def sc(pil):
        return C.score_photo(model, pil, dev, args.mode)

    # ---------------- TEST A ----------------
    print("TEST A - blackout " + "-" * 44)
    print(f"{'class':<10}{'variant':<9}{'median':>8}{'mean':>8}{'>=0.90':>9}{'n':>5}")
    res = {}
    for cls, paths in sets.items():
        acc = {"full": [], "centre": [], "border": []}
        for p in paths:
            im = Image.open(p).convert("RGB")
            acc["full"].append(sc(im))
            acc["centre"].append(sc(C.centre_only(im)))
            acc["border"].append(sc(C.blackout_centre(im)))
        res[cls] = acc
        for k in ("full", "centre", "border"):
            v = acc[k]
            print(f"{cls:<10}{k:<9}{statistics.median(v):>8.3f}{statistics.mean(v):>8.3f}"
                  f"{sum(x >= 0.90 for x in v) / len(v):>8.0%}{len(v):>5}")

    # ---------------- TEST B ----------------
    print("\nTEST B - detail destroyed " + "-" * 36)
    print(f"{'detail kept':<16}{'infested >=0.90':>17}{'healthy >=0.90':>16}{'gap':>8}")
    blur = {}
    for px, name in ((None, "full detail"), (64, "64x64"), (32, "32x32"), (16, "16x16")):
        su = [sc(Image.open(p).convert("RGB") if px is None
                 else C.blur_down(Image.open(p).convert("RGB"), px)) for p in sets["Unhealthy"]]
        sh = [sc(Image.open(p).convert("RGB") if px is None
                 else C.blur_down(Image.open(p).convert("RGB"), px)) for p in sets["Healthy"]]
        rate = sum(x >= 0.90 for x in su) / len(su)
        blur[name] = rate
        print(f"{name:<16}{rate:>16.0%}{sum(x >= 0.90 for x in sh) / len(sh):>16.0%}"
              f"{statistics.median(su) - statistics.median(sh):>8.2f}")

    # ---------------- verdict ----------------
    def rate(cls, var):
        v = res[cls][var]
        return sum(x >= 0.90 for x in v) / len(v)

    got = {"border": rate("Unhealthy", "border"), "full": rate("Unhealthy", "full"),
           "false_alarm": rate("Healthy", "full"), "blur64": blur["64x64"]}

    print("\nSCORECARD " + "-" * 51)
    print(f"{'test':<34}{'v1':>7}{'now':>8}{'target':>9}{'':>6}")
    lines = [("infested, background only", "border", "<"),
             ("infested, 64x64 blur", "blur64", "<"),
             ("infested, full photo", "full", ">"),
             ("healthy, false alarm", "false_alarm", "<")]
    allpass = True
    for label, key, direction in lines:
        ok = got[key] < TARGET[key] if direction == "<" else got[key] > TARGET[key]
        allpass &= ok
        print(f"{label:<34}{V1_BASELINE[key]:>6.0%}{got[key]:>8.0%}"
              f"{direction + f'{TARGET[key]:.0%}':>9}{'  PASS' if ok else '  FAIL':>6}")
    print("-" * 61)
    print("VERDICT:", "ALL FOUR PASS - the model stopped using the background"
          if allpass else "NOT YET - see the FAIL lines above")
    print("\nNote: all four must pass together. A model that answers 'healthy' to")
    print("everything passes the first two lines and is useless - lines 3 and 4")
    print("exist to catch exactly that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
