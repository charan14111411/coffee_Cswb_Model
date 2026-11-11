"""Phase 1b - Group photos into pseudo-plants and make a leak-proof split.

Consecutive frame numbers (gap <= 4) or timestamps within 120 s from the same
camera series are treated as ONE plant. The split is done on plants, not
photos: 70/15/15 per class, so no plant ever appears in two sets. Flagged
(blurry/clipped/dark) images are excluded. The script refuses to finish if
any leakage is detected.
"""
import csv
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "work" / "clean"
OUT = ROOT / "work" / "splits" / "split.csv"

FRAME_GAP = 4       # frames
TIME_GAP = 120.0    # seconds
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

TS_RE = re.compile(r"(?P<pre>.*IMG)_(?P<d>\d{8})_(?P<t>\d{6})", re.I)
NUM_RE = re.compile(r"^(?P<pre>.*?)(?P<num>\d+)(?P<post>[^\d]*)$")


def sort_key(name, label):
    stem = Path(name).stem
    m = TS_RE.match(stem)
    if m:
        ts = datetime.strptime(m.group("d") + m.group("t"), "%Y%m%d%H%M%S")
        return (label, "ts:" + m.group("pre"), ts.timestamp())
    m = NUM_RE.match(stem)
    if m:
        return (label, "fr:" + m.group("pre") + "|" + m.group("post"), float(m.group("num")))
    return (label, "one:" + stem, 0.0)


def main() -> int:
    with open(CLEAN / "images.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    # --- group into pseudo-plants (within class + camera series only) ---
    keyed = sorted(rows, key=lambda r: sort_key(r["file"], r["label"]))
    groups = []
    prev = None
    for r in keyed:
        label, series, pos = sort_key(r["file"], r["label"])
        gap = TIME_GAP if series.startswith("ts:") else FRAME_GAP
        if prev and prev[0] == (label, series) and pos - prev[1] <= gap:
            groups[-1].append(r)
        else:
            groups.append([r])
        prev = ((label, series), pos)
    for gi, g in enumerate(groups):
        for r in g:
            r["group"] = f"plant_{gi:03d}"

    # --- split groups per class, excluding flagged images from counts ---
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for g in groups:
        usable = [r for r in g if not r["flags"]]
        if usable:
            by_class[g[0]["label"]].append(g)

    assign = {}
    for label, gs in by_class.items():
        rng.shuffle(gs)
        total = sum(len([r for r in g if not r["flags"]]) for g in gs)
        want = {s: total * frac for s, frac in FRACTIONS.items()}
        got = {s: 0 for s in FRACTIONS}
        # give each split one group first (so every set has both classes)
        order = ["test", "val", "train"]
        for s, g in zip(order, gs):
            assign[g[0]["group"]] = s
            got[s] += len([r for r in g if not r["flags"]])
        for g in gs[len(order):]:
            s = max(FRACTIONS, key=lambda s: want[s] - got[s])
            assign[g[0]["group"]] = s
            got[s] += len([r for r in g if not r["flags"]])

    for r in rows:
        r["split"] = "excluded" if r["flags"] else assign.get(r["group"], "excluded")

    # --- leakage checks ---
    plant_splits = defaultdict(set)
    for r in rows:
        if r["split"] != "excluded":
            plant_splits[r["group"]].add(r["split"])
    leaks = {g: s for g, s in plant_splits.items() if len(s) > 1}
    if leaks:
        print(f"LEAKAGE: {leaks}")
        return 1

    dup_leaks = []
    dup_file = CLEAN / "duplicates.csv"
    if dup_file.exists():
        split_of = {r["file"]: r["split"] for r in rows}
        with open(dup_file, newline="") as f:
            for d in csv.DictReader(f):
                sa, sb = split_of.get(d["file_a"]), split_of.get(d["file_b"])
                if sa and sb and sa != sb and "excluded" not in (sa, sb):
                    dup_leaks.append((d["file_a"], sa, d["file_b"], sb, d["kind"]))
    for d in dup_leaks:
        print(f"WARNING near-dup across splits: {d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "group", "split"])
        w.writeheader()
        w.writerows([{k: r[k] for k in ("file", "label", "group", "split")} for r in rows])

    print(f"Plants: {len(groups)} total "
          f"(Healthy {len(by_class.get('Healthy', []))}, Unhealthy {len(by_class.get('Unhealthy', []))})")
    tally = defaultdict(lambda: defaultdict(int))
    for r in rows:
        tally[r["split"]][r["label"]] += 1
    for s in ("train", "val", "test", "excluded"):
        h, u = tally[s].get("Healthy", 0), tally[s].get("Unhealthy", 0)
        print(f"  {s:8s}: {h:3d} healthy + {u:3d} unhealthy = {h + u}")
    print("LEAKAGE CHECK OK" + (f" ({len(dup_leaks)} near-dup warnings)" if dup_leaks else ""))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
