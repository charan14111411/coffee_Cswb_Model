"""STEP 2 - honest test groups.

You have 9 healthy plants, and v1's validation set had only 2 healthy PHOTOS.
A result decided by 2 photos is luck, not measurement.

So the plants are rotated through 3 folds. In fold k:

    TEST  = block k of plants            (never seen during training)
    VAL   = a quarter of what is left    (used to stop training)
    TRAIN = the rest

Every plant is tested exactly once across folds 0, 1, 2.

THE RULE THAT NEVER BREAKS: all photos of one plant stay in the same set.
If photo 1 of a plant is in train and photo 2 is in test, the model has already
seen that plant and the test score is a lie. The script aborts if that happens.

Reads work/splits/split.csv. Writes only version_rbn/work/folds.csv.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SPLIT = ROOT / "work" / "splits" / "split.csv"
OUT = HERE / "work" / "folds.csv"
N_FOLDS = 3
VAL_SHARE = 0.25
MIN_VAL_PHOTOS = 3     # early stopping on 1 or 2 photos is meaningless
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    with open(SPLIT, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] != "excluded"]

    plants = defaultdict(list)                      # group -> rows
    label_of = {}
    for r in rows:
        plants[r["group"]].append(r)
        label_of[r["group"]] = r["label"]

    by_class = defaultdict(list)
    for g in plants:
        by_class[label_of[g]].append(g)

    print(f"photos: {len(rows)}")
    for c, gs in by_class.items():
        print(f"  {c:10s} {len(gs):3d} plants, {sum(len(plants[g]) for g in gs):3d} photos")

    # Plants are very uneven: one healthy plant holds 11 of 28 photos, one
    # infested plant holds 120 of 468. Dealing plants one by one would leave a
    # fold with almost no healthy photos to train on. So we balance by PHOTO
    # COUNT: biggest plant first, always into the fold that is currently
    # emptiest.
    rng = random.Random(SEED)
    size = {g: len(rs) for g, rs in plants.items()}
    blocks = {c: [[] for _ in range(args.folds)] for c in by_class}
    for c, gs in by_class.items():
        gs = sorted(gs, key=lambda g: (-size[g], g))
        tot = [0] * args.folds
        for g in gs:
            i = min(range(args.folds), key=lambda j: tot[j])
            blocks[c][i].append(g)
            tot[i] += size[g]
        print(f"  {c:10s} test blocks hold {tot} photos")

    out_rows = []
    for k in range(args.folds):
        test_g = {g for c in blocks for g in blocks[c][k]}

        # Validation: take the SMALLEST remaining plants until ~20% of the
        # remaining photos are covered. Smallest first, so validation never
        # swallows the one big plant that training needs.
        val_g = set()
        for c in by_class:
            rest = sorted([g for g in by_class[c] if g not in test_g],
                          key=lambda g: (size[g], g))
            want = max(MIN_VAL_PHOTOS, round(sum(size[g] for g in rest) * VAL_SHARE))
            budget = max(1, len(rest) - 2)          # always leave >=2 plants to train on
            chosen, took = [], 0
            for g in rest[:budget]:                 # smallest plants first
                if took >= want:
                    break
                chosen.append(g)
                took += size[g]
            if took < MIN_VAL_PHOTOS:
                # Too few photos this way (one big plant holds nearly all of
                # them). Take the smallest single plant that is big enough -
                # early stopping on 1 photo is worthless.
                big = [g for g in rest if size[g] >= MIN_VAL_PHOTOS]
                if big and budget >= 1:
                    chosen, took = [big[0]], size[big[0]]   # smallest plant big enough
            val_g.update(chosen)

        for g, grows in plants.items():
            s = "test" if g in test_g else ("val" if g in val_g else "train")
            for r in grows:
                out_rows.append({"file": r["file"], "label": r["label"], "group": g,
                                 "fold": k, "split": s})

        # --- leakage check, per fold ---
        where = defaultdict(set)
        for r in out_rows:
            if r["fold"] == k:
                where[r["group"]].add(r["split"])
        bad = {g: s for g, s in where.items() if len(s) > 1}
        if bad:
            print(f"LEAKAGE in fold {k}: {bad}")
            return 1

        cnt = defaultdict(lambda: defaultdict(int))
        pl = defaultdict(lambda: defaultdict(set))
        for r in out_rows:
            if r["fold"] == k:
                cnt[r["split"]][r["label"]] += 1
                pl[r["split"]][r["label"]].add(r["group"])
        print(f"\nfold {k}")
        for s in ("train", "val", "test"):
            h, u = cnt[s]["Healthy"], cnt[s]["Unhealthy"]
            ph, pu = len(pl[s]["Healthy"]), len(pl[s]["Unhealthy"])
            print(f"  {s:6s} healthy {h:3d} photos / {ph} plants   "
                  f"unhealthy {u:3d} photos / {pu} plants")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "group", "fold", "split"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nLEAKAGE CHECK OK - no plant is in two sets in any fold")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
