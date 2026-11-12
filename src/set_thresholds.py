"""Phase 3 - choose and freeze the decision thresholds.

Rationale (recorded in config.json):
  The first recipe took BOTH thresholds from the infected score distribution.
  Because the model early-stopped on validation PR-AUC = 1.0, validation
  infected scores are nearly all >0.99, so quantile(0.02) and quantile(0.10)
  collapsed onto each other (0.9071 / 0.9149) - an inconclusive band 0.008
  wide, and every hedged score (0.6-0.9) was announced as "no visible signs".

  Replacement rule, calibrated on validation only:
    T_low  = 0.50  - the neutral point. Validation healthy photos score
                     0.26 and 0.39, so 0.50 sits clearly above the healthy
                     band; below it the model genuinely leans healthy.
    T_high = 0.90  - strong confidence; validation infected plants all sit
                     above it.
  The test split is used to CHECK this choice, never to pick it.

Reads reports/failure_analysis.csv (scores already computed per image),
writes models/v1/config.json and prints val/test behaviour separately.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports" / "failure_analysis.csv"
CONFIG = ROOT / "models" / "v1" / "config.json"

T_LOW = 0.50
T_HIGH = 0.90


def verdict(score):
    if score >= T_HIGH:
        return "wsb_likely"
    if score < T_LOW:
        return "no_visible_signs"
    return "inconclusive"


def report(rows, title, by_plant):
    if by_plant:
        best, lab = {}, {}
        for r in rows:
            k = r["plant"]
            s = float(r["score"])
            if s > best.get(k, -1):
                best[k] = s
            lab[k] = r["truth"]
        items = [{"score": best[k], "truth": lab[k]} for k in best]
    else:
        items = [{"score": float(r["score"]), "truth": r["truth"]} for r in rows]

    tp = fn = fp = tn = inc = 0
    for it in items:
        v = verdict(it["score"])
        infected = it["truth"] == "Unhealthy"
        if v == "inconclusive":
            inc += 1
        elif v == "wsb_likely":
            tp += infected
            fp += not infected
        else:
            fn += infected
            tn += not infected
    unit = "plants" if by_plant else "photos"
    print(f"\n{title} ({len(items)} {unit})")
    print(f"  caught infected   : {tp}")
    print(f"  MISSED infected   : {fn}")
    print(f"  false alarms      : {fp}")
    print(f"  correct clears    : {tn}")
    print(f"  inconclusive      : {inc}  -> farmer is asked to retake")
    if tp + fn:
        print(f"  RECALL            : {tp / (tp + fn):.1%}")
    if tp + fp:
        print(f"  PRECISION         : {tp / (tp + fp):.1%}")
    if tp + tn + fp + fn:
        print(f"  ACCURACY (decided): {(tp + tn) / (tp + tn + fp + fn):.1%}")
    return {"caught": tp, "missed": fn, "false_alarms": fp,
            "cleared": tn, "inconclusive": inc}


def main():
    rows = [r for r in csv.DictReader(open(ANALYSIS))
            if not r["origin"].endswith("(simulated)")]  # real photos only
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    print(f"Thresholds: T_low={T_LOW}  T_high={T_HIGH}")
    print(f"(previous recipe gave T_low=0.9071 T_high=0.9149 - band 0.008 wide)")

    report(by_split["val"], "VALIDATION - used to calibrate", by_plant=False)
    test_photo = report(by_split["test"], "TEST - held out, honest number", by_plant=False)
    test_plant = report(by_split["test"], "TEST - plant level (how the app decides)", by_plant=True)

    sim = [r for r in csv.DictReader(open(ANALYSIS)) if r["origin"].endswith("(simulated)")]
    base = {r["sample"]: float(r["score"]) for r in csv.DictReader(open(ANALYSIS))}
    flips = 0
    for r in sim:
        src = r["origin"].split(" ")[0]
        if src in base and verdict(float(r["score"])) != verdict(base[src]):
            flips += 1
    print(f"\nLIGHTING STRESS: {flips} of {len(sim)} brightened/darkened copies "
          f"change verdict vs their original")

    CONFIG.write_text(json.dumps({
        "model": "contender_B.pt",
        "backbone": "convnext_tiny.fb_in22k_ft_in1k_384",
        "img": 384,
        "T_low": T_LOW,
        "T_high": T_HIGH,
        "plant_score": "max over the plant's photos",
        "answers": {
            "score >= T_high": "wsb_likely",
            "T_low <= score < T_high": "inconclusive",
            "score < T_low": "no_visible_signs",
        },
        "calibrated_on": "validation split only; test split used as a check",
        "rationale": ("validation healthy photos score 0.26-0.39, so T_low=0.50 "
                      "sits above the healthy band; T_high=0.90 marks strong "
                      "confidence. The earlier infected-quantile recipe collapsed "
                      "both thresholds to ~0.91 and reported hedged scores as "
                      "no_visible_signs."),
        "test_photo_level": test_photo,
        "test_plant_level": test_plant,
    }, indent=2))
    print(f"\nfrozen -> {CONFIG}")


if __name__ == "__main__":
    main()
