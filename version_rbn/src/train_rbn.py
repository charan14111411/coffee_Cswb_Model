"""STEP 5, 6, 8 - train version_rbn in two rounds, then set the thresholds.

ROUND 1  every piece carries its photo's label. This is wrong on purpose:
         most pieces of a sick plant show clean bark. Round 1 only has to be
         good enough to guess WHICH pieces hold the damage.

ROUND 2  score every training piece with the round-1 model.
           sick photo    -> keep only its TOPK best pieces
           healthy photo -> keep ALL pieces (every part of a healthy plant is
                            healthy, so every piece is a true sample)
         then throw round 1 away and train again from ImageNet weights.

This is Multiple Instance Learning: "the bag is positive, but I do not know
which item inside it is".

Early stopping watches PHOTO accuracy scored as max-over-pieces - the way the
model will actually be used. Never piece accuracy.

Thresholds come from HEALTHY validation scores. Never from infected scores:
that exact mistake already happened in this project and collapsed both lines
to 0.91.

Writes only models/version_rbn/ .
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rbn_common as C

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PIECES = HERE / "work" / "pieces"
MODELS = ROOT / "models" / "version_rbn"
SEED = 42


class PieceSet(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(PIECES / r["file"]).convert("RGB")
        return self.tf(im), 1 if r["label"] == "Unhealthy" else 0


def load_rows(fold):
    with open(HERE / "work" / "pieces.csv", newline="") as f:
        pieces = list(csv.DictReader(f))
    with open(HERE / "work" / "folds.csv", newline="") as f:
        split_of = {r["file"]: r["split"] for r in csv.DictReader(f)
                    if int(r["fold"]) == fold}
    out = defaultdict(list)
    for p in pieces:
        s = split_of.get(p["photo"])
        if s:
            out[s].append(p)
    return out


@torch.no_grad()
def piece_scores(model, rows, dev, bs=48):
    model.eval()
    dl = DataLoader(PieceSet(rows, C.eval_tf), batch_size=bs, num_workers=2, pin_memory=True)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            out.append(torch.softmax(model(x.to(dev, non_blocking=True)).float(), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def to_photo(rows, scores):
    """max over a photo's pieces -> (photo names, scores, truths)"""
    best, truth = defaultdict(float), {}
    for r, s in zip(rows, scores):
        best[r["photo"]] = max(best[r["photo"]], float(s))
        truth[r["photo"]] = 1 if r["label"] == "Unhealthy" else 0
    names = sorted(best)
    return names, np.array([best[n] for n in names]), np.array([truth[n] for n in names])


def fit(train_rows, val_rows, dev, epochs, patience, tag):
    torch.manual_seed(SEED)
    n_h = sum(r["label"] == "Healthy" for r in train_rows)
    n_u = len(train_rows) - n_h
    print(f"  train pieces: {n_h} healthy + {n_u} unhealthy")

    # balance the batches (17:1 imbalance) AND weight the loss - v1's recipe
    wts = [1.0 / max(n_h, 1) if r["label"] == "Healthy" else 1.0 / max(n_u, 1)
           for r in train_rows]
    sampler = WeightedRandomSampler(wts, num_samples=len(train_rows), replacement=True)
    dl = DataLoader(PieceSet(train_rows, C.train_tf), batch_size=16, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    cls_w = torch.tensor([len(train_rows) / (2 * max(n_h, 1)),
                          len(train_rows) / (2 * max(n_u, 1))],
                         dtype=torch.float32, device=dev)
    model = timm.create_model(C.BACKBONE, pretrained=True, num_classes=2).to(dev)
    crit = nn.CrossEntropyLoss(weight=cls_w)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        warm = ep <= 2
        for p in model.parameters():
            p.requires_grad = not warm
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-3 if warm else 1e-4, weight_decay=0.05)
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)

        _, ps, py = to_photo(val_rows, piece_scores(model, val_rows, dev))
        acc = C.balanced_accuracy(py, ps)
        hist.append({"epoch": ep, "loss": tot / len(dl.dataset), "val_photo_bal_acc": acc})
        print(f"  epoch {ep:2d} {'(warm-up)' if warm else '         '} "
              f"loss {tot / len(dl.dataset):.4f}   val photo bal-acc {acc:.3f}")
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}  (best {best:.3f})")
                break

    model.load_state_dict(best_state)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": C.BACKBONE, "img": C.IMG,
                "scoring": "max over pieces", "piece_frac": C.PIECE_FRAC,
                "overlap": C.OVERLAP, "val_photo_bal_acc": best, "history": hist},
               MODELS / f"{tag}.pt")
    print(f"  -> {MODELS / (tag + '.pt')}   best val photo bal-acc {best:.3f}")
    return model, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--skip-round2", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no GPU. This will take many hours - use Colab with a T4.")

    R = load_rows(args.fold)
    print(f"fold {args.fold}: pieces  train {len(R['train'])}  "
          f"val {len(R['val'])}  test {len(R['test'])}")

    print(f"\n=== ROUND 1 - every piece keeps its photo's label ===")
    model, _ = fit(R["train"], R["val"], dev, args.epochs, args.patience,
                   f"rbn_fold{args.fold}_round1")

    if not args.skip_round2:
        print(f"\n=== ROUND 2 - keep the top {args.topk} pieces of each sick photo ===")
        s = piece_scores(model, R["train"], dev)
        by = defaultdict(list)
        for i, r in enumerate(R["train"]):
            by[r["photo"]].append((float(s[i]), i))
        keep = []
        for photo, lst in by.items():
            if R["train"][lst[0][1]]["label"] == "Healthy":
                keep += [i for _, i in lst]
            else:
                keep += [i for _, i in sorted(lst, reverse=True)[: args.topk]]
        tr2 = [R["train"][i] for i in sorted(keep)]
        print(f"  {len(R['train'])} pieces -> {len(tr2)} pieces")
        model, _ = fit(tr2, R["val"], dev, args.epochs, args.patience,
                       f"rbn_fold{args.fold}")
    else:
        import shutil
        shutil.copy(MODELS / f"rbn_fold{args.fold}_round1.pt", MODELS / f"rbn_fold{args.fold}.pt")

    # ---------- STEP 8: thresholds from HEALTHY validation scores ----------
    _, vs, vy = to_photo(R["val"], piece_scores(model, R["val"], dev))
    healthy = np.sort(vs[vy == 0])
    if len(healthy):
        t_low = float(min(0.60, max(0.35, healthy.max() + 0.05)))
    else:
        t_low = 0.50
    t_high = 0.90
    print(f"\n=== THRESHOLDS (from {len(healthy)} healthy validation photos) ===")
    print(f"  healthy val scores: {np.round(healthy, 3).tolist()}")
    print(f"  T_low  = {t_low:.2f}   (just above the healthy band)")
    print(f"  T_high = {t_high:.2f}")

    # ---------- held-out TEST, photo level ----------
    names, ts, ty = to_photo(R["test"], piece_scores(model, R["test"], dev))
    print(f"\n=== HELD-OUT TEST, photo level (score = max over pieces) ===")
    for thr in (t_low, t_high):
        pred = (ts >= thr).astype(int)
        tp = int(((pred == 1) & (ty == 1)).sum()); fn = int(((pred == 0) & (ty == 1)).sum())
        fp = int(((pred == 1) & (ty == 0)).sum()); tn = int(((pred == 0) & (ty == 0)).sum())
        print(f"  at {thr:.2f}: caught {tp:3d}  missed {fn:3d}  "
              f"false alarms {fp:3d}  cleared {tn:3d}  bal-acc {C.balanced_accuracy(ty, ts, thr):.3f}")

    cfg = {"model": f"rbn_fold{args.fold}.pt", "backbone": C.BACKBONE, "img": C.IMG,
           "scoring": "max over pieces", "piece_frac": C.PIECE_FRAC, "overlap": C.OVERLAP,
           "T_low": round(t_low, 3), "T_high": t_high,
           "plant_score": "max over the plant's photos",
           "calibrated_on": "healthy validation photos of this fold only",
           "fold": args.fold, "topk": args.topk,
           "note": "version_rbn scores a photo as the HIGHEST of its pieces. "
                   "Serving MUST cut pieces the same way. v1 is untouched."}
    (MODELS / f"config_fold{args.fold}.json").write_text(json.dumps(cfg, indent=1),
                                                         encoding="utf-8")
    (MODELS / f"test_scores_fold{args.fold}.json").write_text(json.dumps(
        [{"photo": n, "score": float(x), "truth": int(t)} for n, x, t in zip(names, ts, ty)],
        indent=1), encoding="utf-8")
    print(f"\n-> {MODELS / f'config_fold{args.fold}.json'}")
    print(f"NEXT: measure.py --ckpt models/version_rbn/rbn_fold{args.fold}.pt --mode pieces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
