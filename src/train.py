"""Phase 2b - Train the CWSB classifier. Runs on Colab GPU (or CPU, slowly).

Two contenders, validation decides which ships:
  --mode A : frozen ConvNeXt-Tiny backbone -> logistic regression head
             (~1.5k trainable params, nearly impossible to overfit 28 images)
  --mode B : full ConvNeXt-Tiny fine-tune (2-epoch head warm-up, then all
             layers), weighted sampling + class-weighted loss + augmentation

Positive class = Unhealthy (infected). Model selection metric = validation
PR-AUC (average precision). Seed 42 everywhere.

Usage (from the project root, after clean/split/tiles):
  python src/train.py --mode A
  python src/train.py --mode B
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "work" / "clean"
TILES = ROOT / "work" / "tiles"
SPLIT = ROOT / "work" / "splits" / "split.csv"
TILES_CSV = ROOT / "work" / "tiles" / "tiles.csv"
OUT = ROOT / "models" / "v1"

IMG = 384
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
SEED = 42
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10),
    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([  # MUST stay identical in the API
    transforms.Resize(IMG + 32),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class CwsbDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        base = TILES if r.get("tile") else CLEAN
        img = Image.open(base / r["file"]).convert("RGB")
        return self.tf(img), 1 if r["label"] == "Unhealthy" else 0


def load_rows(split, with_tiles=False):
    with open(SPLIT, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if with_tiles and TILES_CSV.exists():
        with open(TILES_CSV, newline="") as f:
            rows += [dict(r, tile="1") for r in csv.DictReader(f) if r["split"] == split]
    return rows


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


@torch.no_grad()
def scores(model, rows, device, bs=16):
    model.eval()
    dl = DataLoader(CwsbDataset(rows, eval_tf), batch_size=bs, num_workers=2)
    ys, ps = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(device)), 1)[:, 1]
        ps += p.cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


@torch.no_grad()
def features(backbone, rows, device, bs=16):
    backbone.eval()
    dl = DataLoader(CwsbDataset(rows, eval_tf), batch_size=bs, num_workers=2)
    feats, ys = [], []
    for x, y in dl:
        f = backbone(x.to(device))
        feats.append(f.cpu().numpy())
        ys += y.tolist()
    return np.concatenate(feats), np.array(ys)


def run_a(device):
    """Frozen backbone + logistic regression."""
    backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(device)
    tr = load_rows("train", with_tiles=True)
    va = load_rows("val")
    print(f"A: {len(tr)} train rows (incl. tiles), {len(va)} val")
    t0 = time.time()
    Xtr, ytr = features(backbone, tr, device)
    Xva, yva = features(backbone, va, device)
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, C=0.1)
    clf.fit(Xtr, ytr)
    ap = average_precision_score(yva, clf.predict_proba(Xva)[:, 1])
    print(f"A: val PR-AUC = {ap:.4f}  ({time.time() - t0:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": "A", "backbone": BACKBONE, "img": IMG,
                "coef": clf.coef_, "intercept": clf.intercept_,
                "val_pr_auc": ap}, OUT / "contender_A.pt")
    return ap


def run_b(device, epochs=40, patience=6):
    """Full fine-tune with warm-up, weighted sampling, class-weighted loss."""
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=2).to(device)
    tr = load_rows("train", with_tiles=True)
    va = load_rows("val")
    n_pos = sum(r["label"] == "Unhealthy" for r in tr)
    n_neg = len(tr) - n_pos
    print(f"B: train {len(tr)} ({n_neg} healthy incl. tiles, {n_pos} unhealthy), val {len(va)}")

    weights = [1.0 / (n_pos if r["label"] == "Unhealthy" else n_neg) for r in tr]
    sampler = WeightedRandomSampler(weights, num_samples=len(tr), replacement=True)
    dl = DataLoader(CwsbDataset(tr, train_tf), batch_size=16, sampler=sampler, num_workers=2)
    cw = torch.tensor([len(tr) / (2 * n_neg), len(tr) / (2 * n_pos)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw.to(device))

    head = [p for n, p in model.named_parameters() if "head" in n]
    body = [p for n, p in model.named_parameters() if "head" not in n]

    best_ap, best_ep, since = 0.0, -1, 0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        warm = ep < 2
        for p in body:
            p.requires_grad = not warm
        opt = torch.optim.AdamW(
            [{"params": head, "lr": 1e-3 if warm else 1e-4},
             {"params": body, "lr": 0 if warm else 1e-4}], weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl))
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        yva, pva = scores(model, va, device)
        ap = average_precision_score(yva, pva)
        print(f"B ep{ep:02d} {'warm' if warm else 'full'}: loss {tot / len(dl):.4f}  val PR-AUC {ap:.4f}")
        if ap > best_ap:
            best_ap, best_ep, since = ap, ep, 0
            torch.save({"mode": "B", "backbone": BACKBONE, "img": IMG,
                        "state_dict": model.state_dict(),
                        "val_pr_auc": ap, "epoch": ep}, OUT / "contender_B.pt")
        else:
            since += 1
            if since >= patience:
                print(f"B: early stop (best ep{best_ep}, PR-AUC {best_ap:.4f})")
                break
    return best_ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--mode", choices=["A", "B", "both"], default="both")
    args = ap_.parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    results = {}
    if args.mode in ("A", "both"):
        results["A"] = run_a(device)
    if args.mode in ("B", "both"):
        results["B"] = run_b(device)
    if results:
        winner = max(results, key=results.get)
        (OUT / "training_result.json").write_text(json.dumps(
            {"val_pr_auc": results, "winner": winner,
             "backbone": BACKBONE, "img": IMG, "seed": SEED}, indent=2))
        print(f"\nWinner on validation: contender {winner}  {results}")


if __name__ == "__main__":
    main()
