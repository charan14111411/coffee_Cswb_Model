"""Shared helpers for version_rbn.

Everything here is READ-ONLY with respect to v1. Nothing in version_rbn ever
writes into models/v1/, src/, colab/, old_data/ or work/.

The one idea of version_rbn: do not feed the whole photo. Feed overlapping
pieces, and score a photo as the HIGHEST piece score.
"""
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

# --- the geometry of a piece -------------------------------------------------
PIECE_FRAC = 0.50     # piece side = half the SHORT edge of the photo
OVERLAP = 0.50        # neighbouring pieces share half their area
IMG = 384             # what the network eats

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"

# A piece is already square, so we only resize it to 384. No centre crop -
# a centre crop would throw away the edge of the piece, and the damage may be
# exactly there.
#
# LANCZOS on purpose: cut_pieces.py saves the training pieces with LANCZOS, so
# scoring a live photo must shrink them the same way. If training used LANCZOS
# and serving used bilinear, the model would see slightly different pictures
# than it was taught on - a small train/serve mismatch that is easy to avoid.
RESAMPLE = transforms.InterpolationMode.LANCZOS

eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG), interpolation=RESAMPLE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# v1's own evaluation transform, copied from src/train.py where the comment says
# it "MUST stay identical in the API". A whole frame is NOT square, so squashing
# it to 384x384 would show v1 a picture it was never trained or served on, and
# the baseline numbers would not be v1's real behaviour. Used only for
# --mode whole, which exists to measure v1.
whole_tf = transforms.Compose([
    transforms.Resize(IMG + 32),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.30, 0.30, 0.30, 0.05),   # kills the camera's colour
    transforms.RandomGrayscale(p=0.10),               # kills colour as a shortcut
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
])


def piece_boxes(w: int, h: int, frac: float = PIECE_FRAC, overlap: float = OVERLAP):
    """Top-left corners of overlapping square pieces covering the whole photo."""
    side = max(int(min(w, h) * frac), 32)
    stride = max(int(side * (1.0 - overlap)), 1)

    def starts(total):
        xs = list(range(0, max(total - side, 0) + 1, stride))
        if total > side and xs[-1] != total - side:
            xs.append(total - side)       # always cover the far edge
        return xs

    return side, [(x, y) for y in starts(h) for x in starts(w)]


def pieces_of(pil):
    """Cut a PIL photo into its pieces. Returns (side, boxes, list of PIL pieces)."""
    w, h = pil.size
    side, boxes = piece_boxes(w, h)
    return side, boxes, [pil.crop((x, y, x + side, y + side)) for x, y in boxes]


def load_model(ckpt_path, device=None):
    """Load any checkpoint written by this project. Never writes to the file."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    model = timm.create_model(ck.get("backbone", BACKBONE), pretrained=False, num_classes=2)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)
    return model, ck, dev


@torch.no_grad()
def score_whole(model, pil, dev):
    """v1 style: one score for the whole frame, using v1's own eval transform."""
    x = whole_tf(pil).unsqueeze(0).to(dev)
    return float(torch.softmax(model(x), 1)[0, 1])


@torch.no_grad()
def score_pieces(model, pil, dev, batch=24):
    """version_rbn style. Returns (best_score, all_scores, side, boxes)."""
    side, boxes, pcs = pieces_of(pil)
    out = []
    for i in range(0, len(pcs), batch):
        x = torch.stack([eval_tf(p) for p in pcs[i:i + batch]]).to(dev)
        out.append(torch.softmax(model(x), 1)[:, 1].cpu())
    s = torch.cat(out).numpy()
    return float(s.max()), s, side, boxes


def score_photo(model, pil, dev, mode):
    """mode is 'whole' (v1) or 'pieces' (version_rbn)."""
    if mode == "whole":
        return score_whole(model, pil, dev)
    return score_pieces(model, pil, dev)[0]


def blur_down(pil, px):
    """Throw away ALL fine detail: shrink to px x px, blow back up to 384.

    At 64x64 a bore hole cannot exist. Whatever score survives this is coming
    from overall colour and layout - that is background, not disease.
    """
    return pil.resize((px, px), Image.BILINEAR).resize((IMG, IMG), Image.BILINEAR)


def blackout_centre(pil):
    """Remove the middle 50% - where the stem normally is. Background only."""
    w, h = pil.size
    cw, chh = w // 2, h // 2
    x0, y0 = (w - cw) // 2, (h - chh) // 2
    out = pil.copy()
    out.paste((0, 0, 0), (x0, y0, x0 + cw, y0 + chh))
    return out


def centre_only(pil):
    """Keep only the middle 50%."""
    w, h = pil.size
    cw, chh = w // 2, h // 2
    x0, y0 = (w - cw) // 2, (h - chh) // 2
    return pil.crop((x0, y0, x0 + cw, y0 + chh))


def balanced_accuracy(truth, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    truth = np.asarray(truth)
    per = [(pred[truth == c] == c).mean() for c in (0, 1) if (truth == c).any()]
    return float(np.mean(per))
