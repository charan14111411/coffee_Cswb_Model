"""How much does the outside of the frame drive the decision?

Scores every held-out test photo three ways:
  full    - the normal centre crop the model is trained on
  tight   - centre 70% of the frame (less surroundings)
  tighter - centre 50% of the frame (plant only, roughly)

If scores barely move, the model is already reading the plant and masking or
removing the background would buy little. If they swing, the surroundings are
doing real work in the decision - for better or worse.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_report_pdf as M  # noqa: E402
import train as T  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tf_for(frac):
    """Keep the central `frac` of the image, then the usual 384 pipeline."""
    def crop(img):
        w, h = img.size
        cw, ch = int(w * frac), int(h * frac)
        return img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    return transforms.Compose([
        transforms.Lambda(crop),
        transforms.Resize(M.IMG + 32), transforms.CenterCrop(M.IMG),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def score(model, pil, tf):
    return torch.softmax(model(tf(pil).unsqueeze(0)), 1)[0, 1].item()


def main():
    model = M.load_model()
    rows = T.load_rows("test")
    tfs = {"full": tf_for(1.0), "tight": tf_for(0.70), "tighter": tf_for(0.50)}
    out = {k: [] for k in tfs}
    truth = []

    for i, r in enumerate(rows, 1):
        with Image.open(ROOT / "work" / "clean" / r["file"]) as im:
            pil = ImageOps.exif_transpose(im).convert("RGB")
        for k, tf in tfs.items():
            out[k].append(score(model, pil, tf))
        truth.append(1 if r["label"] == "Unhealthy" else 0)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    truth = np.array(truth)
    base = np.array(out["full"])
    print(f"\n{len(rows)} held-out test photos "
          f"({int((truth == 0).sum())} healthy, {int(truth.sum())} infected)\n")
    print(f"{'crop':8s} {'recall':>8s} {'clears':>8s} {'mean |change|':>14s} {'verdicts flipped':>18s}")

    def verdicts(s):
        return np.where(s >= M.T_HIGH, 2, np.where(s < M.T_LOW, 0, 1))

    for k in tfs:
        s = np.array(out[k])
        v = verdicts(s)
        inf, hea = truth == 1, truth == 0
        dec_i, dec_h = inf & (v != 1), hea & (v != 1)
        rec = (v[dec_i] == 2).mean() if dec_i.any() else float("nan")
        clr = (v[dec_h] == 0).mean() if dec_h.any() else float("nan")
        flipped = int((v != verdicts(base)).sum())
        print(f"{k:8s} {rec:8.1%} {clr:8.1%} {np.abs(s - base).mean():14.3f} "
              f"{flipped:18d}")

    print("\nper-class mean score")
    for k in tfs:
        s = np.array(out[k])
        print(f"  {k:8s} healthy {s[truth == 0].mean():.3f}   infected {s[truth == 1].mean():.3f}")


if __name__ == "__main__":
    main()
