"""Phase 4 - build the PDF evidence report for every test2 image.

Runs the winning model locally, computes a Grad-CAM heatmap per image
(implemented here, no extra package), applies the frozen thresholds from
models/v1/config.json, and writes reports/cwsb_test_report.pdf:

  page 1      summary - accuracy, recall, breakdown by category
  then        every image that failed or came back inconclusive
  then        every correct image, in filename order

Each image page shows the photo as the model saw it, the heatmap of what
drove the decision, the score, the verdict and the truth.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "models" / "v1" / "config.json").read_text())
T_LOW, T_HIGH = CFG["T_low"], CFG["T_high"]
IMG = CFG["img"]

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_tf = transforms.Compose([
    transforms.Resize(IMG + 32), transforms.CenterCrop(IMG),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
disp_tf = transforms.Compose([transforms.Resize(IMG + 32), transforms.CenterCrop(IMG)])


def load_model():
    ck = torch.load(ROOT / "models" / "v1" / "contender_B.pt",
                    map_location="cpu", weights_only=False)
    m = timm.create_model(ck["backbone"], pretrained=False, num_classes=2)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def score_and_cam(model, pil):
    """Forward + backward once: returns (prob_infected, 384x384 heatmap 0-1)."""
    store = {}

    def grab(module, inp, out):
        out.retain_grad()          # keep the gradient on this non-leaf tensor
        store["a"] = out
        # must return None: a returned value would replace the layer's output

    handle = model.stages[-1].register_forward_hook(grab)
    x = eval_tf(pil).unsqueeze(0)
    logits = model(x)
    prob = torch.softmax(logits, 1)[0, 1].item()
    model.zero_grad()
    logits[0, 1].backward()
    handle.remove()

    a = store["a"]                                   # [1,C,H,W]
    g = a.grad                                       # same shape
    w = g.mean(dim=(2, 3), keepdim=True)             # importance per channel
    cam = F.relu((w * a).sum(1, keepdim=True))       # weighted sum, keep positives
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    rng = cam.max() - cam.min()
    return prob, (cam - cam.min()) / rng if rng > 1e-9 else np.zeros_like(cam)


def verdict_of(s):
    if s >= T_HIGH:
        return "WSB SIGNS LIKELY"
    if s < T_LOW:
        return "No visible WSB signs"
    return "INCONCLUSIVE - retake"


def page_for(pdf, rec, disp, cam, dpi=90):
    col = {"CORRECT": "#2e7d32", "WRONG": "#b3261e", "inconclusive": "#a05a00"}[rec["mark"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    ax[0].imshow(disp); ax[0].set_title("photo as the model saw it", fontsize=10)
    ax[1].imshow(disp); ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title("what drove the decision (red = most)", fontsize=10)
    for a in ax:
        a.axis("off")
    sim = "  [brightness-modified copy]" if rec["simulated"] else ""
    fig.suptitle(
        f"{rec['sample']}   true: {rec['truth']}   score {rec['score']:.3f}  ->  "
        f"{rec['verdict']}   [{rec['mark']}]{sim}\n"
        f"lighting: {rec['lighting']}    camera: {rec['source']}    "
        f"split: {rec['split']}    plant: {rec['plant']}",
        color=col, fontsize=11)
    pdf.savefig(fig, dpi=dpi); plt.close(fig)


def summary_page(pdf, recs, scope):
    real = [r for r in recs if not r["simulated"]]
    dec = [r for r in real if r["mark"] != "inconclusive"]
    acc = sum(r["mark"] == "CORRECT" for r in dec) / max(len(dec), 1)
    inf = [r for r in dec if r["truth"] == "Unhealthy"]
    rec_ = sum(r["verdict"].startswith("WSB") for r in inf) / max(len(inf), 1)
    hea = [r for r in dec if r["truth"] == "Healthy"]
    spec = sum(not r["verdict"].startswith("WSB") for r in hea) / max(len(hea), 1)

    plants = {r["plant"] for r in real}
    hp = {r["plant"] for r in real if r["truth"] == "Healthy"}
    L = ["CWSB SCREENING MODEL - TEST REPORT", "",
         f"Scope          : {scope}",
         f"Model          : contender_B (ConvNeXt-Tiny, fine-tuned)",
         f"Thresholds     : no signs < {T_LOW}   |   inconclusive   |   >= {T_HIGH} WSB likely",
         f"Images tested  : {len(recs)}  ({len(real)} real photos" +
         (f" + {len(recs) - len(real)} brightness-modified copies)" if len(recs) > len(real) else ")"),
         f"Distinct plants: {len(plants)}  ({len(hp)} healthy, {len(plants) - len(hp)} infested)",
         "",
         "None of these photos were used to train the model.", "",
         "--- RESULTS ON THE REAL PHOTOS ---", "",
         f"  Recall on infected     : {rec_:5.1%}   (infested plants correctly flagged)",
         f"  Correct clears         : {spec:5.1%}   (healthy plants correctly cleared)",
         f"  Accuracy where decided : {acc:5.1%}",
         f"  Sent to retake         : {sum(r['mark'] == 'inconclusive' for r in real)} of {len(real)}",
         f"  Wrong                  : {sum(r['mark'] == 'WRONG' for r in real)} of {len(real)}",
         ""]

    for field, title in (("split", "By data split"), ("lighting", "By lighting"),
                         ("source", "By camera"), ("truth", "By true class")):
        L.append(f"--- {title.upper()} ---")
        for v in sorted({r[field] for r in recs}):
            sub = [r for r in recs if r[field] == v and r["mark"] != "inconclusive"]
            if sub:
                a = sum(r["mark"] == "CORRECT" for r in sub) / len(sub)
                L.append(f"  {v:18s} {a:6.1%} correct   ({len(sub)} decided)")
        L.append("")

    wrong = [r for r in recs if r["mark"] == "WRONG"]
    L.append(f"--- THE {len(wrong)} FAILURES (pages follow) ---")
    for r in sorted(wrong, key=lambda r: r["score"]):
        kind = "false alarm" if r["truth"] == "Healthy" else "missed infestation"
        L.append(f"  {r['sample']}  score {r['score']:.3f}  {kind:18s} plant {r['plant']}")

    fig = plt.figure(figsize=(10, 5.4 * max(1, len(L) / 36)))
    fig.text(0.04, 0.97, "\n".join(L), va="top", family="monospace", fontsize=8.5)
    plt.axis("off"); pdf.savefig(fig, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["all", "test", "val"], default="all",
                    help="which data split to report on (test = the held-out report card)")
    ap.add_argument("--real-only", action="store_true",
                    help="exclude the brightness-modified stress copies")
    ap.add_argument("--dpi", type=int, default=90)
    args = ap.parse_args()

    model = load_model()
    key = {r["sample"]: r for r in csv.DictReader(open(ROOT / "test2" / "answer_key2.csv"))}

    def wanted(p):
        k = key[p.name]
        if args.split != "all" and k["split"] != args.split:
            return False
        if args.real_only and k["original"].endswith("(simulated)"):
            return False
        return True

    paths = [p for p in sorted((ROOT / "test2").glob("t2_*.jpg")) if wanted(p)]
    scope = {"all": "validation + held-out test photos",
             "test": "HELD-OUT TEST SPLIT ONLY - never used for training or tuning",
             "val": "validation split only"}[args.split]
    if args.real_only:
        scope += " (real photos only, no simulated copies)"
    name = "cwsb_test_report" + ("" if args.split == "all" else f"_{args.split}") + \
           ("_real" if args.real_only else "") + ".pdf"
    OUT = ROOT / "reports" / name
    print(f"thresholds T_low={T_LOW} T_high={T_HIGH}; {len(paths)} images; scope: {args.split}")

    recs, cache = [], {}
    for i, p in enumerate(paths, 1):
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        s, cam = score_and_cam(model, pil)
        k = key[p.name]
        v = verdict_of(s)
        truth = k["true_label"]
        mark = ("inconclusive" if v.startswith("INCONCL") else
                "CORRECT" if (v.startswith("WSB") == (truth == "Unhealthy")) else "WRONG")
        recs.append({"sample": p.name, "truth": truth, "score": s, "verdict": v,
                     "mark": mark, "lighting": k["lighting"], "source": k["source"],
                     "split": k["split"], "plant": k["plant"],
                     "simulated": k["original"].endswith("(simulated)")})
        cache[p.name] = (np.array(disp_tf(pil)), cam)
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    order = ([r for r in recs if r["mark"] == "WRONG"] +
             [r for r in recs if r["mark"] == "inconclusive"] +
             [r for r in recs if r["mark"] == "CORRECT"])

    with PdfPages(OUT) as pdf:
        summary_page(pdf, recs, scope)
        for r in order:
            disp, cam = cache[r["sample"]]
            page_for(pdf, r, disp, cam, args.dpi)

    with open(ROOT / "reports" / name.replace(".pdf", "_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    print("\n", dict(Counter(r["mark"] for r in recs)))
    print(f"-> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(recs) + 1} pages)")


if __name__ == "__main__":
    main()
