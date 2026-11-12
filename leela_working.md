# Leela — Working Log

**Owner: Leela.** Charan's agent reads this file but must never edit it.
Leela's agent appends a dated entry here at the end of every working session.
Charan's equivalent is `charan_working.md`.

For facts about the project, see `PROJECT_STATE.md`. For the roadmap, see `NEXT_STEPS.md`.
This file is the *narrative*: what was tried, what was learned, what was abandoned and why.

---

## 2026-11-12 — Handover prepared

Created `PROJECT_STATE.md`, `NEXT_STEPS.md` and this file, and made sure none of them are
in `.gitignore`, so Charan receives them on `git pull`.

---

## Up to 2026-11-12 — Everything built so far

### Phases 0–2: data pipeline and training

Built `src/clean.py`, `src/make_split.py`, `src/make_tiles.py`, `src/train.py`.
Cleaning strips EXIF GPS (the photos carried plantation coordinates), re-saves at q=95,
and deduplicates by SHA-256 plus perceptual dhash.

Splitting groups photos into **pseudo-plants** by filename frame gap and EXIF timestamp gap,
then splits 70/15/15 **by plant**. The script aborts if any plant lands in two splits —
splitting by photo would have leaked the same plant into train and test and inflated every
number we report.

Trained on **Colab T4 GPU** (local machine is CPU-only). Two contenders compared by
validation PR-AUC: A (frozen backbone + logistic regression, ~1,500 params) and B (full
fine-tune of ConvNeXt-Tiny). **B won.**

### Phase 3: thresholds — and the bug that mattered most

Originally derived both thresholds from quantiles of the *infected* score distribution.
Because B early-stopped at PR-AUC 1.0, infected validation scores clustered above 0.99, so
the two thresholds collapsed to 0.9071 and 0.9149 — a decision band 0.008 wide. Hedged
scores were being reported as "no visible signs".

`src/analyze_failures.py` was written specifically to find this, and did: **14 of 18 observed
misses were caused by threshold placement alone, not by the model.** Replacing them with
`T_low = 0.50`, `T_high = 0.90` moved recall from ~86% to 93.6% **with identical weights**.

Lesson worth keeping: when a model looks broken, check the decision rule before blaming the
weights.

### Phase 4: evidence PDFs

`src/make_report_pdf.py` produces Grad-CAM reports so the team can see *where* the model
looked. Grad-CAM is hand-implemented: a forward hook on the last ConvNeXt stage that calls
`out.retain_grad()`, then gradients read back from `a.grad`.

Two bugs hit here, both worth remembering:

- `register_full_backward_hook` returning `go[0]` → `RuntimeError: hook has changed the size
  of value`. PyTorch treats a returned value as a *replacement* gradient. Fixed by dropping
  the backward hook entirely.
- A forward hook must return `None`. Returning anything replaces the layer's output.

### The big realisation

Reading the actual images, **every photo in the dataset is a wide whole-plant shot from
2–3 m. Not one is a stem close-up.** The model therefore reads whole-plant condition —
canopy density, colour, dieback — not bore holes.

I had earlier interpreted Grad-CAM heat on foliage as the model "grasping at noise" and
**corrected that**: heat on foliage *is* the real signal, because foliage is the only
evidence present in the training data.

This was confirmed catastrophically: 19 unique close-up branch photos of uninfested plants
(`test_new/`) all returned "WSB SIGNS LIKELY" at 0.98–1.00, with heat on ordinary dried
leaves. The model is a whole-plant-condition reader being asked to judge a close-up.

### Things tried and abandoned, with reasons

- **OOD guard** (`src/ood_guard.py`) — cosine similarity to the K=5 nearest training photos
  in 768-d feature space. Built it, fitted it, measured it honestly: it catches only **5 of
  the 19** bad photos, because the distributions overlap (new 0.51–0.85 vs in-distribution
  0.40–0.94, median 0.79). **Abandoned.** A *supervised* framing gate is the right answer
  and is now Priority 2 in `NEXT_STEPS.md`.
- **Background removal** — investigated after the question was raised. Rejected: in a dense
  plantation canopy the subject plant is entangled with neighbouring coffee, so there is no
  figure/ground for rembg/SAM-style tools to find; their errors would become noise; and the
  canopy context is precisely the signal the model uses.
- **Centre cropping** — measured rather than guessed, via `src/crop_experiment.py`. Cropping
  pushes *all* scores down (healthy 0.393→0.132, infected 0.950→0.878) rather than separating
  the classes. Recall falls 95.9%→87.5% while clears rise 87.5%→100%. That is an operating-point
  shift the thresholds already control, not better discrimination. Not adopted as an
  inference-time filter; a trained bounding-box crop remains a V2 experiment.

### Smaller findings

- Full-resolution scoring gives **95.9%** recall vs **93.6%** at 768 px short side.
  Downscaling costs ~2 points — relevant to how the API sizes its input.
- Lighting robustness is good: only 1 of 30 brightness-stressed copies changes verdict.
- Every remaining failure traces to exactly two plants: `plant_012` (all misses — blurry
  canopy with no visible evidence) and `plant_003` (both false alarms — `CB_DSC_0285` is an
  estate-wide landscape shot).

### Operational notes from the Colab runs

- `nvidia-smi: command not found` means the session is on CPU despite the runtime dialog.
  Fix: Runtime → Change runtime type → T4 GPU → Save. **This wipes session files.**
- Use `unzip -oq` everywhere. The interactive replace prompt rejects lowercase `a`; it wants
  capital `A`.
- First packaging attempt produced a **2,765 MB** zip (ZIP_STORED, full resolution).
  `package_for_colab.py` now resizes the short side to 768 → **199 MB**.
- Colab output unzipped nested as `models/v1/models_v1/models/v1/`. Flatten it after download.

---

### For Charan

Three things I would value your side picking up, because they are independent of my work:

1. **The framing gate** (Priority 2). It needs wrong-framing photos, which need no agronomy
   knowledge to collect, and it is the highest-value fix we have identified.
2. **Look-alike negatives** (Priority 1) — stressed-but-uninfested plants. We have none, and
   their absence is why healthy close-ups score 0.99.
3. Please **do not re-derive the thresholds** from infected-score quantiles. That is the bug
   documented above. If you retrain, derive `T_low` from the *healthy* score distribution.

Please log whatever you do in `charan_working.md` so I pick it up on pull.
