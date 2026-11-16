# CWSB Model — Project State

**What this file is:** the shared, factual record of what exists in this project right now.
Both Leela's agent and Charan's agent read this file first, before doing anything else.
It is committed to Git so it travels with every `git pull`.

Last updated: **2026-11-12**

---

## 0. How the two of us work together

| File | Owner | Purpose |
|---|---|---|
| `PROJECT_STATE.md` | both | Facts. What exists, what was measured. Either agent may update it. |
| `NEXT_STEPS.md` | both | What still has to be done, in priority order. |
| `leela_working.md` | Leela's agent only | Dated log of what Leela's side did. Charan's agent reads it, never edits it. |
| `charan_working.md` | Charan's agent only | Dated log of what Charan's side did. Leela's agent reads it, never edits it. |

Rules:

1. **Read before you work.** At the start of every session, read all four files. They are the only memory that survives between machines.
2. **Write after you work.** Append a dated entry to your own `*_working.md` and update `PROJECT_STATE.md` if a fact changed (a new number, a new file, a new result).
3. **Never edit the other person's working file.** If you need to flag something to them, write it in your own file under a `### For <name>` heading.
4. **Never add these four files to `.gitignore`.** They are the handover.
5. **Pull before you start, push when you finish.** Keep the working files conflict-free by only touching your own.

---

## 1. Goal

Detect **coffee White Stem Borer (WSB)** infestation from photographs of coffee plants,
so that a farmer can screen plants with a phone and act before the plant is lost.
Target deployment is a farmer-facing mobile/PWA screening app.

---

## 2. Data — the single most important section

Source folder: `Data_final/` (498 photos, 2.9 GB, **not in Git** — see section 9).

After cleaning and leak-proof splitting (`work/splits/split.csv`):

| Class | Photos | Distinct plants |
|---|---|---|
| Healthy | **28** (18 train / 2 val / 8 test) | **9** |
| Unhealthy (WSB) | **470** (364 train / 53 val / 51 test, 2 excluded) | **69** |
| **Total** | 498 | 78 |

**This imbalance is the project's defining constraint.** The model has seen nine healthy
plants. Validation contains two healthy photos. Almost every weakness documented below
traces back to these two numbers.

Other properties of the dataset that matter:

- **Every photo is a wide whole-plant / canopy shot taken from 2–3 m.** There is not a
  single stem close-up in the training data. The model therefore learned *"does this bush
  look unwell"*, **not** *"does this stem have a borer hole"*. This was confirmed by
  reading the actual images and by Grad-CAM, which puts its heat on foliage, not stems.
- **Labels are plant-level, not photo-level.** Some photos of an infected plant show no
  visible evidence at all (see `plant_012` in section 7).
- There are **no look-alike negatives** — no drought stress, nutrient deficiency,
  shot-hole borer, pruned plants or natural senescence labelled healthy. So anything that
  looks unwell reads as WSB.
- There are **almost no early-stage infestations** — nearly all infected photos show
  advanced decline, which is commercially the least useful stage to detect.

---

## 3. Pipeline — what has been built (Phases 0–4 complete)

All scripts live in `src/`. They run in this order.

| Phase | Script | What it does |
|---|---|---|
| 1a | `src/clean.py` | Reads `Data_final/` read-only → writes `work/clean/`. EXIF-rotates, strips GPS metadata, re-saves JPEG q=95. Computes blur (Laplacian variance), clipped-pixel and dark-pixel fractions. Deduplicates by SHA-256 and perceptual dhash. Outputs `images.csv`, `duplicates.csv`. Gate constants: `BLUR_MIN=60.0`, `CLIPPED_MAX=0.20`, `DARK_MAX=0.40`, `NEAR_DUP_HAMMING=6`. |
| 1b | `src/make_split.py` | Groups photos into **pseudo-plants** by filename frame gap (4) and EXIF time gap (120 s), then splits 70/15/15 **by plant, never by photo**, `SEED=42`. Refuses to finish if any plant appears in two splits. Output: `work/splits/split.csv`. |
| 2a | `src/make_tiles.py` | Multi-crop augmentation for the starved healthy class: 3×2 grid + centre = 7 tiles per training healthy photo, tile side 55% of the short edge. Output: `work/tiles/Healthy/`, `tiles.csv`. |
| 2b | `src/train.py` | Trains and compares two contenders. See section 4. |
| 3 | `src/set_thresholds.py` | Fixes the decision thresholds and writes `models/v1/config.json`. |
| 4 | `src/make_report_pdf.py` | Grad-CAM evidence PDFs. Flags: `--split {all,test,val}`, `--real-only`, `--dpi`. |

Supporting scripts:

| Script | Purpose |
|---|---|
| `src/predict_folder.py` | Score any folder of photos the way the production API will (quality gate → model → thresholds). `--pdf` adds heatmaps. |
| `src/analyze_failures.py` | Scores val + test2, lists every failure with its category. This is what exposed the threshold bug. |
| `src/ood_guard.py` | Out-of-distribution guard by cosine similarity in 768-d feature space. **Built, measured, found insufficient** — see section 7. |
| `src/crop_experiment.py` | Measures how much the frame periphery drives the decision. Result in section 8. |
| `src/package_for_colab.py` | Resizes to short side 768 and zips for upload (199 MB, 624 images). |
| `src/shrink_for_training.py` | In-place resize to 768 for the all-on-Colab route. |

Colab notebooks (`colab/`):

- `CWSB_full_pipeline_colab.ipynb` — 13 cells, embeds every script via `%%writefile`, takes raw `Data_final.zip`. **Use this one.**
- `CWSB_train_colab.ipynb` — 5 cells, needs `cwsb_package.zip` produced locally first.

**All training runs on Google Colab GPU (T4).** The local Windows machine is CPU-only and is
used for scoring, reports and experiments, not training.

---

## 4. The model

Two contenders were trained and compared by validation PR-AUC:

- **Contender A** — frozen ConvNeXt backbone used as a feature extractor, with
  `LogisticRegression(class_weight="balanced", max_iter=5000, C=0.1)` on top (~1,500 trainable
  parameters). File: `models/v1/contender_A.pt` (11 KB).
- **Contender B** — full fine-tune. 2 warm-up epochs (head lr 1e-3), then full fine-tune at
  lr 1e-4, AdamW, weight decay 0.05, cosine schedule, batch 16, ≤40 epochs, patience 6.
  File: `models/v1/contender_B.pt` (106 MB).

**Winner: B** (`models/v1/training_result.json`, `val_pr_auc: 1.0`).

> The PR-AUC of exactly 1.0 is **not** a sign of a great model. It is a symptom of having
> two healthy photos in validation. Treat it as meaningless.

Backbone: `convnext_tiny.fb_in22k_ft_in1k_384` (timm), ~28M parameters,
stages 3-3-9-3, widths 96→768, final 12×12 feature grid, input 384×384.

Class imbalance handling: `WeightedRandomSampler` + class-weighted cross-entropy + the
multi-crop tiling from Phase 2a.

Transforms:

```python
train_tf = Compose([RandomResizedCrop(384, scale=(0.6,1.0)), RandomHorizontalFlip(),
  RandomRotation(10), ColorJitter(brightness=.25, contrast=.25, saturation=.10),
  RandomApply([GaussianBlur(5)], p=0.15), ToTensor(), Normalize(MEAN,STD)])
eval_tf  = Compose([Resize(416), CenterCrop(384), ToTensor(), Normalize(MEAN,STD)])
```

---

## 5. Decision thresholds

From `models/v1/config.json`:

```
score <  0.50          -> no_visible_signs
0.50 <= score < 0.90   -> inconclusive  (ask the farmer to retake)
score >= 0.90          -> wsb_likely
```

Plant score = **max** over that plant's photos.

**Why these numbers:** validation healthy photos score 0.26–0.39, so `T_low = 0.50` sits
just above the healthy band. `T_high = 0.90` marks strong confidence.

**The bug this replaced (important history):** both thresholds were originally derived from
quantiles of the *infected* score distribution. Because contender B early-stopped on a
PR-AUC of 1.0, infected validation scores all clustered above 0.99, so
`quantile(0.02) = 0.9071` and `quantile(0.10) = 0.9149` — a band 0.008 wide. 14 of 18
observed misses were caused by this alone. Fixing the thresholds, with identical model
weights, moved recall from ~86% to 93.6%.

---

## 6. Measured results

Held-out test split, photo level (`models/v1/config.json`):

| | Count |
|---|---|
| Caught | 44 |
| Missed | 3 |
| False alarms | 1 |
| Cleared | 7 |
| Inconclusive | 4 |

Same test split, plant level (`models/v1/config.json`, plant score = max over photos):
5 caught, 0 missed, 1 false alarm, 0 cleared, 2 inconclusive (8 plants).

Headline: **accuracy 91.5%, recall 91.7%** (as reported on the test2 run at 768 px).
On full-resolution originals the same test gives **95.9% recall**.

> With 59 test photos the margin of error is roughly **±7 points**. The true accuracy could
> be 85% or 97% and we cannot tell. Do not quote 91.5% as a precise figure.

---

## 7. Known failures and their root causes

1. **All remaining misses come from one plant, `plant_012`** (`CB_2_DSC01024`, `01027`,
   `01030`). The photos are a blurry canopy with no visible stem damage or decline. This is
   a plant-level label with no photo-level evidence. Needs expert review — it may be
   mislabelled, or simply unphotographed evidence.

2. **Both false alarms come from `plant_003`.** `CB_DSC_0285` is an estate-wide landscape
   shot, not a single-plant photo. A framing gate would reject it.

3. **The model fails catastrophically on out-of-framing photos.** 19 unique close-up branch
   photos of *uninfested* plants (in `test_new/`) all returned "WSB SIGNS LIKELY" at
   0.98–1.00, with Grad-CAM heat on ordinary dried leaves. The classifier has no way to say
   "this is not my kind of photo" — it must answer healthy or infected for any image.

4. **The OOD guard does not solve #3.** `src/ood_guard.py` catches only 5 of those 19,
   because the similarity distributions overlap (new photos 0.51–0.85 vs in-distribution
   0.40–0.94, median 0.79). A **supervised framing gate** is needed instead.

---

## 8. Experiments run, and what they showed

**Crop experiment** (`src/crop_experiment.py`) — scores the 59 test photos at full frame,
centre 70% and centre 50%:

| Crop | Mean score healthy | Mean score infected | Recall | Correct clears |
|---|---|---|---|---|
| Full | 0.393 | 0.950 | 95.9% | 87.5% |
| 70% | 0.236 | 0.914 | 89.4% | 87.5% |
| 50% | 0.132 | 0.878 | 87.5% | 100% |

Reading: cropping pushes **all** scores down rather than separating the classes. It shifts
the operating point (which the thresholds already control); it does not improve
discrimination. The surroundings genuinely carry signal the model uses. Note also that
part of the loss is train/serve mismatch — the model was trained on full frames.

**Resolution:** full-resolution originals give 95.9% recall vs 93.6% at 768 px short side.
Downscaling costs about two points. Do not shrink aggressively for speed in the API.

**Lighting robustness:** only 1 of 30 brightness-stressed copies changes verdict. Good.

---

## 9. Locked decisions

- **No image filters, no background removal.** A preprocessing stage must behave identically
  in training and serving or it introduces a mismatch; in a dense plantation canopy there is
  no clean figure/ground to segment, and the segmenter's own errors become noise. The 2019
  reference paper measured raw 0.84 vs preprocessed 0.87 — essentially no gain.
- **Split by plant, never by photo.** `make_split.py` aborts on leakage.
- **Seed 42 everywhere** for reproducibility.
- **Weather / temperature / humidity are NOT model inputs.** They are real WSB risk factors,
  but with 69 infected plants collected in a narrow window the model would learn
  "this month means infected". Use them as a **separate, displayed risk score**, not fused
  into the CNN.
- **Training happens on Colab GPU only.**

---

## 10. Repository layout

```
coffee_Cswb_Model/
├── PROJECT_STATE.md        <- this file           (in Git)
├── NEXT_STEPS.md           <- the roadmap         (in Git)
├── leela_working.md        <- Leela's log         (in Git)
├── charan_working.md       <- Charan's log        (in Git)
├── README.md                                      (in Git)
├── src/            12 scripts, 0.1 MB             (in Git)
├── colab/          2 notebooks                    (in Git)
├── reports/        CSVs in Git. PDFs are in .gitignore, but the 3 existing PDFs
│                   (94 MB) were committed earlier and are still tracked
├── Data_final/     498 photos, 2.9 GB             (NOT in Git)
├── work/           clean/, splits/, tiles/ 2.6 GB (NOT in Git — regenerable)
├── models/v1/      contender_A.pt, contender_B.pt (106 MB), config.json,
│                   training_result.json, ood_reference.npz   (NOT in Git)
├── test/           12 blind samples + answer_key.csv, 67 MB  (NOT in Git)
├── test2/          144 images + answer_key2.csv, 54 MB       (NOT in Git)
└── test_new/       30 user photos, 152 MB                    (NOT in Git)
```

Anything marked NOT in Git has to be transferred by pendrive or cloud drive. See the
transfer list at the bottom of `NEXT_STEPS.md`.
