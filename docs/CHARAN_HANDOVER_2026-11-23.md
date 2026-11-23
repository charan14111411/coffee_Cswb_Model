# CWSB handover from Charan — state on 23 Nov 2026

This file is for Leela and Leela's agent. It records everything done on Charan's
machine between 20 and 23 Nov 2026, so the work can continue from here without
asking Charan. Read `docs/V1_HANDOVER.md` first if you do not know the project;
this file only covers what came after it.

**Branch:** `Charan_dev_23/11` (pushed). Last commit `b7a4ece`.
**Not committed:** this file. `charan_working.md` has no entry for this work yet.

---

## 1. Summary

| Item | State |
|---|---|
| v1 on Charan's machine | Verified. All 7 file fingerprints match, all committed numbers reproduce. |
| v2 retrain (normal camera) | Done on Colab. **Not better than v1. Keep v1 in use.** |
| Retrain notebook | Changed to read one upload folder or zip from Drive. |
| Microscopic data | 50 images received. They are crops of normal photos, not usable for training. |
| Microscopic notebooks | Two new notebooks written and tested locally on fake data. Not yet run on Colab. |
| Data request to the field team | Drafted, numbers not yet agreed with the manager. |

---

## 2. v1 verification

- `models/v1/` was replaced from the pendrive. SHA-256 prefixes match for all
  seven files (`cwsb_v1.pt` / `contender_B.pt` = `f63f0c25a012006f…`).
- Reproduced locally:
  - held-out test split: 44 caught / 3 missed / 1 false alarm / 7 cleared / 4 retake.
    This reproduces only on the 768 px copies in `test2/`. At full resolution
    the same photos give 47 / 2 / 1 / 7 / 2.
  - `images_health_16_11_v2`: 198 / 9 / 4.
  - `test_new/` (30 healthy close-ups): 30 of 30 scored "WSB likely".
  - `measure.py`: 78 / 42 / 92 / 4.

**Open correction (Charan's side):** `charan_working.md` on branch
`origin/charan_20/11`, lines 326 and 359, says the `contender_B.pt` fingerprint
differs between machines. That is wrong: an MD5 was compared with a SHA-256
prefix. The files are identical. The line has not been corrected yet.

---

## 3. v2 retrain (normal camera)

### What was run

- **Notebook:** `colab/CWSB_retrain_end2end_colab.ipynb`, Tesla T4.
- **Upload:** `MyDrive/cwsb_data/COLLAB_UPLOAD_FOLDER.zip` with
  `old_data/{Healthy, Unhealthy(Infected)}`, `new_data/{healthy, infected}`,
  `old_models/cwsb_v1.pt`.
- **Data:** 28 healthy + 470 infested old photos, plus 485 new infested files.
  The 485 are Roboflow exports: 211 unique images, which are augmentations of
  47 source photos. No new healthy photos.
- **Merged:** Healthy 28, Unhealthy 681.
- **Split:** train 450, val 94, test 100, excluded 65.
- **Result:** best epoch 7, validation PR-AUC 0.99977. Thresholds stayed at the
  default 0.50 / 0.90 because validation had only 2 healthy photos. ONNX parity
  2.24e-07.

### Changes made to the retrain notebook

- Cell 0 and Cell 2 rewritten for a single upload (Drive folder or zip).
- Roboflow file names (`vf_10_jpg.rf.<hash>.jpg`) are renamed to
  `rfsrc{N}_{kkk}`, so all versions of one source photo form one plant in the
  split.
- Byte-identical duplicates are dropped.
- The embedded pipeline scripts are still byte-identical to `src/`.

### Where the outputs are

`models/v2/` on Charan's machine (gitignored, share by pendrive):
`cwsb_v2.pt` (SHA-256 `8db2ae59f95f58cd…`), `cwsb_v2.onnx`, `config_v2.json`,
`split_v2.csv`, `training_result_v2.json`, `cwsb_v2_test_report.pdf` (101 pages).

### v1 against v2

Counts are caught / missed / false alarm / cleared / retake.

| Test set | v1 | v2 | Note |
|---|---|---|---|
| v2's test split (100) | 88 / 2 / 1 / 7 / 2 | 77 / 7 / 0 / 7 / 9 | Unfair to v2: all 75 old infested test photos were v1 train or val photos. |
| `test_new/`, 30 healthy close-ups | 30 false alarms | 30 false alarms | No change. |
| 17 new infested close-ups | 17 caught | 17 caught | No change. |
| `test/`, 12 blind samples | 7 caught, 4 cleared, 1 false alarm | 5 caught, 4 cleared, 3 retake | |
| `test2/`, 114 real photos | 90 / 3 / 1 / 9 / 11 | 97 / 3 / 0 / 6 / 8 | Favours v2: it trained on 58 of them and validated on 46. |

**Verdict: keep v1.** v2 is not shown to be better, and the main fault (every
healthy close-up flagged as infested) is unchanged. All seven of v2's misses on
its test split are from the `IMG_20170725` series.

### Known faults in the normal-camera pipeline (found, not fixed)

1. **`src/clean.py` enlarges small photos before the blur check.** It always
   resizes to 1024 px wide. 63 of the 211 new 640 px photos were thrown out as
   "blurry" for this reason. Fix: shrink only, never enlarge. `src/` belongs to
   Leela, so it was left alone.
2. **The retrain notebook re-splits everything each time.** Old test photos can
   become training photos, so the old-vs-new table is not a fair comparison.
   Fix: reuse the previous split file for old photos. The microscopic retrain
   notebook already does this (section 5).
3. **Docs out of date.** `docs/V1_HANDOVER.md` and
   `docs/CHARAN_REFERENCE_NOTES.md` still describe the three-zip upload.

---

## 4. Microscopic data

**Proposal from the team:** stage 1 is the normal camera (v1). When stage 1 is
inconclusive, stage 2 uses a microscopic camera to confirm. Stage 2 needs its
own model; v1 or v2 cannot be reused for it.

**What was received:** `Microscopic_level_data/` (gitignored), 50 PNG files.

| Symptom, from the file name | Images |
|---|---|
| Holes | 39 |
| Ridges | 9 |
| Bulging | 1 |
| Crack | 1 |

**Why it cannot be used for training:**

- The files are crops of normal photos, each a different size (67×56 up to
  1222×632 px), most of them blurry from enlargement. They are not microscope
  captures.
- All 50 show damage. There are no healthy or look-alike examples.
- No plant IDs, so a split by plant is not possible.

**What the field team was told** (message drafted, sent by Charan): send
original camera files; include healthy images taken the same way; give every
coffee plant a number (`P001`, `P002` …) in the file name; say which device and
magnification was used; send a trial of 10 plants first.

**Still unanswered by the team:**

- Were the 50 images taken with the microscopic camera or cropped from phone
  photos?
- Which device and magnification will be used?
- Is the goal one score per image (classification), or finding and counting
  holes and ridges (detection)? Detection needs box labels and a different
  pipeline.

---

## 5. Microscopic notebooks (new)

- `colab/cwsb_train_microscopic.ipynb` builds the first model, `cwsb_micro_v1`,
  from `new_data/` only.
- `colab/cwsb_retrain_microscopic.ipynb` builds `cwsb_micro_v2` onwards from
  `old_data/` + `new_data/`, and compares with the model in `old_models/`.

Both are self-contained. The pipeline scripts are embedded in the notebook
(`clean_micro.py`, `split_micro.py`, `train_micro.py`) and do not exist in
`src/`. Nothing in `src/` was changed.

### Upload layout

Default path: `MyDrive/cwsb_data/MICRO_UPLOAD_FOLDER.zip` (folder or zip). Set
`UPLOAD_PATH` in Cell 1.

```
MICRO_UPLOAD_FOLDER/
├── old_data/            retrain only
│   ├── healthy/
│   └── infested/
├── new_data/
│   ├── healthy/
│   └── infested/
└── old_models/          retrain only
    ├── cwsb_micro_v1.pt
    ├── cwsb_micro_v1.onnx
    └── split_micro_v1.csv
```

A plant ID is needed on every image, either as a folder (`healthy/P001/…`) or
at the start of the file name (`P001_hole_1.jpg`).

### What is the same as v1

ConvNeXt-Tiny `convnext_tiny.fb_in22k_ft_in1k_384`, 384 px, 2 classes, contender
B recipe, seed 42, selection by validation PR-AUC, thresholds from healthy
validation scores, Grad-CAM PDF, ONNX input `rgb_384` with outputs
`probabilities` and `feature_map`.

### What is different

| Step | Normal-camera notebook | Microscopic notebooks |
|---|---|---|
| Plant grouping | Guessed from frame number and time gap | Plant ID from folder or file name |
| Quality check | Excludes flagged photos; enlarges before blur check | Reports only by default (`QUALITY_GATE = 'report'`); never enlarges; images under 224 px always left out |
| Healthy tiles | 7 per healthy training photo | None |
| Augmentation | Horizontal flip | Horizontal and vertical flip |
| Retrain split | Everything re-split | Old images keep their set, matched by SHA-256 against `split_micro_vN.csv` |
| Test report | Per image | Per image and per plant (highest score of the plant) |
| Same file in both class folders | Not checked | Left out and listed |
| Training log | Epoch lines not shown in Colab | One line per epoch; history saved in `training_result` |

Retraining starts from ImageNet weights on old + new data together. The old
`.pt` is only used for the comparison table.

### Release zip contents

`cwsb_micro_v{N}.pt`, `.onnx`, `config_micro_v{N}.json`,
`training_result_micro_v{N}.json`, `split_micro_v{N}.csv`,
`quality_micro_v{N}.csv`, `cwsb_micro_v{N}_test_report.pdf`.

### How it was tested

Locally, on CPU, with fake data: 512 px crops of the normal photos, 16 plants
for the first run and 8 more for the retrain run, 3 epochs and 1 epoch. Every
cell ran in both notebooks, covering zip and folder upload, duplicates, missing
plant IDs, a broken file, an undersized image, the split, training, thresholds,
PDF, ONNX parity (4.5e-07) and packaging. In the retrain run all 49 old images
kept their previous set.

**Not tested:** a Colab GPU run, and real microscope images. The scores from the
fake-data runs say nothing about real accuracy. The quality limits
(`BLUR_MIN 60`, `CLIPPED_MAX 0.20`, `DARK_MAX 0.40`) are the phone-photo values
and must be re-set from the first real batch.

---

## 6. Notebooks in `colab/` and when to use each

| Notebook | Camera | Use |
|---|---|---|
| `CWSB_full_pipeline_colab.ipynb` | Normal | Train from a fresh dataset; zips with `healthy/` and `unhealthy/`. Not run by Charan. |
| `CWSB_retrain_end2end_colab.ipynb` | Normal | Old + new data. Used for v2. |
| `cwsb_train_microscopic.ipynb` | Microscopic | First microscopic model. |
| `cwsb_retrain_microscopic.ipynb` | Microscopic | Later microscopic versions. |
| `CWSB_version_rbn_colab.ipynb` | Normal | Charan's `version_rbn` work. Not touched in this period. |

---

## 7. Data requested from the field

These are Charan's recommendations. **None of the counts is agreed with the
manager yet.**

**Normal camera**

- 300 healthy and 300 infested plants, 3 photos each: whole plant from 2–3 m,
  full stem from about 1 m, stem close-up from about 30 cm. 1,800 photos.
- About 75 of the healthy plants should be look-alikes (unwell for other
  reasons), and about 100 of the infested plants early-stage.
- Both classes photographed the same way: same device, same three shots, same
  estates and days. Every label confirmed by stem inspection.
- First milestone: 150 + 150 plants, then train and measure.

**Microscopic camera**

- Pilot of 150 infested and 150 non-infested plants, 3 to 5 images each, same
  device and magnification, with look-alike negatives.
- Collected on the same field visits as the normal-camera photos.

**Why this is needed:** today's data is 28 healthy photos from 9 plants against
470 infested, and 268 of the infested photos come from a camera that never
photographed a healthy plant. v1 leans on framing and background: with the
centre of the photo blacked out, 78% of infested photos are still called
infested.

---

## 8. What to do next, in order

1. **Wait for the field data.** No further model work is useful until healthy
   photos taken the same way as infested ones arrive.
2. **When the 10-plant microscopic trial arrives:** run
   `cwsb_train_microscopic.ipynb` once just to read Cell 3's quality table, set
   the limits in Cell 1, and check that the plant IDs are picked up.
3. **Decide classification or detection** for the microscopic stage, with the
   agronomy side.
4. **If normal-camera photos arrive with plant IDs:** build a normal-camera
   train/retrain pair in the microscopic style (plant ID split, kept previous
   split, no enlarging in the quality check). Charan offered this; it is not
   started.
5. **Fix the two normal-camera pipeline faults** in section 3 before the next
   normal-camera retrain. This needs Leela's agreement because it touches `src/`.
6. **Housekeeping on Charan's side:** correct the fingerprint lines in
   `charan_working.md`; add a log entry for the v2 run and the microscopic
   notebooks; update the two out-of-date docs; delete the redundant
   `colab_upload/` zips (about 3.3 GB, gitignored).

---

## 9. Rules that still apply

- v1 and its retraining belong to Leela's handover document. `version_rbn/`,
  `colab/CWSB_version_rbn_colab.ipynb` and `models/version_rbn/` belong to
  Charan.
- Never edit the other person's working file.
- Photos, models and zips are gitignored and move by pendrive.
- Never overwrite a previous model version's folder.
