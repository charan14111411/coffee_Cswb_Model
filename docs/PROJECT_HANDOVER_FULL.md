# CWSB project — complete handover

**Written:** 23 Nov 2026, on Charan's machine.
**Branch:** `Charan_dev_23/11`, last commit `b7a4ece`.
**For:** any engineer or coding agent who has never seen this project and has to
continue it.

This file is self-contained. It explains what the application is, what has been
built, what was measured, what went wrong, and what to do next. Where a detail
lives in another file, the file is named.

---

## Contents

1. What the application is
2. People, ownership and working rules
3. Repository layout, and what is not in Git
4. The data
5. The model (v1)
6. The pipeline scripts
7. The Colab notebooks
8. What v1 measures, and where it fails
9. History: what was done, in order
10. The v2 retrain and why v1 stays
11. The microscopic camera work
12. Data requested from the field
13. Known faults and traps
14. What to do next
15. Machine notes
16. Glossary

---

## 1. What the application is

**CWSB** = Coffee White Stem Borer, a beetle whose larva bores inside the stem
of a coffee plant and kills it.

**Goal:** screen coffee plants from photographs, so an infested plant is found
before it is lost. The first target was a farmer's phone app; the team is now
also discussing a handheld device.

**How it works today:**

```
one photo  ->  model  ->  one score between 0 and 1  (probability of infestation)

score <  0.50          ->  no_visible_signs
0.50 <= score < 0.90   ->  inconclusive   (take the photo again)
score >= 0.90          ->  wsb_likely

a plant's score = the HIGHEST score among that plant's photos
```

It is an image **classifier**. It scores the whole photo and does not draw a box
around the damage.

**Two-stage idea now on the table (proposal, nothing built for stage 2):**

1. **Stage 1, normal camera.** Photo of the plant and stem. This is the model
   that exists (v1).
2. **Stage 2, microscopic camera.** Used when stage 1 is inconclusive, to
   confirm from a close view of the bark. This needs its own model. Notebooks
   for it are written (section 11) but there is no usable data yet.

---

## 2. People, ownership and working rules

Two engineers, each with a coding agent: **Leela** and **Charan**.

| File or folder | Owner | Rule |
|---|---|---|
| `PROJECT_STATE.md`, `NEXT_STEPS.md` | both | Facts and roadmap. Last updated 12 Nov 2026, so they do **not** include anything in sections 9 to 12 of this file. |
| `leela_working.md` | Leela's agent | Charan's side reads it, never edits it. |
| `charan_working.md` | Charan's agent | Leela's side reads it, never edits it. |
| `docs/V1_HANDOVER.md` | written 20 Nov 2026 | The detailed v1 reference. Every setting in it was read from the code. |
| `src/` | v1 pipeline | Changed only with agreement; the notebooks embed byte-identical copies. |
| `version_rbn/`, `colab/CWSB_version_rbn_colab.ipynb`, `models/version_rbn/` | Charan, according to `docs/V1_HANDOVER.md` | A separate experiment. **Confirm with the humans before editing it.** It was not touched between 20 and 23 Nov. |

Working rules, from `PROJECT_STATE.md`:

1. Read the four handover files before working.
2. Append a dated entry to your own `*_working.md` after working.
3. Never edit the other person's working file. To flag something, write it in
   your own file under `### For <name>`.
4. Pull before you start, push when you finish.
5. Commit and push only when the human asks.

**Not yet done:** `charan_working.md` has no entry for the work in sections 10
and 11. This file is the record until one is added.

---

## 3. Repository layout, and what is not in Git

```
coffee_Cswb_Model/
├── PROJECT_STATE.md, NEXT_STEPS.md          shared facts and roadmap (12 Nov)
├── leela_working.md, charan_working.md      per-person logs
├── README.md
├── docs/
│   ├── V1_HANDOVER.md                       detailed v1 reference
│   ├── PROJECT_HANDOVER_FULL.md             this file
│   ├── CHARAN_REFERENCE_NOTES.md
│   ├── CWSB_meeting_notes.jpg               one-page notes for the 23 Nov meeting
│   └── synopsis/                            source of the management synopsis PDF
├── src/                                     v1 pipeline scripts (section 6)
├── version_rbn/src/                         separate experiment (section 2)
├── colab/                                   notebooks (section 7)
├── labels/                                  211 YOLO label files (in Git)
├── reports/                                 CSVs in Git; PDFs gitignored
│
│   ---- everything below is gitignored and moves by pendrive ----
├── Data_final/            498 photos, the training source that clean.py reads
├── old_data/              the same 498 photos
├── images_health_16_11_v2/  211 infested stem close-ups (name is misleading)
├── test/, test2/, test_new/ test sets (section 4)
├── Microscopic_level_data/  50 symptom crops (section 11)
├── COLLAB_UPLOAD_FOLDER/    the upload used for the v2 retrain (+ a .zip of it)
├── colab_upload/            three older zips, now redundant, about 3.3 GB
├── models/v1/               the model in use
├── models/v2/               the retrained model, not in use
└── work/                    clean/, splits/, tiles/ (regenerable)
```

`.gitignore` covers every photo folder, `models/`, `work/`, `*.zip`, `*.pt`,
`*.onnx` and `reports/*.pdf`.

---

## 4. The data

### Training data for v1

| Class | Photos | Distinct plants | Split (train / val / test) |
|---|---|---|---|
| Healthy | 28 | 9 | 18 / 2 / 8 |
| Unhealthy (infested) | 470 | 69 | 364 / 53 / 51, plus 2 excluded as blurry |

Facts about it that explain most of the model's weaknesses:

- **17 infested photos for every healthy one**, and only 9 healthy plants.
- **Nearly all are whole-plant shots from 2–3 m.** There are almost no stem
  close-ups, so the model learned "does this bush look unwell".
- **268 of the infested photos come from one camera that never photographed a
  healthy plant.** The camera alone predicts the class.
- **No look-alike negatives**: no drought stress, nutrient deficiency, other
  borers or pruned plants labelled healthy.
- **Almost no early-stage infestation.**
- Labels are per plant. Some photos of an infested plant show no visible sign.

### Other photo sets on Charan's machine

| Folder | Contents | Used for |
|---|---|---|
| `images_health_16_11_v2/` | 211 **infested** stem close-ups, files named `source{N}_…`, from 47 source photos | v1 scores 198 right, 9 retake, 4 wrong. `labels/` holds their boxes. |
| `new_data/infected` inside `COLLAB_UPLOAD_FOLDER/` | 485 Roboflow export files = the same 211 images with duplicates | Added as new infested data for v2 |
| `test/` | 12 blind samples + `answer_key.csv` | Small check |
| `test2/` | 144 images + `answer_key2.csv`: 114 real photos at 768 px and 30 brightness-changed copies | The "all types" check |
| `test_new/` | 30 close-up photos of **healthy** plants | v1 and v2 call all 30 "WSB likely". The clearest demonstration of the bias. |
| `Microscopic_level_data/` | 50 PNG crops of symptoms | Section 11 |

---

## 5. The model (v1)

```
backbone   convnext_tiny.fb_in22k_ft_in1k_384   (timm), about 28 M parameters
input      384 x 384 RGB
classes    2   [Healthy, Unhealthy]   index 1 = Unhealthy
seed       42
```

Transfer learning: pre-trained on ImageNet, then fine-tuned on the coffee photos.

**Training recipe ("contender B", the one that shipped):**

| Setting | Value |
|---|---|
| Epochs | up to 40, early stop after 6 without improvement |
| Epochs 0–1 | head only, lr 1e-3 (warm-up) |
| Epoch 2 on | all layers, lr 1e-4 |
| Optimiser | AdamW, weight decay 0.05, cosine schedule, batch 16 |
| Imbalance handling | weighted sampler, class-weighted loss, 7 extra crops ("tiles") per healthy training photo |
| Augmentation | random resized crop, horizontal flip, rotation 10°, colour jitter, occasional blur |
| Selection | validation PR-AUC, not accuracy |

**Evaluation transform, which must be identical wherever the model is served:**
`Resize(416) -> CenterCrop(384) -> ToTensor -> Normalize(ImageNet)`.

**Thresholds:** `T_low = 0.50`, `T_high = 0.90`. They are set from **healthy**
validation scores. An earlier recipe took both from infested score quantiles;
they collapsed to 0.9071 and 0.9149 and caused 14 of 18 misses. Never do that
again.

**Files in `models/v1/`** (gitignored). SHA-256 prefixes, verified on Charan's
machine on 20 Nov:

| File | Prefix | What |
|---|---|---|
| `contender_B.pt` and `cwsb_v1.pt` | `f63f0c25a012006f` | the model in use; the two files are identical |
| `cwsb_v1.onnx` | `45286379` | the same model as ONNX |
| `config.json` | `badeae29` | thresholds, `remove_bg: false`, test results |
| `contender_A.pt` | `ea3131e3` | the losing first try |
| `ood_reference.npz` | `3df7ccde` | out-of-distribution guard reference |
| `training_result.json` | `be508431` | validation PR-AUC |

`config.json` must always travel with the `.pt`. Serving reads it.

**ONNX contract:** input `rgb_384`, uint8, 384×384×3, already resized and
centre-cropped. Outputs `probabilities` (1×2) and `feature_map` (1×768×12×12).

---

## 6. The pipeline scripts (`src/`)

They run in this order:

```
Data_final/  --clean.py-->  work/clean/ + images.csv + duplicates.csv
             --make_split.py-->  work/splits/split.csv
             --make_tiles.py-->  work/tiles/
             --train.py-->  models/v1/contender_B.pt
             --set_thresholds.py-->  models/v1/config.json
```

| Script | What it does |
|---|---|
| `clean.py` | Reads `Data_final/{Healthy, Unhealthy(Infected)}`. Fixes rotation, strips GPS, saves JPEG q95. Flags blurry (`BLUR_MIN 60`), over-exposed (`CLIPPED_MAX 0.20`), dark (`DARK_MAX 0.40`). Reports duplicates. |
| `make_split.py` | Groups photos into plants by file-number gap (4) or time gap (120 s), then splits **plants** 70 / 15 / 15 per class. Aborts if a plant is in two sets. |
| `make_tiles.py` | 7 crops per healthy **training** photo. |
| `train.py` | `--mode A`, `B` or `both`. |
| `set_thresholds.py` | Writes the thresholds and test counts to `config.json`. |
| `predict_folder.py` | Scores a folder the way serving will. Hard-codes `models/v1/`. |
| `make_report_pdf.py` | Grad-CAM heatmap PDF for `test2/`. Hard-codes `models/v1/`. |
| `make_labels_proof_pdf.py` | Compares heatmaps with the boxes in `labels/`. |
| `analyze_failures.py`, `ood_guard.py`, `crop_experiment.py`, `remove_bg.py` | Experiments; results in `PROJECT_STATE.md` sections 7 and 8. |
| `package_for_colab.py`, `shrink_for_training.py` | Shrink photos to 768 px for upload. |

---

## 7. The Colab notebooks (`colab/`)

All training runs on a Colab T4 GPU. The local machine has no GPU.

| Notebook | Camera | Use | State |
|---|---|---|---|
| `CWSB_train_colab.ipynb` + `CWSB_onnx_export_colab.ipynb` | Normal | The pair that produced v1 | Historical |
| `CWSB_full_pipeline_colab.ipynb` | Normal | Train from a fresh dataset. Zips with `healthy/` and `unhealthy/` folders. | Leela's; not run by Charan |
| `CWSB_retrain_end2end_colab.ipynb` | Normal | Old + new data. **Used for v2.** Upload layout below. | Changed by Charan |
| `cwsb_train_microscopic.ipynb` | Microscopic | First microscopic model | New, tested locally only |
| `cwsb_retrain_microscopic.ipynb` | Microscopic | Later microscopic versions | New, tested locally only |
| `CWSB_v2_convnext_colab.ipynb`, `CWSB_background_filter_colab.ipynb`, `CWSB_detector_colab.ipynb` | Normal | Earlier experiments | Not used in this period |
| `CWSB_version_rbn_colab.ipynb` | Normal | The `version_rbn` experiment | Not touched |

**Every notebook trains from ImageNet weights on all the data together.** An old
`.pt` is only ever used as the yardstick in the comparison table. "Retrain" does
not mean continuing from the old model.

**Upload layout for `CWSB_retrain_end2end_colab.ipynb`** (path set by
`UPLOAD_PATH` in Cell 2, default `MyDrive/cwsb_data/COLLAB_UPLOAD_FOLDER.zip`;
a plain folder also works):

```
COLLAB_UPLOAD_FOLDER/
├── old_data/
│   ├── Healthy/
│   └── Unhealthy(Infected)/
├── new_data/
│   ├── healthy/
│   └── infected/
└── old_models/
    └── cwsb_v1.pt (+ .onnx)
```

**Folder-name rule in the normal-camera notebooks:** a path containing
`unhealthy` or `infect` is Unhealthy; otherwise a path containing `health` is
Healthy; anything else is skipped. A folder called `images_health_16_11_v2`
would therefore be filed as Healthy even though it is infested.

---

## 8. What v1 measures, and where it fails

**Held-out test split, 59 photos** (from `models/v1/config.json`):

| Level | Caught | Missed | False alarms | Cleared | Retake |
|---|---|---|---|---|---|
| Photo | 44 | 3 | 1 | 7 | 4 |
| Plant | 5 | 0 | 1 | 0 | 2 |

These counts reproduce on Charan's machine only on the 768 px copies in `test2/`.
At full resolution the same photos give 47 / 2 / 1 / 7 / 2.

With 59 test photos and 8 healthy ones, the margin of error is about ±7 points.

**Where it fails:**

- **Healthy close-ups.** All 30 photos in `test_new/` are scored "WSB likely".
- **It leans on background and framing.** On 40 infested photos, 78% are still
  called infested with the stem blacked out, and 42% when blurred to 64×64.
  Reproduce with
  `python version_rbn/src/measure.py --ckpt models/v1/contender_B.pt --mode whole --source old_data`.
- **It does not locate damage.** It decides 198 of 211 labelled close-ups
  correctly, but the heatmap peak falls inside a labelled box in only 20 of 207
  (9.7%).
- **The out-of-distribution guard does not help**: it catches 5 of 19.

**Root cause:** the data, not the code. The fix is more healthy plants,
photographed the same way as the infested ones.

---

## 9. History: what was done, in order

| When | Who | What |
|---|---|---|
| Up to 12 Nov | Leela | Built the pipeline, trained contenders A and B, found and fixed the threshold bug, wrote the evidence PDFs. See `leela_working.md`. |
| 12 Nov | Charan | Joined, read the repo, reproduced the numbers. |
| 18 Nov | Charan | Management synopsis PDF. Scored three new photo batches. Found that `images_health_16_11_v2` is infested data, not healthy. Wrote `src/make_labels_proof_pdf.py`. See `charan_working.md`. |
| 20 Nov | Leela | `docs/V1_HANDOVER.md`; commit `db2e0ab` fixed `version_rbn/src/measure.py` to score whole frames the way v1 is served. |
| 20 Nov | Charan | Replaced `models/v1/` from the pendrive and verified all seven fingerprints; reproduced every committed number. |
| 20–23 Nov | Charan | Changed the retrain notebook to one upload; ran the v2 retrain on Colab; compared v1 and v2 (section 10). |
| 23 Nov | Charan | Team meeting on the model, the imbalance and the data needed. Notes in `docs/CWSB_meeting_notes.jpg`. |
| 23 Nov | Charan | Received 50 "microscopic" images, reviewed them, wrote the two microscopic notebooks (section 11). Commit `b7a4ece`. |

**One correction still open.** `charan_working.md` on branch
`origin/charan_20/11`, lines 326 and 359, says the `contender_B.pt` fingerprint
differs between the two machines. That is wrong: an MD5 was compared with a
SHA-256 prefix. The files are identical.

---

## 10. The v2 retrain and why v1 stays

### What was run

- **Notebook:** `colab/CWSB_retrain_end2end_colab.ipynb`, Tesla T4.
- **Data:** the 498 old photos, plus 485 new infested files (211 unique images,
  augmentations of 47 source photos). **No new healthy photos.**
- **After merging:** Healthy 28, Unhealthy 681.
- **Split:** train 450, val 94, test 100, excluded 65.
- **Result:** best epoch 7, validation PR-AUC 0.99977. Thresholds stayed at the
  default 0.50 / 0.90 because validation had 2 healthy photos. ONNX parity
  2.24e-07.

### Notebook changes made for this run

- Cells 0 and 2 rewritten for one upload from Drive, as a folder or a zip.
- Roboflow names (`vf_10_jpg.rf.<hash>.jpg`) are renamed to `rfsrc{N}_{kkk}`, so
  every version of one source photo lands in the same set.
- Byte-identical duplicates are dropped.
- The embedded scripts are still byte-identical to `src/`.

### Outputs

`models/v2/` (gitignored): `cwsb_v2.pt` (SHA-256 `8db2ae59f95f58cd`),
`cwsb_v2.onnx`, `config_v2.json`, `split_v2.csv`, `training_result_v2.json`,
`cwsb_v2_test_report.pdf` (101 pages of heatmaps).

### v1 against v2

Counts are caught / missed / false alarm / cleared / retake.

| Test set | v1 | v2 | How to read it |
|---|---|---|---|
| v2's test split, 100 photos | 88 / 2 / 1 / 7 / 2 | 77 / 7 / 0 / 7 / 9 | Unfair to v2: all 75 old infested photos in it were v1 training or validation photos. |
| `test_new/`, 30 healthy close-ups | 30 false alarms | 30 false alarms | No change. |
| 17 new infested close-ups in the test split | 17 caught | 17 caught | No change. |
| `test/`, 12 blind samples | 7 caught, 4 cleared, 1 false alarm | 5 caught, 4 cleared, 3 retake | |
| `test2/`, 114 real photos | 90 / 3 / 1 / 9 / 11 | 97 / 3 / 0 / 6 / 8 | Favours v2: it trained on 58 of them and validated on 46. |

**Verdict: keep v1 in use.** v2 is not shown to be better, and the main fault
is unchanged, because v2 got more infested photos and no healthy ones. All seven
of v2's misses on its test split come from the `IMG_20170725` photo series.

To serve v2 anyway, `src/predict_folder.py` and `src/make_report_pdf.py` would
need their hard-coded `models/v1/` paths changed.

---

## 11. The microscopic camera work

### The data that arrived

`Microscopic_level_data/`: 50 PNG files named by symptom.

| Symptom | Images |
|---|---|
| Holes | 39 |
| Ridges | 9 |
| Bulging | 1 |
| Crack | 1 |

**They cannot be trained on:**

- They are crops of normal photos, each a different size (67×56 up to
  1222×632 px) and mostly blurry from enlargement. They are not microscope
  captures.
- All 50 show damage. There is no healthy or look-alike example.
- There are no plant IDs.

They are useful as a reference of which symptoms to photograph.

### What the field team was asked for

Original camera files with no cropping; healthy images taken the same way; a
number for every coffee plant (`P001`, `P002` …) in the file name; the device
and magnification; a trial of 10 plants first.

**Still unanswered:**

1. Were the 50 images taken with a microscopic camera or cropped from phone
   photos?
2. Which device and magnification will be used?
3. Should the model give one score per image (classification), or find and
   count holes and ridges (detection)? Detection needs box labels and a
   different pipeline from the one below.

### The two notebooks

- `colab/cwsb_train_microscopic.ipynb`: first model, `cwsb_micro_v1`, from
  `new_data/` only.
- `colab/cwsb_retrain_microscopic.ipynb`: `cwsb_micro_v2` onwards, from
  `old_data/` + `new_data/`, compared with the model in `old_models/`.

Both are self-contained. Their scripts (`clean_micro.py`, `split_micro.py`,
`train_micro.py`) are embedded in the notebook and are not in `src/`.

**Upload** (default `MyDrive/cwsb_data/MICRO_UPLOAD_FOLDER.zip`, set in Cell 1):

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

Every image needs a plant ID, as a folder (`healthy/P001/…`) or at the start of
the file name (`P001_hole_1.jpg`).

**Same as v1:** backbone, 384 px, recipe, seed, selection metric, threshold
rule, heatmap PDF, ONNX contract.

**Different from the normal-camera retrain notebook:**

| Step | Normal-camera notebook | Microscopic notebooks |
|---|---|---|
| Plant grouping | Guessed from file number and time | Plant ID from folder or file name |
| Quality check | Excludes flagged photos; enlarges small photos first | Reports only by default (`QUALITY_GATE = 'report'`); never enlarges; images under 224 px always left out |
| Healthy tiles | Yes | No |
| Flips | Horizontal | Horizontal and vertical |
| Split on retrain | Everything re-split | Old images keep their set, matched by content against `split_micro_vN.csv` |
| Test report | Per image | Per image and per plant |
| Same file in both class folders | Not checked | Left out and listed |
| Epoch log | Not shown in Colab | One line per epoch, saved in the training result |

**Release zip:** `cwsb_micro_v{N}.pt`, `.onnx`, `config_micro_v{N}.json`,
`training_result_micro_v{N}.json`, `split_micro_v{N}.csv`,
`quality_micro_v{N}.csv`, `cwsb_micro_v{N}_test_report.pdf`.

**Testing done:** locally on CPU with fake data (512 px crops of normal photos,
16 plants then 8 more, 3 epochs and 1 epoch). Every cell ran in both notebooks.
ONNX parity 4.5e-07. In the retrain run all 49 old images kept their previous
set.

**Not tested:** a Colab GPU run, and real microscope images. The fake-data
scores say nothing about accuracy. The quality limits are the phone-photo values
and must be re-set from the first real batch.

---

## 12. Data requested from the field

These are Charan's recommendations from the 23 Nov meeting. **The counts are not
yet agreed with the manager.**

**Normal camera**

- 300 healthy and 300 infested plants, 3 photos each: whole plant from 2–3 m,
  full stem from about 1 m, stem close-up from about 30 cm. 1,800 photos.
- About 75 of the healthy plants should be look-alikes, and about 100 of the
  infested plants early-stage.
- Both classes photographed the same way: same device, same three shots, same
  estates and days.
- Every label confirmed by inspecting the stem.
- First milestone 150 + 150 plants, then train and measure.

**Microscopic camera**

- Pilot of 150 infested and 150 non-infested plants, 3 to 5 images each, one
  device and magnification, with look-alike negatives.
- Collected on the same field visits.

**What this would and would not give.** It fixes the imbalance, the healthy
close-up false alarms and the camera shortcut, and it makes the accuracy
measurable to about ±8 points. It does not guarantee any particular accuracy.

---

## 13. Known faults and traps

**Faults found, not fixed**

1. **`src/clean.py` enlarges small photos before the blur check.** It always
   resizes to 1024 px wide. In the v2 run, 63 of the 211 new 640 px photos were
   excluded as "blurry" for this reason. Fix: shrink only. It needs agreement
   because it changes `src/` and the notebook copies.
2. **The normal-camera retrain notebook re-splits everything.** Old test photos
   can become training photos, so its old-vs-new table is not a fair
   comparison. Fix: keep the previous split for old photos, as the microscopic
   retrain notebook does.
3. **`docs/V1_HANDOVER.md` section 8 and `docs/CHARAN_REFERENCE_NOTES.md`
   describe the old three-zip upload** for the retrain notebook. The layout in
   section 7 of this file is the current one.
4. **`PROJECT_STATE.md` and `NEXT_STEPS.md` stop at 12 Nov.**

**Traps**

- `clean.py` reads a folder named `Data_final`, not `old_data`. With it missing,
  the script fails with `IndexError` at line 84.
- Never set `remove_bg` by hand on a trained model. Turning it on for v1 dropped
  infested stem scores from 0.97 to 0.20.
- Never split by photo, only by plant.
- Use the test split once, at the end. Thresholds come from validation.
- Set the Colab runtime to T4 GPU **before** running anything; changing it later
  wipes the session.
- Unzip a Colab release flat into `models/v{N}/`. Never overwrite an earlier
  version's folder.
- A validation PR-AUC of 1.0 here means "two healthy validation photos", not a
  good model.

---

## 14. What to do next

In priority order.

1. **Wait for field data.** No model change will fix the bias without healthy
   photos taken the same way as infested ones.
2. **Get the three microscopic questions answered** (section 11).
3. **When the 10-plant microscopic trial arrives:** run
   `cwsb_train_microscopic.ipynb`, read the quality table from Cell 3, set the
   limits in Cell 1, and check the plant IDs are picked up.
4. **If normal-camera photos arrive with plant IDs:** build a normal-camera
   train and retrain pair in the microscopic style. Charan offered this; it is
   not started.
5. **Fix faults 1 and 2 in section 13** before the next normal-camera retrain.
6. **Housekeeping:** add a `charan_working.md` entry for sections 10 and 11;
   correct the fingerprint lines (section 9); update `PROJECT_STATE.md`,
   `NEXT_STEPS.md` and the two out-of-date docs; delete `colab_upload/*.zip`.
7. **Longer term, from `NEXT_STEPS.md`:** a supervised framing gate that rejects
   photos of the wrong kind, then the API and the app.

---

## 15. Machine notes (Charan's Windows machine)

- Windows 11, Python 3.14, PyTorch 2.12 CPU build. No GPU: scoring and reports
  run locally, training runs on Colab.
- In Git Bash, `python - <<EOF … EOF` drops into an interactive prompt and
  hangs. Write the code to a file and run `python file.py < /dev/null`.
- Set `PYTHONIOENCODING=utf-8` before printing notebook text; the default
  console encoding fails on arrows and box characters.
- No `pdftoppm`; read PDFs with PyMuPDF (`fitz`).
- `COLLAB_UPLOAD_FOLDER/` and its zip are the exact v2 training upload. Keep
  them until v2 is either adopted or dropped.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| CWSB / WSB | Coffee White Stem Borer |
| v1 | The model in use, `models/v1/contender_B.pt` = `cwsb_v1.pt` |
| v2 | The 20–23 Nov retrain, `models/v2/cwsb_v2.pt`, not in use |
| Contender A / B | Two training approaches; B (full fine-tune) won |
| Caught / missed | Infested photo scored ≥ 0.90 / < 0.50 |
| False alarm / cleared | Healthy photo scored ≥ 0.90 / < 0.50 |
| Retake / inconclusive | Score from 0.50 up to 0.90 |
| Plant, group | All photos of one coffee plant; the unit of the split |
| Tiles | Extra crops cut from healthy training photos |
| PR-AUC | Area under the precision-recall curve, the model-selection metric |
| Grad-CAM | Heatmap of which image area drove the score |
| Parity | Largest difference between `.pt` and `.onnx` scores; must be under 1e-4 |
| Roboflow export | Augmented copies of labelled photos, named `<name>.rf.<hash>.jpg` |
| `version_rbn` | A separate experiment that scores pieces of the photo instead of the whole frame |
