# Charan — Working Log

**Owner: Charan.** Leela's agent reads this file but must never edit it.
Charan's agent appends a dated entry here at the end of every working session.
Leela's equivalent is `leela_working.md`.

For facts about the project, see `PROJECT_STATE.md`. For the roadmap, see `NEXT_STEPS.md`.
This file is the *narrative*: what was tried, what was learned, what was abandoned and why.

---

## 2026-11-12 — Joined the project

### What I read

`git pull` first (already up to date at `5e9b99c`), then all of `PROJECT_STATE.md`,
`NEXT_STEPS.md` and `leela_working.md`, including the "For Charan" section. Created this file.

### State I found the repo in

- Branch `main`, clean working tree, level with `origin/main`.
- **The large files are already on this machine** — no pendrive transfer is needed for
  this side. Checked against the transfer list in `NEXT_STEPS.md`:

  | Folder | Found | Matches the docs? |
  |---|---|---|
  | `models/v1/` | 5 files, 108 MB, flat (not nested) | yes — `contender_A.pt`, `contender_B.pt`, `config.json`, `training_result.json`, `ood_reference.npz` |
  | `Data_final/` | 498 photos (28 Healthy / 470 Unhealthy), 2.9 GB | yes |
  | `work/` | `clean/`, `splits/split.csv` (498 rows), `tiles/`, 2.6 GB | yes |
  | `test/` | 12 samples + `answer_key.csv` | yes |
  | `test2/` | 144 images + `answer_key2.csv` | yes |
  | `test_new/` | 30 photos | yes |
  | `reports/` | 3 PDFs + 5 CSVs, 95 MB | yes |

- `models/v1/config.json` reads `T_low = 0.5`, `T_high = 0.9`, model `contender_B.pt`,
  as documented. It also holds a plant-level test result that `PROJECT_STATE.md` did not
  quote: 5 caught, 0 missed, 1 false alarm, 0 cleared, 2 inconclusive. Added to section 6.
- Python environment on this machine: torch 2.12.0 (CPU only, no CUDA), timm 1.0.31,
  torchvision and scikit-learn all import. Enough for scoring, reports and experiments;
  training stays on Colab as per the locked decision.

### What I measured

Nothing about the model this session. I did not load or run it — this entry is an
inventory only. The numbers above are file counts and config contents, not re-scored results.

### What did not work / was not done

Nothing was attempted and abandoned. Not yet done: I have not re-run
`src/predict_folder.py` or `src/analyze_failures.py` to confirm that this machine
reproduces the documented numbers. That is the first thing to do next session, before
building anything on top of them.

### What I plan to work on

Taking up the three items in Leela's "For Charan" section, in this order:

1. **Reproduce the baseline here** — score `test2/` and `test_new/` with
   `src/predict_folder.py` and confirm the documented figures (93.6% recall at 768 px;
   all 19 unique close-ups in `test_new/` scoring 0.98–1.00).
2. **Framing gate (Priority 2).** Start collecting and labelling wrong-framing photos
   (close-ups, landscapes, ground, sky, people, indoor). The 498 training photos are the
   "right framing" class; `test_new/` and `CB_DSC_0285` are the first wrong-framing examples.
   Then a small binary classifier ahead of the WSB model.
3. **Look-alike negatives (Priority 1)** — stressed-but-uninfested plants. This is field
   collection, not something that can be done at the desk.

Both warnings are noted and will be followed: thresholds are not to be re-derived from
infected-score quantiles (`T_low` comes from the healthy distribution), and the close-up
failure is a data/framing problem, not something to tune away.

### For Leela

1. **The report PDFs are tracked in Git even though `.gitignore` lists `reports/*.pdf`.**
   Three PDFs (`cwsb_test_report.pdf`, `cwsb_test_report_test_real.pdf`,
   `predictions_test_new.pdf`, about 94 MB together) were committed before the ignore rule
   was added, so the rule does not apply to them and every clone downloads them. I have
   **not** removed them — your call. If you want them out: `git rm --cached reports/*.pdf`
   and commit. That stops future tracking; they stay in history unless it is rewritten.
2. **`PROJECT_STATE.md` section 10 said "src/ 14 scripts"; there are 12** `.py` files in
   `src/`, in Git and on disk. I corrected the count. If two scripts exist on your machine
   and were never committed, please push them.
3. The pendrive is not needed for my side — everything on the transfer list is already here.

---

## 2026-11-12 (second session) — Reviewed the external YOLOv8 repo

Charan's team lead pointed to an existing YOLO approach. It was downloaded (GitHub zip,
commit `0b27cd4`) into `external_repo/Detection-of-white-stem-borer-using-YoloV8/`.
The folder is in `.gitignore` (118 MB), so it does **not** travel by Git.

### What it is

- YOLOv8s **object detector** (11.1M parameters), trained in Colab on 2023-09-12,
  100 epochs, image size 416, starting from `yolov8s.pt`. Weights are included:
  `runs/detect/train/weights/best.pt`.
- Dataset is a Roboflow export (`capstone2-kvkgn/wsb_2`, version 2, CC BY 4.0), 640×640.
- Three box classes: `hole` (exit holes), `ring` (stem ridges), `worm` (larva in split stem).
- The photos are **stem close-ups and trunk shots** — the exact view our classifier never saw.

### What I counted

| Split | Images | Boxes | hole | ring | worm |
|---|---|---|---|---|---|
| train | 422 | 1306 | 706 | 583 | 17 |
| valid | 41 | 134 | 64 | 70 | 0 |
| test | 22 | 66 | 33 | 31 | 2 |
| **total** | **485** | **1506** | 803 | 684 | 19 |

- **Healthy: 0. Infected: 485.** Every image has at least one box; there is no healthy
  class and not a single empty-label (background) image.
- The 485 images come from only **212 unique source photos**; Roboflow augmentation made
  up to 3 copies of each in train. No source photo appears in more than one split.
- Number of distinct *plants* is not recorded anywhere in the repo.
- Boxes are small: median box covers 0.3–0.5% of the image.

### Its reported results (validation, 41 images, from the notebook output)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.80 | 0.64 | 0.70 | 0.30 |
| hole | 0.83 | 0.59 | 0.65 | 0.24 |
| ring | 0.78 | 0.69 | 0.75 | 0.35 |

`worm` has zero validation examples, so it was never evaluated. The confusion matrix shows
33% of holes and 26% of rings missed. These are the repo's own numbers; I did not re-run
the model.

### What this means for us

- It is complementary, not a replacement: it finds the diagnostic marks on a stem; ours
  judges whole-plant condition. It directly addresses "stem close-ups" in Priority 1.
- It has the same blind spot in a different form: **with no healthy stems in training or
  validation, its false-alarm rate on clean stems is unmeasured.** Mossy bark, pruning
  scars and natural bark cracks are untested.
- A val set of 41 images is too small to trust the 0.70 mAP to better than roughly ±0.1.

### What did not work / was not done

- Did not run `best.pt` on anything. `ultralytics` is not installed here. The obvious next
  measurement is to run it on `test_new/` (19 uninfested close-ups) to see how many false
  boxes it draws on healthy stems — that is the number the repo cannot give.
- Small data hygiene issues found in the download: a stray `abcdef.txt` in
  `data/train/images/`, and two label files with no `.txt` extension
  (`train/labels/1_JPG.rf.f60c…`, `test/labels/15_jpg.rf.4f7f…`) which YOLO will ignore.

### For Leela

1. The YOLO repo is public: `Detection-of-white-stem-borer-using-YoloV8` on GitHub
   (commit `0b27cd4`); the dataset is on Roboflow Universe at `capstone2-kvkgn/wsb_2`.
   Download it to `external_repo/` if you want to look. 212 real stem close-up photos with
   hole/ring boxes, CC BY 4.0 — usable as the "stem close-up" class for the framing gate
   and as a start on close-up evidence, but it adds **zero** healthy examples.

---

## 2026-11-12 (third session) — Retraining notebook for future data

Charan asked: when new healthy / unhealthy photos arrive, how do we train them into
`contender_B`? The existing pipeline could do it but had four problems for a second run:
it overwrote `models/v1/`, re-splitting would reshuffle the test set, thresholds were
hard-coded, and it had no way to compare the new model with the old one.

### What I built

**`colab/CWSB_retrain_colab.ipynb`** — one self-contained notebook (no `%%writefile`, no
dependency on `src/`). Edit the CONFIG cell (`VERSION`, `DATA_SOURCES`), Run all.

- Takes a list of zips/folders (`Data_final.zip` + each new delivery). Finds
  `Healthy*/` and `Unhealthy*/` / `Infected*/` folders at any depth; accepts
  jpg/jpeg/png/webp/heic. A sub-folder per plant under the class folder is used as the
  plant identity; loose files fall back to the v1 frame-number grouping.
- Cleaning = `src/clean.py` rules (same blur/clipped/dark constants). Exact duplicates
  (by source-file SHA-256) are skipped; a photo filed under both classes is dropped and
  reported.
- **Frozen split.** Reads the previous split file and keeps every old photo's plant and
  set. Only new plants are dealt out, to reach 70/15/15 overall. New photos that are
  near-duplicates of an old photo are put in that old photo's set. Aborts if any old
  photo is missing from the data, if a plant is in two sets, or if an old photo moved.
- Healthy tiling is "auto": on while there are fewer than 150 healthy training photos.
- Training recipe is contender B unchanged (same transforms, optimiser, warm-up, sampler,
  loss). Optional `START_FROM = "previous"` warm-starts from the old checkpoint; it is
  refused without the previous split, because old training photos could otherwise land
  in the new test set.
- **One deliberate change to model selection:** best epoch = highest val PR-AUC, ties
  broken by lower balanced validation loss. In v1 PR-AUC hit 1.0 and the strict `>`
  comparison kept the first epoch that reached it.
- **Thresholds from healthy validation scores**: `T_low` = 95th percentile of healthy val
  scores (clipped to 0.20–0.80), `T_high` = max(0.90, 99th percentile + 0.05), band forced
  ≥ 0.10 wide. With fewer than 20 healthy val photos it keeps the hand-set 0.50 / 0.90.
  Nothing is derived from infected scores.
- Reports val/test per photo and per plant with 95% ranges, test ROC-AUC, and a
  side-by-side with the previous model on the same test photos (all, and old-only).
- Writes to `Drive/cwsb/<VERSION>/`: `contender_B.pt`, `config.json` (same keys as v1 plus
  extras), `split_<VERSION>.csv`, `images.csv`, `duplicates.csv`, `scores_val_test.csv`,
  `training_log.csv`, `score_histograms.png`. Refuses to overwrite an existing version.

**`splits/split_v1.csv`** — new, in Git. It is `work/splits/split.csv` plus a
`src_sha256` column (hash of each original file in `Data_final/`). This is the file the
notebook needs to freeze the v1 split, and `work/` is not in Git.

### What I measured (local CPU, mechanics only — not model quality)

- Data steps on the real `Data_final/` with `split_v1.csv`: all 498 photos matched, the
  output split is **identical** to v1 (18/2/8 healthy, 364/53/51 unhealthy, 2 excluded as
  blurry), 126 tiles — the same numbers as PROJECT_STATE section 2.
- Full run, fresh start, on a 26-photo sample (3 epochs): completed, all outputs written.
- Full run, 26 old + 9 new photos in a zip, previous split + warm start from the real
  `models/v1/contender_B.pt`: old photos kept their sets, the planted duplicate was
  skipped, the comparison table and histogram were produced.

### What was NOT done / limits

- **Not run on Colab or on a GPU.** The Drive mount and the GPU path are untested; the
  code is the same but the first real run is the real test.
- No real retrain was done — there is no new data yet, so there is no v2 and no new numbers.
- The notebook scores on the 768 px copies, so its figures compare with the v1 "768 px"
  numbers (93.6% recall), not the full-resolution ones (95.9%).
- `src/predict_folder.py` and `src/analyze_failures.py` still read `models/v1/` only. To
  use a v2 locally they need a version option. Not changed.
- Installed `opencv-python-headless` and `matplotlib` on this machine for the test.

### Facts noticed on the way

- Healthy plants per set in v1: 6 train, 2 val, **1 test**. All 8 healthy test photos are
  one plant, so "healthy cleared" on the test set is a statement about a single plant.

### For Leela

1. New notebook `colab/CWSB_retrain_colab.ipynb` and `splits/split_v1.csv` — please check
   that `split_v1.csv` matches your `work/splits/split.csv` (it should be the same rows
   plus the hash column). If your `Data_final/` files differ byte-for-byte from mine, the
   notebook falls back to matching by file name, so it still works.
2. After any retrain, commit the new `split_<VERSION>.csv` into `splits/` — the next
   retrain needs it.
3. The model-selection tie-break and the threshold rule are my choices; say if you
   disagree with either before the first real v2 run.

---

## 2026-11-12 (fourth session) — v1 data laid out by class and set

Charan asked for the v1 data as `healthy/{train,test,valid}` and
`unhealthy/{train,test,valid}`.

- Created **`Data_v1_split/`** at the repo root (2.5 GB, in `.gitignore`). Photos are
  **copies** of `work/clean/` (rotation fixed, no GPS, full resolution — what v1 trained
  on), placed according to `splits/split_v1.csv`. `Data_final/` and `work/` are untouched,
  so the existing pipeline and the retraining notebook still work.
- Counts: healthy 18 train / 2 valid / 8 test; unhealthy 364 train / 53 valid / 51 test
  = 496. The 2 photos v1 excluded as blurry are left out. The 126 healthy crops (tiles)
  are not included — they are derived, not source photos.
- `Data_v1_split/manifest.csv` lists file, class, set and plant. Checked: 78 plants, none
  in more than one set.
- Nothing measured on the model; nothing tried and abandoned.

### For Leela

`Data_v1_split/` is a convenience copy only and is not in Git. To make the same folder on
your machine, copy each file in `splits/split_v1.csv` from `work/clean/` into
`<class>/<set>/` (val → `valid`).

---

## 2026-11-12 (fifth session) — Should the YOLO close-ups go into the Unhealthy class? (No)

Charan asked whether the external repo's train/valid/test images can be added to
`Data_v1_split/unhealthy/` and the model retrained. I did **not** merge them and did
**not** retrain. Reason, with a measurement:

### What I measured

Scored the external stem close-ups with the unchanged v1 model (`contender_B.pt`,
thresholds 0.50 / 0.90), one copy per source photo — 211 photos. Scores saved to
`reports/v1_on_external_yolo_closeups.csv`.

| Verdict | Photos |
|---|---|
| WSB likely (≥ 0.90) | 202 (96%) |
| Inconclusive | 5 |
| No visible signs | 4 |

Median score 0.999.

Put next to the earlier result (19 of 19 **uninfested** close-ups in `test_new/` also
scored 0.98–1.00), this says v1 calls a close-up "infected" whether it is or not. It is
answering "is this a close-up", not "is there borer damage".

### Why adding them would hurt

- Every close-up we have with a label is infected; there is not one healthy close-up.
  Training on them teaches "close-up = infected" even harder — the exact failure in
  PROJECT_STATE section 7 item 3.
- The model already scores them 0.999, so there is nothing for it to learn from them
  about infection.
- The test numbers would go *up* (more easy infected photos caught) while the model got
  no better — a misleading improvement.
- 485 images are only ~212 photos; the rest are augmented copies, which would be
  over-weighted, and copies of one photo must not be spread carelessly across sets.

### What the close-ups are good for instead

- Wrong-framing examples for the framing gate (Priority 2): "this is a close-up, not a
  whole-plant photo".
- The stem-evidence detector (YOLO route), once healthy stem close-ups exist as negatives.

### Not done

No retrain. It also cannot be run here: training is Colab-GPU only and this machine is
CPU-only.

### For Leela

Please do not add the external close-ups to the Unhealthy class either, for the reasons
above. The measurement is reproducible from `reports/v1_on_external_yolo_closeups.csv`.
