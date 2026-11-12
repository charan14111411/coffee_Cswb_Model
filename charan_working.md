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
