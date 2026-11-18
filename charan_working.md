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

## 2026-11-18 — Management synopsis PDF

### What was done

`git pull` (already up to date at `451b33f`), read the four handover files, then built a
10-page synopsis for management: **`Varsapradaya_CWSB_Synopsis.pdf`** (repo root, 3.9 MB).
Source is in `docs/synopsis/` (HTML + bundled fonts + images; regenerate with headless Chrome,
command in `docs/synopsis/README.md`).

Contents: cover, one-page executive summary, the problem, how it works (photo → quality gate →
model → 3-way verdict), the 8 components built, test results, Grad-CAM examples (1 caught,
1 cleared, 1 close-up false alarm), benefits, limitations with planned fixes, locked design
decisions, roadmap, asks of management, technical appendix and glossary.
Branding (colours `#004225` / `#003153`, Lexend Deca + Lora, logo) taken from the
`varsapradaya-web` project.

### Numbers used (all from existing files, nothing re-measured)

- Test split, from `reports/cwsb_test_report_test_real.pdf` page 1 and `models/v1/config.json`:
  59 photos / 8 plants; recall 93.6% (44 of 47 decided), correct clears 87.5% (7 of 8),
  accuracy where decided 92.7% (51 of 55), 4 sent to retake. Plant level 5 / 0 / 1 / 0 / 2.
- **Worth knowing:** that test split contains only **one healthy plant** (`plant_003`), so
  the 87.5% clear rate is 8 photos of a single plant. The PDF says so explicitly.
- 95.9% full-resolution recall, 1 of 30 lighting flips, 14 of 18 misses from the old threshold
  bug, 19 of 19 close-ups flagged, OOD guard 5 of 19: all quoted from `PROJECT_STATE.md`.

### What was not included, and why

Leela's 16 Nov CSVs (`predictions_images_16_11.csv`, `predictions_images_health_16_11.csv`,
`results_data_test.csv`, `mask_experiment_images_16_11.csv`, `predictions_data.csv`) have no
write-up or ground truth in `leela_working.md`, so they are left out of the synopsis rather than
interpreted. No YOLO / detector work is mentioned (project is classification only).

### For Leela

1. Could you add a short entry for the 16 Nov runs: what `images_health_16_11/` contains
   (healthy plants?) and what the results mean? If those 16 photos are healthy, then 0 of 16 were
   cleared (9 WSB likely, 6 retake, 1 blurry), which would be worth stating in `PROJECT_STATE.md`.
   **Answered in the entry below** — those 16 are indoor cut-stem close-ups believed healthy, and
   0 of 16 were cleared. `images_health_16_11_v2/` despite its name is **not** healthy data: it
   arrived with confirmed damage boxes on every image.
2. `models/v1/cwsb_v1.onnx` (13 Nov) exists on this machine but `PROJECT_STATE.md` does not list
   it or a parity result. If the 4-decimal parity check passed, please record the number.

### 2026-11-18 (later) — Synopsis reworked for officials

The synopsis is going to officials, not only the manager. At the user's request the PDF is now a
**7-page solution overview** with no dataset information and no test details: removed the dataset
table and all counts, the results page, the close-up failure example, the limitations/design-decisions
page, the technical appendix and glossary, the cover meta block (prepared for/by, date, status),
the "Project synopsis" labels, and the "Kodagu & Chikkamagaluru" and "Working prototype (V1)" pills.
Footer changed from "Internal" to "varsapradaya.com". Roadmap and asks kept, with numbers removed.
The internal 10-page version with results is still in git history (`feee910`).

---

## 2026-11-18 — Scored three new photo batches; built the labels-vs-model proof report

### What I ran

- `images_16_11/` (30 WhatsApp photos, infested stem close-ups) → 26 flagged
  at 0.91–1.00, 2 blurry-gated, 2 inconclusive.
- `images_health_16_11/` (16 indoor cut-stem photos, believed healthy) → **0 cleared**:
  8 flagged at 0.95–0.99, 7 in the 0.80–0.90 hedge band, 1 blurry-gated.
- `images_health_16_11_v2/` (211 images) → 198 flagged, 9 inconclusive, 4 cleared.

Reports in `reports/predictions_<folder>.pdf` + `.csv`. The three new image
folders are gitignored; transfer by pendrive as usual.

### What the first two batches showed

v1 gives nearly the same answer for healthy and infested **stem close-ups**
(0.80–0.99 vs 0.91–1.00). Both batches are out of distribution — indoor cut
stems on a white table — and the model fires on "rough bark close-up". This is
the same close-up failure already documented, now with a healthy counter-example
to go with it. Confirms Priority 2 (framing gate) is the right next step.

### images_health_16_11_v2 is NOT healthy data

The folder name is misleading. A `labels/` folder arrived with it: 211 YOLO
`.txt` files, one per image, 645 boxes (class 0: 337, class 1: 300, class 2: 8),
**no empty files**. Every image is confirmed infested with the damage boxed by
the annotation team. The images are 640×640 dataset exports with `sourceN_<hash>`
names, and scores cluster tightly by `sourceN` — each group is several crops of
the same original photo, so this is ~47 distinct photos, not 211 samples.

### New script: `src/make_labels_proof_pdf.py`

Built to answer the manager's question ("to what extent it reaches") by checking
predictions against the confirmed boxes. Maps YOLO boxes through the same
Resize(416)/CenterCrop(384) geometry as `eval_tf`, so boxes land exactly where
the model sees them (verified visually before finalising). Outputs
`reports/proof_labels_vs_model.pdf` (212 pages, summary first, misses first) and
`.csv`. Rerunnable on any future labelled batch.

Results:

- **Image level: 198/211 flagged (93.8%)**, 9 inconclusive, 4 missed. All 4
  misses are `source10` (moss-covered stem, scores 0.11–0.35). By source photo:
  44 of 47 fully flagged.
- **Location level: heatmap peak inside a confirmed box in 20/207 (9.7%)**, mean
  7% of attention inside boxes. Boxes cover 5.1% of the image on average, so
  this is only ~2× chance.

### What this means

v1 decides correctly at image level but does not localise. That is expected —
it is a classifier trained on whole-image labels with a coarse 12×12 Grad-CAM,
so it judges the whole stem. Pointing at the damage needs a detector trained on
these boxes. Note this is also why the 4 `source10` misses matter: moss changes
the whole-stem texture, and there is nothing forcing the model to look at the
boxed damage.

### For Leela

1. **`labels/` is now in Git** (28 KB, 211 files). The matching images are not —
   they are gitignored like every other photo folder. The labels are confirmed
   ground truth and cost nothing to track; the images come by pendrive.
2. **We now have real detector-grade data**: `data/` (485 labelled, infested
   only) plus these 211. Still zero labelled healthy stems, which is the gap
   that blocks both the detector and the framing gate.
3. If you want the localisation number to improve, it is a new model, not a
   tuning job — `make_labels_proof_pdf.py` is ready to measure any candidate the
   same way.
