# CWSB Model — What Still Has To Be Done

Read `PROJECT_STATE.md` first. This file is the roadmap, in priority order.
Both agents may update it. Tick items off with `[x]` and add the date.

Last updated: **2026-11-12**

---

## Priority 1 — Data collection (blocks everything else)

Nothing in the modelling can be fixed without this. The project has **9 healthy plants**
against 69 infected. In rough order of value per photo taken:

- [ ] **Healthy plants, from many distinct plants.** Target **150+ distinct healthy plants**,
      2–3 photos each, across different estates, shade levels, times of day and seasons.
      Photos of a plant already in the set add little; new plants are the real currency.
- [ ] **Look-alike negatives.** Plants that look unwell for reasons other than WSB:
      drought stress, nutrient deficiency, shot-hole borer, die-back from other causes,
      recently pruned plants, natural leaf senescence. Without these the model cannot learn
      the difference, and in the field they are everywhere. This is why 19 of Charan-side
      test photos of healthy branches scored 0.99.
- [ ] **Early-stage infestation.** Fresh ridges or a bore hole with the canopy still green.
      We have almost none. A plant that is already wilting is already lost, so late-stage
      detection has little commercial value. **This is the biggest capability gap.**
- [ ] **Stem close-ups paired with the wide shot** — both frames of the same plant, so the
      model can learn the actual diagnostic evidence instead of inferring from foliage.
- [ ] **Verified labels with severity.** Labels must come from physical stem inspection
      recorded against the photo, graded early / mid / late — not a guess from the image.
- [ ] **Expert review of `plant_012`** (`CB_2_DSC01024`, `01027`, `01030`): is it mislabelled,
      or is the evidence simply not in frame?

**Target for V2:** ~400 healthy photos from 150+ plants, plus a comparable spread of
infected photos stratified by severity. At that size the accuracy margin of error narrows
from ±7 points to about ±2.5.

---

## Priority 2 — A supervised framing gate

The single highest-value engineering fix. It would have caught **all 19** of the false
alarms on out-of-framing photos.

- [ ] Collect/label wrong-framing examples (close-ups, landscape shots, ground, sky, people,
      indoor). These are cheap to gather — no agronomy needed.
- [ ] Train a small binary classifier: "is this the whole-plant view the model expects?"
- [ ] Wire it ahead of the WSB model, returning *"photo not suitable — retake as instructed"*.
- [ ] Retire `src/ood_guard.py`, which was measured and found insufficient (catches 5 of 19).

---

## Priority 3 — Retrain as V2

Only after Priorities 1 and 2 have produced data.

- [x] 2026-11-12 — Retraining notebook ready: `colab/CWSB_retrain_colab.ipynb` (frozen split,
      versioned output, healthy-derived thresholds, old-vs-new comparison). Untested on Colab.
- [ ] Re-run Phases 1–2 on the enlarged dataset (Colab GPU).
- [ ] **Re-derive thresholds on real healthy data.** The current `T_low = 0.50` was set by
      hand because two validation photos could not support a principled choice.
- [ ] Hold out **at least 30 healthy plants** for validation so reported numbers mean something.
- [ ] Optional experiment: train on plant-bounding-box crops (real pixels, just bounded) and
      compare against full frames. Must be applied identically in training and serving.
- [ ] Optional experiment: learning curve — train on 25 / 50 / 75 / 100% of the data to
      measure empirically how much more data actually buys.

---

## Priority 4 — Phases 5 to 7 (productionisation)

- [ ] **Phase 5 — ONNX export.** Export the winning checkpoint to `cwsb_v1.onnx` with a
      4-decimal parity check against the PyTorch output. This is the file mobile integration
      consumes.
- [ ] **Phase 6 — FastAPI `/v1/predict`.** Quality gate → framing gate → model → thresholds.
      `src/predict_folder.py` already mirrors the intended behaviour; port it.
      Do **not** downscale input aggressively — full resolution is worth ~2 points of recall.
- [ ] **Phase 7 — farmer PWA.** Camera guide enforcing the whole-plant framing, the
      three-way verdict (no signs / retake / likely), and the retake prompts from the
      quality gate.
- [ ] Add the **agronomic risk score** (season, altitude, shade, recent rainfall) as a
      separate displayed figure alongside the photo verdict — never as a model input.

---

## Open questions

- Is `plant_012` mislabelled? Blocks a clean recall figure.
- What is the real-world healthy:infected ratio at screening time? It determines whether
  false alarms or misses are the costlier error, and therefore where `T_low` belongs.
- Does a bounding-box crop trained end-to-end beat the full frame? Unmeasured.

---

## Transfer list — what must move by pendrive, not Git

These are in `.gitignore` because of size. Total if you take everything: **~6.3 GB**.

**Essential (the friend cannot work without these) — ~110 MB**

| Folder / file | Size | Why |
|---|---|---|
| `models/v1/` | 107 MB | The trained model. `contender_B.pt` is the one in use. Without it no scoring, no reports, no experiments. |

**Needed to re-run the pipeline or retrain — ~2.9 GB**

| Folder | Size | Why |
|---|---|---|
| `Data_final/` | 2.9 GB | The source photographs. Everything else is derived from this. |

**Needed to reproduce the exact splits and numbers — ~2.6 GB (or regenerate instead)**

| Folder | Size | Why |
|---|---|---|
| `work/` | 2.6 GB | `clean/`, `splits/split.csv`, `tiles/`. **Regenerable** by running `src/clean.py` then `src/make_split.py` then `src/make_tiles.py` on `Data_final/` — seed 42 makes it identical. If pendrive space is tight, skip this and regenerate. But do copy `work/splits/split.csv` (a few KB) so the splits are provably identical. |

**Test sets — ~273 MB**

| Folder | Size | Why |
|---|---|---|
| `test/` | 67 MB | 12 blind samples + `answer_key.csv` |
| `test2/` | 54 MB | 144 images + `answer_key2.csv` — the "all types of images" set |
| `test_new/` | 152 MB | 30 photos that exposed the framing failure. Worth having: it is the clearest demonstration of the open problem. |

**Optional — ~94 MB**

| Folder | Size | Why |
|---|---|---|
| `reports/` | 94 MB | Generated PDFs and CSVs. All regenerable from the model + test sets. Copy only if the friend wants to read the evidence PDFs without re-running anything. |

**Skip these:** `Final_cwsb_Data/` (2.9 GB, a duplicate of `Data_final/`),
`*.zip` at the repo root (`Data_final.zip`, `cwsb_package.zip`, `test2_images.zip` — all
regenerable), `venv/`, `__pycache__/`.

**Minimum useful transfer: `models/v1/` + `Data_final/` + `work/splits/split.csv` ≈ 3.0 GB.**
Everything else can be regenerated or skipped.

After copying, the folders must sit at the repo root with exactly these names, next to `src/`.
Note that `models/v1/` must be flat — `contender_B.pt` directly inside it, not nested in a
further `models/v1/` folder (this nesting bug happened once when unzipping Colab output).
