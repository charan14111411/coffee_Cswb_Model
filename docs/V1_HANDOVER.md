# CWSB v1 — handover

For the engineer (and the engineer's coding agent) taking over **version 1** and its
retraining. Written 2026-11-20. Every setting below was read out of the code in this
repo, not from memory.

**Scope split:** v1 and its retraining belong to this document. The folder
`version_rbn/`, the notebook `colab/CWSB_version_rbn_colab.ipynb` and
`models/version_rbn/` belong to Charan. Do not edit those.

---

## 1. What v1 does

A **binary image classifier** for Coffee White Stem Borer.

```
one photo  ->  model  ->  one number between 0 and 1
```

The number is the probability of **Unhealthy (infested)**. Class index 1 = Unhealthy.
It is a classifier, not a detector: it scores the whole frame and does not localise.

Two frozen thresholds turn the number into a verdict:

```
score <  0.50          ->  no_visible_signs
0.50 <= score < 0.90   ->  inconclusive   (photograph again)
score >= 0.90          ->  wsb_likely

a plant's score = the MAXIMUM over that plant's photos
```

### Held-out test results (recorded in `models/v1/config.json`)

| Level | caught | missed | false alarms | cleared | inconclusive |
|---|---|---|---|---|---|
| photo | 44 | 3 | 1 | 7 | 4 |
| plant | 5 | 0 | 1 | 0 | 2 |

---

## 2. The model

```
backbone   convnext_tiny.fb_in22k_ft_in1k_384     (timm)
params     27.8 M
input      384 x 384 RGB
classes    2        [Healthy, Unhealthy]
head       Linear(768 -> 2)
seed       42
```

Pretrained on ImageNet-22k, fine-tuned on ImageNet-1k, then fine-tuned here.

### Files in `models/v1/` (gitignored — must be copied by hand)

| File | What |
|---|---|
| `contender_A.pt` | try 1, frozen backbone + logistic regression. Lost. |
| `contender_B.pt` | try 2, full fine-tune. **Winner.** 107 MB |
| `cwsb_v1.pt` | byte-identical copy of `contender_B.pt` |
| `cwsb_v1.onnx` | same model exported to ONNX |
| `config.json` | thresholds, the `remove_bg` flag, test results |
| `training_result.json` | validation PR-AUC of both contenders |
| `ood_reference.npz` | reference features for the out-of-distribution guard |

SHA-256 of `contender_B.pt` starts `f63f0c25a012006f84cef9efd9edf59`.

**`config.json` must always travel with the `.pt`.** Serving reads it.

---

## 3. The data v1 was trained on

```
old_data/
├── Healthy/                 28 photos
└── Unhealthy(Infected)/    470 photos
                            ---
                            498
```

Photos are not the unit — **plants** are. Photos of one plant were grouped together:

```
Healthy      9 plants
Unhealthy   69 plants
```

`Final_cwsb_Data/` holds the same 498 photos plus a `labels.csv`. Use one, not both.
Both folders are gitignored.

---

## 4. The pipeline — four scripts, in this order

```
Data_final/                      <- INPUT FOLDER NAME (see the trap below)
   |
   |  python src/clean.py
   v
work/clean/{Healthy,Unhealthy}/  + images.csv + duplicates.csv
   |
   |  python src/make_split.py
   v
work/splits/split.csv            file,label,group,split
   |
   |  python src/make_tiles.py
   v
work/tiles/Healthy/ + tiles.csv
   |
   |  python src/train.py --mode both
   v
models/v1/contender_A.pt, contender_B.pt, training_result.json
```

### TRAP — `clean.py` reads a folder named `Data_final`

```python
SRC = ROOT / "Data_final"
CLASSES = {"Healthy": "Healthy", "Unhealthy(Infected)": "Unhealthy"}
```

It does **not** read `old_data/`, even though `old_data/` has exactly the right shape.
If the folder is missing, `clean.py` finds zero images without complaining and every
later stage fails. First command on a fresh machine:

```powershell
Copy-Item -Recurse old_data Data_final
```

`Data_final/` must contain subfolders named exactly `Healthy` and `Unhealthy(Infected)`.

### 4a. `clean.py`

Opens every `*.jpg`, applies the EXIF rotation, strips EXIF (removing GPS), re-saves at
JPEG quality 95 into `work/clean/`. Measures and flags:

```
BLUR_MIN         = 60.0    Laplacian variance below this -> "blurry"
CLIPPED_MAX      = 0.20    >20% near-white  -> "clipped"
DARK_MAX         = 0.40    >40% near-black  -> "dark"
NEAR_DUP_HAMMING = 6       dhash distance <= 6 -> near-duplicate
```

On the current data: 496 clean, 2 flagged `blurry`.

### 4b. `make_split.py` — the most important script

Groups photos into **pseudo-plants**, then splits the *plants* 70/15/15 per class,
seed 42.

```
same plant if: consecutive frame numbers with gap <= 4
            or timestamps within 120 seconds
            (within the same class and the same camera series)
```

It **refuses to write the file** if any plant lands in two splits, and warns about
near-duplicates that cross splits.

> Never split by photo. Five photos of one plant spread across train and test means the
> model has already seen that plant, and the test score becomes meaningless. This script
> exists to make that impossible.

v1's split:

```
train     healthy  18    unhealthy 364
val       healthy   2    unhealthy  53
test      healthy   8    unhealthy  51
excluded  healthy   0    unhealthy   2   (blurry)
```

### 4c. `make_tiles.py`

The healthy class is tiny, so each **train-split healthy photo only** (never val or
test) is cut into 7 overlapping tiles: a 3x2 grid plus one centre tile, each 55% of the
short edge. Tiles inherit their photo's plant, so no leakage is introduced.

### 4d. `train.py`

```
python src/train.py --mode A      # frozen backbone + logistic regression
python src/train.py --mode B      # full fine-tune
python src/train.py --mode both   # runs both, prints the winner
```

**Selection metric: validation PR-AUC** (`average_precision_score`). Not accuracy.

Contender B, the one that shipped:

| Setting | Value |
|---|---|
| epochs | 40 max, early stop patience 6 |
| epochs 0–1 | head only, lr 1e-3, body lr 0 (warm-up) |
| epoch 2+ | head and body, both lr 1e-4 |
| optimizer | AdamW, weight_decay 0.05 |
| scheduler | CosineAnnealingLR, `T_max = len(dataloader)` |
| batch | 16 |
| sampler | `WeightedRandomSampler`, inverse class frequency, with replacement |
| loss | `CrossEntropyLoss` with class weights |
| checkpoint | written **every time** validation PR-AUC improves |

Training augmentation:

```python
RandomResizedCrop(384, scale=(0.6, 1.0))
RandomHorizontalFlip()
RandomRotation(10)
ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10)
RandomApply([GaussianBlur(5)], p=0.15)
ToTensor(); Normalize(ImageNet mean/std)
```

Evaluation transform — the code comments say this must never change:

```python
Resize(416)        # 384 + 32
CenterCrop(384)
ToTensor(); Normalize(ImageNet mean/std)
```

Training and serving must use the identical eval transform, or the scores drift.

---

## 5. Thresholds — `src/set_thresholds.py`

Reads `reports/failure_analysis.csv`, writes `models/v1/config.json`.

```
T_low  = 0.50
T_high = 0.90
```

### The bug that must not come back

The first recipe took **both** thresholds from the *infected* score distribution. The
model early-stopped at validation PR-AUC 1.0, so validation infected scores were nearly
all above 0.99. `quantile(0.02)` and `quantile(0.10)` collapsed onto each other —
0.9071 and 0.9149, an inconclusive band 0.008 wide — and every hedged score between 0.6
and 0.9 was announced as **"no visible signs"**. The most dangerous possible failure.

The replacement rule, calibrated on validation only:

- validation healthy photos scored **0.26 and 0.39**
- so `T_low = 0.50` sits clearly above the healthy band
- `T_high = 0.90` is strong confidence; validation infected plants all sit above it

> **Thresholds come from HEALTHY validation scores. Never from infected quantiles.**
> The test split checks the choice. It never makes it.

---

## 6. Running predictions

```powershell
python src/predict_folder.py <folder-of-photos>
```

It reads `models/v1/config.json` and obeys it, including:

```json
"remove_bg": false,
"remove_bg_fill": "blur"
```

`remove_bg` records whether the model was **trained** with background removal. v1 was
trained on full frames, so it is `false` and serving removes nothing.

> **Never flip `remove_bg` by hand on a trained model.** Training and serving must
> match. Turning it on for v1 was measured: infested stem scores fell from 0.97 to 0.20.

Reports:

```powershell
python src/make_report_pdf.py          # Grad-CAM heatmap PDF
python src/make_labels_proof_pdf.py    # box-overlap proof PDF, needs labels/
```

---

## 7. The two notebooks that produced v1

### `colab/CWSB_train_colab.ipynb` — training

Expects **one file**: `cwsb_package.zip`, at the **top level of MyDrive**. The path is
hard-coded:

```python
!unzip -q /content/drive/MyDrive/cwsb_package.zip -d /content/cwsb
```

Build that zip locally **after** clean / split / tiles:

```powershell
python src/package_for_colab.py
```

It shrinks every photo so the short side is 768 px — the model trains at 384, so no
practical detail is lost and the upload is about 20x smaller. Contents:

```
cwsb_package.zip                629 entries, 190 MB
├── work/clean/        500 photos at 768 px short side
├── work/splits/       split.csv
├── work/tiles/        127 files
└── src/train.py
```

Cells: GPU check -> unzip -> train A -> train B -> download `models_v1.zip`.

### `colab/CWSB_onnx_export_colab.ipynb` — ONNX

Takes **no zip**. Browser-upload `contender_B.pt` when cell 1 asks. It rebuilds the
architecture, wraps normalize + model + softmax + feature map into one graph, exports,
then runs a **4-decimal parity check** between the full PyTorch path and the full ONNX
path.

---

## 8. Retraining with new data

Two notebooks can do it. **Both already include the ONNX export**, so the ONNX notebook
is not needed again.

### Recommended: `colab/CWSB_full_pipeline_colab.ipynb`

Accepts **any number of zips with any names**, unwraps zips nested inside zips (three
passes), and finds a `.pt` by itself to compare against.

Here you upload **raw photos**, not `cwsb_package.zip`. The notebook runs clean, split
and tiles itself inside Colab.

#### Upload structure

One zip at the **top level of MyDrive**:

```
cwsb_data.zip
├── Healthy/                 <- 28 photos   from old_data/Healthy/
├── Unhealthy(Infected)/     <- 470 photos  from old_data/Unhealthy(Infected)/
└── <your_new_folder>/       <- the new photos
```

Optionally drop `cwsb_v1.pt` anywhere inside the zip; the notebook finds it and compares
old model against new model on the same test photos.

#### The folder-name rule — read this before zipping

Photos are routed to a class by **folder name**, matched against the whole relative
path, lowercased:

```python
if 'unhealthy' in path or 'infect' in path:  ->  Unhealthy
elif 'health'  in path:                      ->  Healthy
else:                                        ->  SKIPPED (warning printed)
```

Three consequences:

1. A folder named `images_health_16_11_v2` contains the word **health**, so its photos
   are filed as **Healthy** — even when they are infested. Rename such folders to
   something containing `unhealthy`, for example `unhealthy_16_11_v2`.
2. The **zip file name is part of the path**. Do not put `healthy`, `unhealthy` or
   `infect` in the zip's own name unless you mean it.
3. Anything matching neither word is skipped. The notebook prints a warning listing the
   first 10 skipped files. Read that warning.

#### Settings — only cell 2 is edited

```python
VERSION_NUMBER    = 2          # outputs become cwsb_v2.pt / cwsb_v2.onnx
REMOVE_BACKGROUND = False      # 'auto' lets the notebook decide; False = v1 behaviour
BG_FILL           = 'blur'     # ignored when False
USE_DRIVE         = True       # False = upload from the browser
```

**Set Runtime -> Change runtime type -> T4 GPU before running any cell.** Changing it
later wipes every file in the session.

#### Check after the ingest cell

```
images found    : 709
sorted Healthy  : 28
sorted Unhealthy: 681
```

If `sorted Healthy` is larger than expected, a folder name has routed infested photos
into Healthy. Stop and fix the zip.

#### What the notebook does, in order

```
1  settings + GPU check
2  load photos from the zips
3  write clean.py / make_split.py / make_tiles.py / train.py / remove_bg.py
4  background filter check
5  clean -> split (leak-proof) -> tiles
6  train ConvNeXt (contender B recipe)    ~30-60 min on T4
7  thresholds from HEALTHY validation scores
8  held-out test, new model vs old model
9  Grad-CAM heatmap PDF
10 ONNX export + 4-decimal parity check
11 download the release zip
```

Output `cwsb_v2_release.zip`:

```
cwsb_v2.pt
cwsb_v2.onnx
config_v2.json              thresholds AND the remove_bg flag
cwsb_v2_test_report.pdf     heatmaps, failures first
split_v2.csv
training history
```

On the local machine put the `.pt` and `config_v2.json` together in `models/v2/`.
**Never separate them.**

### Alternative: `colab/CWSB_retrain_end2end_colab.ipynb`

Same job, but the zip names are **fixed**:

```
MyDrive/
├── old_data.zip       REQUIRED
├── new_data.zip       REQUIRED   (may contain zips one level deep)
└── old_models.zip     optional   (a .pt to compare against)
```

The same folder-name routing applies inside each zip. Duplicate filenames are prefixed
`old_` or `new_` automatically. Settings live in cell 1 (`VERSION_NUMBER`,
`REMOVE_BACKGROUND`) and cell 2 (`USE_DRIVE`).

---

## 9. Rules that must not be broken

1. **Split by plant, never by photo.** `make_split.py` aborts rather than leak.
2. **Touch the test split once**, at the very end. Tuning on it makes it worthless.
3. **Thresholds from healthy validation scores**, never from infected quantiles.
4. **Training and serving must match.** The eval transform
   (`Resize(416) -> CenterCrop(384)`) and the `remove_bg` flag are both part of that
   contract. `config.json` records it; serving obeys it.
5. **Always train from ImageNet weights** on old and new photos together. Continuing to
   train the old `.pt` makes it forget the old photos. The old `.pt` is only ever a
   baseline to compare against.
6. **`models/` is gitignored.** Model files move by hand. Back them up before any run.

---

## 10. What is and is not in git

| In git | Copied by hand (gitignored) |
|---|---|
| `src/`, `colab/`, `version_rbn/src/` | `old_data/`, `Final_cwsb_Data/` |
| `labels/` (211 YOLO label files) | `models/` |
| `reports/*.csv` | `work/`, `version_rbn/work/` |
| `docs/`, the `*.md` handover files | `cwsb_package.zip`, every `*.zip`, every `*.pt` |
| `.gitignore` | `reports/*.pdf`, `data/`, `test*/`, `images_*/` |

Everything gitignored is large, regenerable, or both — except `old_data/` and `models/`,
which are neither and must be copied.

---

## 11. Known limits of v1

- It decides correctly on 198 of 211 labelled photos (93.8%), but it does **not**
  localise. Peak attention falls inside a labelled box only 9.7% of the time, against
  5.1% expected by chance.
- It leans heavily on the background. With the stem blacked out entirely, 78 of 100
  infested photos are still called infested. Blurred down to 64x64, where a bore hole
  cannot exist, 42 of 100 still are.
- The root cause is resolution plus source bias. Photos are about 3000x4000 and the
  model sees 384x384, so a 100 px bore hole arrives as roughly 12 px. On top of that,
  268 infested photos come from one camera that never photographed a healthy plant, so
  the camera alone predicts the class.
- Charan's `version_rbn` addresses this separately and does not change v1.

If a retrained v2 is to be a real improvement rather than a repeat, the single most
valuable input is **more healthy plants photographed the same way the infested ones
were** — same camera, same distance, same angle, close to the stem.
