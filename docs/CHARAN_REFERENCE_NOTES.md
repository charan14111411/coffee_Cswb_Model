# CWSB — my reference notes

Everything here is checked against the real files on 2026-11-19.
Background filter is **OFF** everywhere in this note. That is the normal way.

---

# PART 1 — THE INITIAL MODEL (v1)

## 1.1 Where the model is stored

Folder: `models/v1/`

| File | What it is |
|---|---|
| `contender_A.pt` | try 1 — frozen backbone + logistic regression (5 min). Lost. |
| `contender_B.pt` | try 2 — full fine-tune. **This one won.** 106 MB |
| `cwsb_v1.pt` | contender_B, renamed. **This is the model we use.** |
| `cwsb_v1.onnx` | same model, ONNX format (for the app / phone) |
| `config.json` | thresholds + `remove_bg: False`. **Must travel with the .pt** |
| `training_result.json` | the training numbers |
| `ood_reference.npz` | for the out-of-distribution guard |

⚠️ `models/` is in `.gitignore` → **not in git**. If I lose this folder it is gone.

Model: ConvNeXt-Tiny `convnext_tiny.fb_in22k_ft_in1k_384`, 384×384, 2 classes.
Class 1 = Unhealthy (infested).

## 1.2 What data trained v1

`old_data/`

| Folder | Photos |
|---|---|
| `old_data/Healthy/` | 28 |
| `old_data/Unhealthy(Infected)/` | 470 |
| **total** | **498** |

`Final_cwsb_Data/` is the **same 498 photos** + a `labels.csv`. Do not use both.

## 1.3 The 2 notebooks used (I was right, there are 2)

### Notebook 1 — training
`colab/CWSB_train_colab.ipynb` (6 cells)

```
Cell 1  GPU check + install
Cell 2  mount Drive, unzip cwsb_package.zip
Cell 3  train contender A   (~5 min)
Cell 4  train contender B   (~20-40 min on T4)
Cell 5  zip models/v1 and download  ->  models_v1.zip
```

### Notebook 2 — ONNX export
`colab/CWSB_onnx_export_colab.ipynb` (7 cells)

```
Cell 1  upload contender_B.pt  (106 MB)
Cell 2  rebuild the architecture, load weights
Cell 3  wrap (normalize + model + softmax + feature map) -> one .onnx
Cell 4  parity check, PyTorch vs ONNX must match to 4 decimals
Cell 5  test on my own photos, PURE ONNX (upload a zip of images)
```

## 1.4 Upload structure for v1 — IMPORTANT

**I do NOT upload `old_data/` to this notebook.**
`CWSB_train_colab.ipynb` expects **one file: `cwsb_package.zip`**, already split and prepared.

### Step 1 — run these on my PC first (in this order)

```
python src/clean.py          # old_data/  ->  work/clean/
python src/make_split.py     # -> work/splits/split.csv   (splits BY PLANT)
python src/make_tiles.py     # -> work/tiles/             (7 tiles per healthy photo)
python src/package_for_colab.py   # -> cwsb_package.zip
```

`package_for_colab.py` shrinks every photo so the short side is 768 px
(model trains at 384, so nothing is lost, upload ~20× smaller).

### Step 2 — what is inside cwsb_package.zip

```
cwsb_package.zip                    (629 entries)
│
├── work/clean/        500 photos   <- the cleaned images
├── work/splits/       split.csv    <- which photo is train / valid / test
├── work/tiles/        127 files    <- tiles + tiles.csv
└── src/train.py                    <- the training script rides along
```

### Step 3 — where to put it

Google Drive → **MyDrive → top level** (not in a sub-folder).
The notebook hard-codes `/content/drive/MyDrive/cwsb_package.zip`.

### Step 4 — ONNX notebook upload

That one takes **no zip**. Browser-upload `contender_B.pt` when Cell 1 asks.

## 1.5 v1 results (for reference)

```
Thresholds:  < 0.50  no visible signs
             0.50-0.89  inconclusive, retake
             >= 0.90  WSB likely
Plant score = MAX of that plant's photos

Test, photo level : caught 44, missed 3, false alarm 1, cleared 7, inconclusive 4
Test, plant level : caught 5, missed 0, false alarm 1, inconclusive 2
```

---

# PART 2 — RETRAINING

There are **2 retrain notebooks**. Both already include ONNX inside, so
**I do not need the ONNX notebook again.**

| Notebook | Zip names | When to use |
|---|---|---|
| `CWSB_retrain_end2end_colab.ipynb` | **fixed names required** | the old way |
| `CWSB_full_pipeline_colab.ipynb` | **any names** | 👈 **use this one** |

Both notebooks do the whole job:
`clean → split → tiles → train → thresholds → test → heatmap PDF → ONNX → download`

⚠️ Big difference from v1: here I upload **raw photos**, not `cwsb_package.zip`.
The notebook runs `clean.py` / `make_split.py` / `make_tiles.py` itself inside Colab.

---

## 2.1 THE FOLDER-NAME RULE (both notebooks)

The notebook decides healthy vs unhealthy **only from the folder name**:

```
name contains "unhealthy" or "infect"   ->  Unhealthy
else name contains "health"             ->  Healthy
else                                    ->  SKIPPED (warning printed)
```

### ⚠️ MY TRAP

`images_health_16_11_v2/` contains the word **health**
→ my 211 **INFESTED** photos would go into **Healthy** and ruin the training.

**I must rename it to `unhealthy_16_11_v2` before zipping.**

Also: do not put `healthy` / `unhealthy` / `infect` in the **zip file name** —
the zip name is part of the path too.

---

## 2.2 Option A — `CWSB_full_pipeline_colab.ipynb` (recommended)

28 cells. Takes **any number of zips, any names**. Unzips zips-inside-zips.
Finds a `.pt` by itself for old-vs-new comparison.

### Cell 2 — the only cell I edit

```python
VERSION_NUMBER    = 2
REMOVE_BACKGROUND = False     # <- False, not 'auto'
BG_FILL           = 'blur'    # ignored when False
USE_DRIVE         = True      # False = upload from laptop
```

**Before running:** Runtime → Change runtime type → **T4 GPU**
(changing it later wipes all files).

### Upload structure

One zip, name it `cwsb_data.zip`:

```
cwsb_data.zip
│
├── Healthy/                 <- 28   from old_data/Healthy/
├── Unhealthy(Infected)/     <- 470  from old_data/Unhealthy(Infected)/
└── unhealthy_16_11_v2/      <- 211  from images_health_16_11_v2/  (RENAMED!)
```

Optional: put `cwsb_v1.pt` anywhere inside the zip → notebook uses it to
compare old model vs new model on the same test photos.

Put `cwsb_data.zip` in **MyDrive top level**.

### Check after Cell 4

```
images found    : 709
sorted Healthy  : 28
sorted Unhealthy: 681
```

If `sorted Healthy` shows **239**, the rename failed → STOP and fix.

### Notebook order (11 sections)

```
1  Settings            <- only cell I edit
2  Load your photos
3  Write project scripts (clean / split / tiles / train / remove_bg)
4  Background filter check
5  Prepare the data    (clean -> split -> tiles)
6  Train               (~30-60 min on T4)
7  Thresholds
8  Results, new vs old model
9  Heatmap PDF
10 ONNX export + parity check
11 Download
```

---

## 2.3 Option B — `CWSB_retrain_end2end_colab.ipynb` (the old way)

16 cells. **Zip names are fixed.** Needs exactly:

```
MyDrive/
├── old_data.zip       REQUIRED
├── new_data.zip       REQUIRED
└── old_models.zip     optional (for comparison)
```

Inside each zip, same folder-name rule:

```
old_data.zip
├── Healthy/                 28
└── Unhealthy(Infected)/     470

new_data.zip                       <- can hold zips inside (1 level)
└── unhealthy_16_11_v2/      211

old_models.zip
└── cwsb_v1.pt
```

Settings: Cell 1 → `VERSION_NUMBER = 2`, `REMOVE_BACKGROUND = False`
Cell 2 → `USE_DRIVE = True` (False = browser upload)

Duplicate file names get prefixed `old_` / `new_` automatically.

---

## 2.4 What I get back

`cwsb_v2_release.zip`:

```
cwsb_v2.pt                  the new model
cwsb_v2.onnx                for the app
config_v2.json              thresholds + remove_bg flag
cwsb_v2_test_report.pdf     heatmaps, failures first
split_v2.csv                which photo went where
training history
```

On my PC: put the `.pt` and `config.json` together in `models/v2/`.
**Never separate them.**

---

# PART 3 — RUNNING THE MODEL ON MY PC

```
python src/predict_folder.py images_health_16_11_v2
```

Files it needs:

```
models/v1/cwsb_v1.pt
models/v1/config.json       <- decides if filter is ON (it is False)
src/predict_folder.py
src/remove_bg.py            <- must exist, stays OFF
```

Photo folders I can test on:

| Folder | Photos | Note |
|---|---|---|
| `images_health_16_11_v2/` | 211 | **INFESTED** (name is misleading) |
| `images_health_16_11/` | 16 | |
| `images_16_11/` | 30 | |
| `test_new/` | 30 | |
| `test2/` | 144 | has `answer_key2.csv` |
| `test/` | 12 | has `answer_key.csv` |

PDF reports:

```
python src/make_report_pdf.py          -> heatmap PDF
python src/make_labels_proof_pdf.py    -> manager proof PDF (needs labels/)
```

`labels/` = 211 YOLO box files, 645 boxes, from the labelling team.

---

# PART 4 — RULES I MUST NOT BREAK

1. **Training and serving must match.** Filter ON in training = ON in serving.
   `config.json` decides. **Never edit `remove_bg` by hand on a trained model.**

2. **v1 cannot get the filter.** v1 was trained on full frames.
   Turning the filter on with v1: infested score falls **0.97 → 0.20**. Wrong answers.

3. **Thresholds come from HEALTHY validation scores**, never from infected
   quantiles. That old bug collapsed both thresholds to ~0.91.

4. **Split BY PLANT**, never by photo. Same plant in train and test = cheating.
   `make_split.py` aborts if it happens.

5. **Always train from ImageNet weights** on old + new photos together.
   Do not continue training the old `.pt` — it forgets.

6. The filter only works on **close-up stem photos**. On whole-plant photos
   the check passes only ~5%. That is why it stays OFF.

---

# PART 5 — NOTEBOOKS I SHOULD IGNORE

| Notebook | Why |
|---|---|
| `CWSB_v2_convnext_colab.ipynb` | old retrain attempt, replaced |
| `CWSB_detector_colab.ipynb` | YOLO **detector** (boxes), different job |
| `CWSB_background_filter_colab.ipynb` | filter only, no training |

Note on the detector: v1 decides correctly (93.8% on the 211 labelled photos)
but does **not** point at the right place — peak inside box only 9.7%
(random chance = 5.1%). For location I need the **detector**, not the classifier.
