# Exercise 1: Object Detection — NorgesGruppen Data

Detect and classify products on grocery shelves (NM i AI 2026).

---

## Application status: **complete, tested, and submitted (baseline)**

| Item | Description |
|------|-------------|
| **What it does** | Converts COCO annotations → YOLO format, trains YOLOv8 (Ultralytics 8.1.0), writes JSON predictions and a **submission zip** with `run.py` + `best.pt`. |
| **On-disk layout** | After unpacking the official zip, data live under `train/images/` and `train/annotations.json` (not `annotations.json` at the dataset root). |
| **Classes** | The real `annotations.json` has **356 categories** with contiguous IDs **0–355**; `unknown_product` is ID **355** (some docs mention 356 — verify against your file). |
| **Local validation** | `validate_competition.py` reproduces the 0.7/0.3 hybrid metric with pycocotools. |
| **Submission smoke test** | `test_run_local.py` runs `submission/run.py` like the sandbox. |
| **Official run** | Uploaded `submission.zip` → **Evaluation complete**, score **0.3889**, ~**18.8 s** GPU sandbox time. |
| **Baseline model submitted** | YOLOv8**s**, imgsz **640**, trained on **CPU** (`experiments/exp01_yolov8s_640_cpu/`). |

**GPU upgrade (done in Colab + Drive):** **YOLOv8m** @ **1280** trained in **`colab_train_yolov8m.ipynb`**; `best.pt` → `submission/best.pt` → `build_submission.py`. See **§ Google Colab & Google Drive** below and root **`README.md`**.

---

## Competition summary

| Parameter | Value |
|-----------|-------|
| Training images | 248 |
| Annotations | ~22.7k COCO boxes |
| Categories (actual file) | 356 (IDs 0–355) |
| Scoring | `0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5` |
| Sandbox | Python 3.11, NVIDIA L4, ultralytics **8.1.0**, **300 s** timeout |
| Submission zip | Max 420 MB uncompressed, `run.py` at zip root |

---

## Data paths (after extracting the zips)

```
data/raw/NM_NGD_coco_dataset/
  NM_NGD_coco_dataset.zip          # optional once extracted
  train/
    images/                        # 248 images
    annotations.json
data/raw/NM_NGD_product_images/
  ... (product reference images — optional for later classification boosts)
```

---

## Local setup

```bash
pip install ultralytics==8.1.0
pip install torch==2.6.0 torchvision==0.21.0
pip install pycocotools pandas
```

On Windows, if loading `.pt` fails, `train_yolov8.py` patches `torch.load` with `weights_only=False` for PyTorch ≥ 2.6.

---

## Pipeline — commands (correct paths)

### 1. Analyse the dataset

```bash
python src/analyze_dataset.py \
  --annotations data/raw/NM_NGD_coco_dataset/train/annotations.json \
  --output_dir data/interim
```

*(On Windows PowerShell, use backticks `` ` `` or put arguments on one line.)*

### 2. COCO → YOLO

```bash
python src/prepare_dataset.py \
  --annotations data/raw/NM_NGD_coco_dataset/train/annotations.json \
  --images_dir data/raw/NM_NGD_coco_dataset/train/images \
  --output_dir data/yolo \
  --val_ratio 0.2 \
  --seed 42
```

### 3. Train (GPU locally if available, or `--device cpu`)

```bash
python src/train_yolov8.py --data data/yolo/dataset.yaml --model yolov8m.pt --imgsz 1280 --epochs 80 --batch 8 --project experiments --name exp_yolov8m_1280
```

### 4. Local inference

```bash
python src/infer_local.py --model experiments/<exp>/weights/best.pt --input_dir data/raw/NM_NGD_coco_dataset/train/images --output predictions.json --imgsz 640 --device cpu
```

### 5. Competition-style metric

```bash
python src/validate_competition.py --annotations data/raw/NM_NGD_coco_dataset/train/annotations.json --predictions predictions.json
```

### 6. Copy weights, smoke test, build zip

**Windows (PowerShell or CMD):**

```bat
copy experiments\exp01_yolov8s_640_cpu\weights\best.pt submission\best.pt
python src/test_run_local.py --run_py submission/run.py --input_dir data/raw/NM_NGD_coco_dataset/train/images --output test_predictions.json
python src/build_submission.py --submission_dir submission --output_zip submission.zip
```

Upload **`submission.zip`** on the NorgesGruppen submit page.

---

## Google Colab & Google Drive (GPU training)

Train **YOLOv8m @ 1280** on [Google Colab](https://colab.research.google.com) using **`colab_train_yolov8m.ipynb`** in this folder. Below is the workflow we used in practice (dataset on **Drive**, training in **`/content`**, submission built **on your PC**).

### 1. Dataset zip (`NM_NGD_coco_dataset.zip`, ~864 MB)

- **Option A — Google Drive:** Upload **`NM_NGD_coco_dataset.zip`** to Drive (e.g. My Drive). In Colab, mount Drive and unzip to `/content/dataset` (see notebook “Step 1”).
- **Option B — No Drive mount (recommended if `drive.mount` fails):** Keep the zip shared on Drive and download inside Colab with **`gdown`**, then unzip:
  - `pip install gdown`
  - `gdown --fuzzy "<file/link>" -O /content/NM_NGD_coco_dataset.zip`
  - **`unzip -o -q`** … **`-d /content/dataset`** — use **`-o`** (overwrite) so `unzip` does not stop with `replace? [y/n/...]` (Colab cannot answer that prompt easily).
- After unzip you should have: **`/content/dataset/train/annotations.json`** and **`/content/dataset/train/images/`**.

### 2. Colab runtime

- **Runtime → Change runtime type → GPU** (e.g. **T4**).
- After **Restart runtime**, `/content` is empty → rerun data download + Steps 1–3 before training.

### 3. Notebook steps (summary)

| Step | What it does |
|------|----------------|
| **1** | Get dataset into `/content/dataset` (Drive + unzip **or** `gdown` + `unzip -o`). |
| **2** | `pip install ultralytics` (notebook pins **`ultralytics>=8.3.0`** in code; **sandbox uses 8.1.0** — for maximum compatibility you can train with **`ultralytics==8.1.0`** to match the platform). Verify **CUDA** + GPU name. |
| **3** | Convert COCO → YOLO under **`/content/yolo_data`** (`dataset.yaml`). |
| **4** | Train: `YOLO('yolov8m.pt')` + `model.train(..., project='/content/runs', name='yolov8m_1280', imgsz=1280, ...)`. |
| **5** (optional) | Check `best.pt` size; copy to Drive **only if** Drive is mounted. |

**Outputs:** weights live under **`/content/runs/yolov8m_1280/weights/best.pt`** (and `last.pt`). The notebook’s “Step 5” path assumes that `name` and `project`; change paths if you rename them.

### 4. Colab gotchas we hit (and fixes)

| Issue | Fix |
|--------|-----|
| **`drive.mount` → `ValueError: mount failed`** | Use **Option B (`gdown`)** or another account / permissions. |
| **PyTorch 2.6+ `UnpicklingError` / `weights_only`** | Patch **`torch.serialization.load`** and **`torch.load`** **once** (see notebook), using **`_real_load = _ts.load` before replacing** — avoids **RecursionError** if the cell is run twice. |
| **`wandb` `UsageError`: project name `/content/runs`** | **`pip uninstall -y wandb`** in Colab, and/or **`WANDB_MODE=disabled`** before importing Ultralytics. |
| **`CUDA out of memory` (T4 ~15 GB)** | Lower **`batch`** (e.g. 8→4→2→1), lower **`imgsz`** if needed, optional **`PYTORCH_ALLOC_CONF=expandable_segments:True`**. |
| **`unzip` stuck on `replace?`** | Use **`unzip -o -q ...`** or `rm -rf /content/dataset` then unzip again. |

### 5. After training: **do not zip the whole Colab `runs/` folder for the competition**

The platform expects **`submission.zip`** built **locally** with only **`run.py`** + **`best.pt`** at the zip root.

1. In Colab **Files** panel: **`content` → `runs` → `yolov8m_1280` → `weights` → `best.pt`** → **Download**.
2. On your PC, replace **`object_detection_ng/submission/best.pt`**.
3. Ensure **`submission/run.py`** uses the same **`IMGSZ`** as training (default **1280** for this Colab flow).
4. Build the upload zip:

```bat
cd object_detection_ng
python src/build_submission.py --submission_dir submission --output_zip submission.zip
```

5. Smoke test (optional but recommended):

```bat
python src/test_run_local.py --run_py submission/run.py --input_dir data/raw/NM_NGD_coco_dataset/train/images --output test_predictions.json --timeout 300
```

6. Upload **`submission.zip`** to the competition site (max **420 MB**).

### 6. Platform sandbox vs Colab

- Sandbox: **Python 3.11**, **Ultralytics 8.1.0**, **PyTorch 2.6**, **~8 GB host RAM**, **300 s** total time, **`run.py`** restrictions (see below). **`submission/run.py`** is patched for **`weights_only`** and sets **`workers=0`** to reduce RAM use. If evaluation still fails, align **Ultralytics version** (retrain with **8.1.0**) and match **`imgsz`** to training.

---

## Project layout

```
object_detection_ng/
├── README.md
├── colab_train_yolov8m.ipynb   # YOLOv8m @1280 GPU training
├── requirements.txt
├── .gitignore
├── data/raw/ ...               # Data (do not commit large zips/weights)
├── data/yolo/                  # Generated YOLO dataset
├── data/interim/               # analyse_dataset reports
├── src/
│   ├── analyze_dataset.py
│   ├── prepare_dataset.py
│   ├── train_yolov8.py
│   ├── validate_competition.py
│   ├── infer_local.py
│   ├── test_run_local.py
│   └── build_submission.py
├── submission/
│   ├── run.py                  # Sandbox entrypoint
│   └── best.pt                 # Replace after each training run
├── experiments/                # Ultralytics outputs
└── submission.zip              # Upload artefact (regenerate as needed)
```

---

## Sandbox: blocked imports in submitted code

Do not use in `run.py`: `os`, `sys`, `subprocess`, `socket`, `pickle`, `shutil`, `yaml`, etc. Use **`pathlib`** and **`json`**. See `docs/Submission.md`.

---

## Experiment log

| ID | Model | imgsz | Local val (train)\* | Leaderboard | Notes |
|----|-------|-------|---------------------|-------------|--------|
| exp01_yolov8s_640_cpu | yolov8s | 640 | ~0.416 hybrid | **0.3889** | Submitted baseline; ~19 s sandbox |
| Colab yolov8m_1280 | yolov8m | 1280 | — | TBD | See **§ Google Colab & Google Drive**; `best.pt` → `submission/` → `build_submission.py` |

\*Local validation uses the same 248 training images; the hidden test set differs → leaderboard numbers will not match local metrics exactly.

---

## Inference defaults in `run.py`

| Parameter | Default | Values to try |
|-----------|---------|----------------|
| imgsz | 1280 | Match training (e.g. 640 if the `.pt` was trained at 640) |
| conf | 0.10 | 0.08, 0.12 |
| iou | 0.60 | 0.55, 0.65 |
| max_det | 400 | 300–500 on very dense shelves |

After switching `best.pt` to a model trained at 1280, keep **`imgsz=1280`** in `run.py` (already the default) to match training.
