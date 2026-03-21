# Exercise 1: Object Detection — NorgesGruppen Data

Detect and classify grocery products on Norwegian store shelves.

## Competition Summary

| Parameter | Value |
|-----------|-------|
| Images | 248 shelf images |
| Annotations | ~22,700 COCO-format bounding boxes |
| Categories | 357 (IDs 0-355 = products, ID 356 = unknown_product) |
| Scoring | `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5` |
| Sandbox | Python 3.11, NVIDIA L4 24GB, CUDA 12.4, 300s timeout |
| Key lib | ultralytics 8.1.0 (pre-installed) |
| Max zip | 420 MB uncompressed, max 3 weight files, max 10 .py files |
| Submissions | 3/day per team |

## Strategy

**Phase A — Baseline** (current focus)  
YOLOv8m fine-tuned at 1280px → valid submission ASAP.

**Phase B — Tuning**  
Compare yolov8m vs yolov8l, tune conf/iou/max_det, image size.

**Phase C — Classification boost**  
Product reference images, class balancing, optional two-stage re-ranking.

---

## Setup

### 1. Install dependencies

```bash
pip install ultralytics==8.1.0
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install pycocotools pandas
```

### 2. Download and extract data

Download from the competition website and extract into:
```
data/raw/NM_NGD_coco_dataset/
    images/          # 248 shelf images
    annotations.json # COCO annotations
data/raw/NM_NGD_product_images/
    metadata.json
    <product_code>/main.jpg, front.jpg, ...
```

---

## Pipeline — Step by Step

### Step 1: Analyze the dataset

```bash
python src/analyze_dataset.py \
    --annotations data/raw/NM_NGD_coco_dataset/annotations.json \
    --output_dir data/interim
```

Outputs: `data/interim/dataset_summary.json`, `data/interim/category_stats.csv`

**Critical checkpoint:** Confirm exact category count, ID range, and unknown_product presence.

### Step 2: Convert COCO → YOLO format

```bash
python src/prepare_dataset.py \
    --annotations data/raw/NM_NGD_coco_dataset/annotations.json \
    --images_dir data/raw/NM_NGD_coco_dataset/images \
    --output_dir data/yolo \
    --val_ratio 0.2 \
    --seed 42
```

Outputs: `data/yolo/dataset.yaml`, images + labels in train/val splits.

### Step 3: Train YOLOv8

```bash
# Experiment 1: YOLOv8m baseline
python src/train_yolov8.py \
    --data data/yolo/dataset.yaml \
    --model yolov8m.pt \
    --imgsz 1280 \
    --epochs 80 \
    --batch 8 \
    --project experiments \
    --name exp01_yolov8m_1280

# Experiment 2: YOLOv8l comparison
python src/train_yolov8.py \
    --data data/yolo/dataset.yaml \
    --model yolov8l.pt \
    --imgsz 1280 \
    --epochs 80 \
    --batch 4 \
    --project experiments \
    --name exp02_yolov8l_1280
```

### Step 4: Local inference

```bash
python src/infer_local.py \
    --model experiments/exp01_yolov8m_1280/weights/best.pt \
    --input_dir data/raw/NM_NGD_coco_dataset/images \
    --output predictions.json \
    --imgsz 1280 \
    --conf 0.10 \
    --iou 0.60 \
    --max_det 400
```

### Step 5: Validate against competition scoring

```bash
python src/validate_competition.py \
    --annotations data/raw/NM_NGD_coco_dataset/annotations.json \
    --predictions predictions.json
```

Reports: detection_mAP@0.5, classification_mAP@0.5, and final hybrid score.

### Step 6: Copy best weights to submission

```bash
cp experiments/exp01_yolov8m_1280/weights/best.pt submission/best.pt
```

### Step 7: Smoke test the submission

```bash
python src/test_run_local.py \
    --run_py submission/run.py \
    --input_dir data/raw/NM_NGD_coco_dataset/images \
    --output test_predictions.json
```

### Step 8: Build submission zip

```bash
python src/build_submission.py \
    --submission_dir submission \
    --output_zip submission.zip
```

Upload `submission.zip` to the competition website.

---

## Project Structure

```
object_detection_ng/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Original competition data (not in git)
│   │   ├── NM_NGD_coco_dataset/
│   │   └── NM_NGD_product_images/
│   ├── interim/              # Analysis reports
│   └── yolo/                 # YOLO-format dataset
│       ├── dataset.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── src/
│   ├── analyze_dataset.py    # WP1: Dataset analysis
│   ├── prepare_dataset.py    # WP2: COCO → YOLO conversion
│   ├── train_yolov8.py       # WP3: Training
│   ├── validate_competition.py # WP4: Competition-aligned scoring
│   ├── infer_local.py        # WP5: Local inference
│   ├── test_run_local.py     # WP7: Smoke test
│   └── build_submission.py   # WP8: Zip builder
├── submission/
│   ├── run.py                # WP6: Sandbox runner
│   └── best.pt               # Trained weights (after training)
└── experiments/              # Training outputs
```

---

## Sandbox Security Notes

The competition sandbox blocks these imports in submitted .py files:
`os, sys, subprocess, socket, ctypes, builtins, importlib, pickle, marshal, shelve, shutil, yaml, requests, urllib, http.client, multiprocessing, threading, signal, gc`

And these calls: `eval(), exec(), compile(), __import__(), getattr()` with dangerous names.

**Use `pathlib` instead of `os`.** Use `json` instead of `yaml`.

---

## Hyperparameter Quick Reference

| Parameter | Baseline | Alternatives to test |
|-----------|----------|---------------------|
| Model | yolov8m.pt | yolov8l.pt, yolov8x.pt |
| imgsz | 1280 | 960, 1536 |
| conf | 0.10 | 0.05, 0.08, 0.12, 0.15 |
| iou | 0.60 | 0.50, 0.55, 0.65 |
| max_det | 400 | 300, 500 |
| epochs | 80 | 100, 120 |
| close_mosaic | 10 | 15, 20 |

---

## Experiment Log

| ID | Model | imgsz | Det mAP | Cls mAP | Score | Submission | Notes |
|----|-------|-------|---------|---------|-------|------------|-------|
| exp01 | yolov8s | 640 | 0.537* | 0.135* | 0.416* | **0.3889** | Baseline CPU, 18.8s sandbox |
| exp02 | yolov8m | 1280 | — | — | — | — | Colab GPU, pending |
| exp03 | yolov8l | 1280 | — | — | — | — | Colab GPU, pending |

*Local validation on training data
