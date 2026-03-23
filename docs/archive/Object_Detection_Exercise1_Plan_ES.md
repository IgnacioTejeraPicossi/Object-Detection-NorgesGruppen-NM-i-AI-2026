> **Language:** Spanish (archived original / mixed draft).  
> **Canonical English version:** [`../Object_Detection_Exercise1_Plan.md`](../Object_Detection_Exercise1_Plan.md)

---

Plan de la NM i AI 2026 para el Ejercicio 1: Object Detection.

....
Plan paso a paso, pensado para ejecutarlo con Cursor AI, con enfoque muy práctico para este Ejercicio 1: Object Detection.
Voy a asumir este objetivo:
•	Objetivo competitivo inicial: conseguir una submission sólida y fiable rápido.
•	Estrategia recomendada: empezar con una solución YOLOv8 fine-tuned porque encaja muy bien con el sandbox, las librerías preinstaladas y el límite de 300 segundos.
•	Objetivo técnico realista en fases:
1.	tener una submission válida cuanto antes,
2.	luego mejorar detección,
3.	finalmente mejorar clasificación de producto.
________________________________________
Implementation Plan for Cursor AI
AI_NM_2026 – Exercise 1: Object Detection (NorgesGruppen Data)
1. Goal
Build a competition-ready object detection pipeline that:
•	trains locally on the provided COCO dataset,
•	predicts grocery product bounding boxes and product classes,
•	runs inside the competition sandbox with:
o	Python 3.11
o	NVIDIA L4 GPU
o	no network access
o	300s timeout
•	outputs predictions in the required JSON format:
o	image_id
o	category_id
o	bbox in COCO [x, y, w, h]
o	score
The first target is a robust baseline submission using YOLOv8 with pinned compatible versions.
The second target is a stronger ensemble / refinement pipeline if time allows.
________________________________________
2. Recommended strategy
Phase A — Fastest path to a valid competitive baseline
Use YOLOv8m or YOLOv8l fine-tuned on the competition dataset.
Why this is the best first move:
•	Ultralytics is already available in the sandbox.
•	.pt weights are supported directly.
•	YOLO inference is simple and fast.
•	Detection score is 70% of total score, so strong localization matters most.
•	A good one-model solution can already be competitive.
Phase B — Improve classification
Use the product reference images to improve class discrimination.
Possible methods:
•	better augmentation,
•	class balancing,
•	product-image-assisted fine-tuning,
•	two-stage re-ranking for hard classes.
Phase C — Advanced improvement
Only if baseline is working:
•	test ensemble,
•	test TTA,
•	test weighted boxes fusion,
•	test detector + classifier re-ranking.
________________________________________
3. Technical decision
Final recommended baseline stack
•	Detector: YOLOv8m first, YOLOv8l if training hardware allows
•	Training framework: ultralytics==8.1.0
•	Weights format: .pt
•	Inference in submission: run.py using Ultralytics
•	Input size: start with imgsz=1280, then compare with 960
•	Inference precision: FP16 on GPU
•	Confidence threshold: tune on validation
•	NMS IoU threshold: tune on validation
Why not start with ONNX?
ONNX is useful later, but the simplest reliable path is:
•	fine-tune with Ultralytics
•	submit .pt
•	use same version as sandbox
That reduces compatibility risk.
________________________________________
4. Project structure for Cursor AI
Ask Cursor AI to create this structure:
object_detection_ng/
├── data/
│   ├── raw/
│   │   ├── NM_NGD_coco_dataset/
│   │   └── NM_NGD_product_images/
│   ├── processed/
│   │   ├── images/
│   │   ├── labels/
│   │   └── folds/
│   └── yolo_dataset.yaml
├── src/
│   ├── prepare_dataset.py
│   ├── analyze_dataset.py
│   ├── train_yolov8.py
│   ├── validate_yolov8.py
│   ├── infer_local.py
│   ├── build_submission.py
│   ├── utils_coco.py
│   ├── utils_boxes.py
│   └── utils_product_images.py
├── submission/
│   ├── run.py
│   ├── best.pt
│   └── utils.py
├── experiments/
│   ├── exp01_yolov8m_1280/
│   └── exp02_yolov8l_1280/
├── requirements.txt
└── README.md
________________________________________
5. Work packages
WP1 — Dataset understanding and sanity checks
Goal
Understand the class distribution, box sizes, class imbalance, and data quality.
Tasks for Cursor AI
Create src/analyze_dataset.py that:
•	loads annotations.json,
•	prints:
o	number of images,
o	number of annotations,
o	number of categories,
o	annotations per category,
o	corrected vs non-corrected counts,
o	bbox width/height stats,
o	small / medium / large object distribution,
•	identifies:
o	rare classes,
o	categories with very few examples,
o	whether unknown_product appears and how often.
Why this matters
This dataset has many classes and relatively few images.
The main risk is class confusion and long-tail classes.
Expected output
A CSV and summary report with:
•	category_id
•	product_name
•	annotation_count
________________________________________
WP2 — Convert COCO to YOLO format
Goal
Prepare clean YOLO training data.
Tasks for Cursor AI
Create src/prepare_dataset.py that:
•	reads COCO annotations.json,
•	converts COCO boxes [x, y, w, h] into YOLO normalized format:
o	class x_center y_center width height
•	copies images into train/val folders,
•	creates label .txt files per image,
•	creates data/yolo_dataset.yaml.
Split strategy
Use:
•	Group-aware validation split by image
•	target: 80/20 or 5-fold split
Best practice:
•	start with single 80/20 split
•	later move to 5-fold CV for more reliable threshold tuning
Important
Make sure the category IDs stay exactly aligned with the competition IDs.
________________________________________
WP3 — Strong baseline training
Goal
Train a first real model that can be submitted.
Recommended first experiments
Experiment 1
•	model: yolov8m.pt
•	imgsz: 1280
•	epochs: 80
•	batch: auto / as large as GPU allows
•	optimizer: default
•	augmentations: standard YOLO augmentations
•	patience: 20
Experiment 2
•	model: yolov8l.pt
•	imgsz: 1280
•	epochs: 80
•	compare against m model
Cursor AI task
Create src/train_yolov8.py with configurable CLI:
python src/train_yolov8.py \
  --data data/yolo_dataset.yaml \
  --model yolov8m.pt \
  --imgsz 1280 \
  --epochs 80 \
  --batch 8 \
  --project experiments \
  --name exp01_yolov8m_1280
Recommended training options
•	enable pretrained weights
•	save best model
•	cache images if possible
•	test close_mosaic=10
•	consider lower LR only if unstable
Important note
Because shelves contain many objects and often small/medium products, higher resolution matters.
Do not start at 640 unless hardware is very limited.
________________________________________
WP4 — Validation aligned to competition scoring
Goal
Measure progress in the same logic as the competition:
•	70% detection mAP
•	30% classification mAP
Key insight
Normal YOLO validation gives class-aware metrics, but the competition also has a class-agnostic detection component.
Cursor AI task
Create src/validate_yolov8.py that computes:
1.	Detection-only mAP@0.5
o	ignore categories
o	only IoU >= 0.5 matters
2.	Classification mAP@0.5
o	IoU >= 0.5
o	correct category required
3.	Hybrid score
score = 0.7 * detection_map50 + 0.3 * classification_map50
Why this matters
This lets you optimize for the actual leaderboard, not generic validation only.
________________________________________
WP5 — Build the first submission package
Goal
Produce a sandbox-compatible zip as early as possible.
Cursor AI task
Create submission/run.py that:
•	parses:
o	--input
o	--output
•	loads best.pt
•	uses CUDA automatically if available
•	iterates over images
•	writes a single JSON array to output
Design requirements
•	no os, no subprocess, no socket
•	use pathlib
•	process images one by one
•	use torch.no_grad()
•	preferably use FP16 on GPU
Skeleton behavior
For each image:
•	parse image_id from file name,
•	run inference,
•	extract boxes and classes,
•	convert XYXY to COCO XYWH,
•	append predictions.
________________________________________
6. Recommended run.py behavior
Baseline inference settings
Start with these and tune later:
•	imgsz=1280
•	conf=0.08 to 0.15
•	iou=0.6
•	max_det=400
Because shelf images may contain many products, avoid a low max_det.
Important tradeoff
•	lower confidence threshold = better recall = often better for detection score
•	too low = too many false positives = worse mAP
Tune this locally.
________________________________________
7. Step-by-step execution order
Step 1 — Build a valid end-to-end baseline
Cursor AI should do the following in order:
1.	create dataset analysis script
2.	create COCO → YOLO conversion script
3.	create train/val split
4.	train yolov8m
5.	export/save best.pt
6.	create local inference script
7.	create submission/run.py
8.	create submission zip
9.	test zip locally with a mock input folder
Success criteria
You can generate a valid predictions.json from unseen images.
________________________________________
Step 2 — Improve detector quality
After valid submission exists:
1.	compare yolov8m vs yolov8l
2.	compare imgsz=960 vs 1280
3.	tune:
o	confidence threshold
o	NMS IoU
o	max_det
4.	inspect failure cases:
o	missed products
o	duplicate boxes
o	bad localization
o	visually similar class confusion
Most likely best returns
The biggest early gain will probably come from:
•	better image size,
•	threshold tuning,
•	stronger model size,
•	better validation split.
________________________________________
Step 3 — Improve classification
Because only 30% of score is class-dependent, do this after detection is stable.
Option A — Better detector training only
Use stronger training:
•	longer training
•	better augmentation
•	class balancing
•	copy-paste / mixup carefully
•	oversample rare classes
Option B — Two-stage classifier for hard cases
Pipeline:
1.	detector finds boxes,
2.	crop each detected product,
3.	classifier re-scores top candidate classes.
This is powerful, but adds complexity and inference time.
Recommendation
Only do two-stage classification if:
•	one-stage YOLO baseline is already working well,
•	you still have submission attempts left,
•	you have enough local validation evidence that it improves score.
________________________________________
8. Best use of product reference images
The product reference pack is highly valuable for classification.
Recommended uses
Use 1 — Visual inspection for confusion groups
Group visually similar products:
•	same brand family
•	same packaging color
•	similar size but different flavor/version
This helps identify where the model will confuse classes.
Use 2 — Product-image-assisted classifier
Later, build a classifier trained on:
•	cropped shelf product instances
•	reference product images
Use 3 — Embedding re-ranking
Advanced option:
•	detector predicts top boxes and category logits,
•	crop the box,
•	compare crop embedding to product reference embeddings,
•	re-rank top-k classes.
This is a strong advanced idea, but not phase 1.
________________________________________
9. Data issues to handle carefully
Cursor AI should explicitly check these:
1. Category count mismatch
Your pasted docs contain both:
•	category_id 0-355 and
•	unknown_product as 356
This suggests a possible off-by-one / total count ambiguity in the description.
Action
Do not assume.
Read the actual annotations.json and confirm:
•	exact number of categories
•	exact max category ID
•	whether unknown_product is included
•	whether total classes are 356 or 357
This is critical.
2. Rare classes
Many categories may have very few examples.
Action
Generate a frequency table and consider:
•	oversampling rare classes in training,
•	focal-like handling through data strategy,
•	special review of underrepresented products.
3. Many products per image
Shelves are dense scenes.
Action
Use:
•	high resolution,
•	enough max_det,
•	careful NMS,
•	avoid too aggressive confidence threshold.
________________________________________
10. Suggested experiments table
Round 1 — Baseline
•	E1: YOLOv8m, 1280, default aug
•	E2: YOLOv8l, 1280, default aug
Round 2 — Tuning
•	E3: best model + lower conf threshold
•	E4: best model + different IoU threshold
•	E5: best model + imgsz 960 vs 1280
Round 3 — Robustness
•	E6: fold-based validation
•	E7: TTA on inference
•	E8: weighted boxes fusion from 2 models
Round 4 — Classification boost
•	E9: crop classifier
•	E10: detector + re-ranker with product reference images
________________________________________
11. Recommended local test harness
Cursor AI should create src/infer_local.py and src/build_submission.py.
infer_local.py
Runs model on a folder and writes predictions.json.
build_submission.py
Copies only allowed files into submission/:
•	run.py
•	best.pt
•	utils.py if needed
Then creates zip and verifies:
•	run.py is at zip root
•	total size is within limits
•	file types are allowed
Add a validator
Create a local script that checks:
•	output JSON structure
•	numeric types
•	bbox length = 4
•	no NaNs
•	image_id parsing works
________________________________________
12. Concrete coding tasks for Cursor AI
You can paste this directly into Cursor AI:
Prompt 1 — Dataset analysis
Create a Python script src/analyze_dataset.py that loads the competition COCO annotations.json, prints image count, annotation count, unique category count, category frequency table, bbox statistics, corrected vs non-corrected counts, and writes a CSV with category_id, product_name, and annotation_count. Use only standard Python, pathlib, json, collections, pandas if available, and keep the script easy to run from CLI.
Prompt 2 — COCO to YOLO conversion
Create src/prepare_dataset.py that converts the competition COCO annotations into YOLO detection labels, preserves the exact class IDs from the original dataset, creates train/val splits, copies images into a YOLO-compatible folder structure, and writes data/yolo_dataset.yaml. The script must validate that every annotation box is within image bounds and skip invalid boxes safely.
Prompt 3 — Training script
Create src/train_yolov8.py using ultralytics==8.1.0 with CLI arguments for dataset YAML, model name, image size, epochs, batch size, project folder, and experiment name. The script should train YOLOv8, save the best weights, and print the final metrics clearly.
Prompt 4 — Competition-aligned validation
Create src/validate_yolov8.py that evaluates predictions against ground truth using two metrics: class-agnostic detection mAP@0.5 and class-aware classification mAP@0.5, then computes 0.7 * detection + 0.3 * classification. Use pycocotools if possible, otherwise implement a reliable fallback.
Prompt 5 — Submission runner
Create submission/run.py for the competition sandbox. It must accept --input and --output, load best.pt, auto-detect CUDA, run inference image by image, convert predictions to JSON with image_id, category_id, bbox, and score, and write the final JSON array. Use pathlib, avoid blocked imports such as os, subprocess, and socket, and keep memory usage low.
Prompt 6 — Packaging
Create src/build_submission.py that validates the submission folder, ensures run.py is at the zip root, checks allowed file types, reports total size, and builds a final submission.zip.
________________________________________
13. Best practical submission strategy
Given the limited daily submissions, I recommend this order:
Submission 1
A safe baseline:
•	single YOLOv8m or YOLOv8l model
•	tuned confidence threshold
•	no ensemble
•	maximum reliability
Submission 2
Best improved model:
•	same architecture or stronger one
•	better threshold tuning
•	maybe TTA if still within timeout
Submission 3
Only if local validation clearly supports it:
•	ensemble or two-stage improvement
Do not waste early submissions on complex pipelines before the baseline is stable.
________________________________________
14. Risks and mitigations
Risk 1 — Class indexing mistake
Mitigation: inspect annotations.json first and generate the class map automatically.
Risk 2 — Timeout in sandbox
Mitigation: process images one by one, use GPU, FP16, and avoid overly heavy ensemble first.
Risk 3 — Weight/version incompatibility
Mitigation: pin ultralytics==8.1.0 for training.
Risk 4 — Overfitting due to only 248 images
Mitigation: use validation carefully, inspect failures manually, do not trust one split too much.
Risk 5 — Detection good but classification weak
Mitigation: prioritize detection first, then add classification refinement only if time remains.
________________________________________
15. Recommended first-week implementation plan
Day 1
•	inspect dataset
•	confirm class count
•	convert COCO to YOLO
•	create first train/val split
Day 2
•	train YOLOv8m baseline
•	build local inference script
•	generate first valid predictions
Day 3
•	create submission run.py
•	package zip
•	test end to end locally
•	upload first safe submission
Day 4
•	train YOLOv8l
•	compare thresholds and image sizes
•	analyze failures
Day 5
•	submit improved model
•	decide whether to pursue classifier/re-ranking path
________________________________________

16. My recommendation in one sentence
Start with a clean YOLOv8m/l fine-tuning pipeline, get a valid submission immediately, optimize detection first, and only then invest in product-reference-image-based classification improvements.
Si quieres, en el siguiente mensaje te hago la versión 2 todavía más operativa: un “Cursor AI execution pack” con:

•	prompts exactos para Cursor,
•	estructura de archivos completa,
•	y un primer run.py base listo para adaptar.

Ignacio: Si haz, en el siguiente mensaje la versión 2 todavía más operativa: un “Cursor AI execution pack” con:
prompts exactos para Cursor,
estructura de archivos completa,
y un primer run.py base listo para adaptar. Gracias
Perfecto. Aquí tienes la versión 2, mucho más operativa, pensada para que la ejecutes con Cursor AI de forma casi directa.
Voy a darte 3 cosas:
1.	Execution pack completo
2.	Prompts exactos para Cursor AI
3.	Un primer run.py base listo para adaptar
Todo está orientado a llegar rápido a una submission válida, y luego mejorarla sin romper compatibilidad con el sandbox.
________________________________________
Cursor AI Execution Pack
AI_NM_2026 – Exercise 1: Object Detection (NorgesGruppen Data)
________________________________________
1. Estrategia de implementación recomendada
Para este ejercicio, la ruta más segura es:
Fase 1
Construir una solución simple, robusta y submitible:
•	conversión COCO → YOLO
•	entrenamiento YOLOv8 fine-tuned
•	run.py compatible con sandbox
•	primer zip válido
Fase 2
Mejorar score:
•	tuning de thresholds
•	comparar yolov8m vs yolov8l
•	input resolution
•	TTA si cabe en tiempo
Fase 3
Mejorar clasificación:
•	usar product reference images
•	classifier / re-ranking
•	ensemble si compensa
La clave aquí es no complicarse demasiado antes de tener una submission estable.
________________________________________
2. Estructura de archivos completa
Pídele a Cursor AI que cree exactamente esta estructura:
object_detection_ng/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── NM_NGD_coco_dataset/
│   │   │   ├── images/
│   │   │   └── annotations.json
│   │   └── NM_NGD_product_images/
│   │       ├── metadata.json
│   │       └── <product_code>/
│   │           ├── main.jpg
│   │           ├── front.jpg
│   │           ├── back.jpg
│   │           └── ...
│   │
│   ├── interim/
│   │   ├── category_stats.csv
│   │   ├── folds.json
│   │   └── dataset_summary.json
│   │
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── dataset.yaml
│
├── src/
│   ├── analyze_dataset.py
│   ├── prepare_dataset.py
│   ├── train_yolov8.py
│   ├── validate_competition.py
│   ├── infer_local.py
│   ├── build_submission.py
│   ├── test_run_local.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── coco_utils.py
│   │   ├── box_utils.py
│   │   ├── io_utils.py
│   │   ├── scoring_utils.py
│   │   └── image_utils.py
│
├── experiments/
│   ├── exp01_yolov8m_1280/
│   ├── exp02_yolov8l_1280/
│   └── notes.md
│
├── submission/
│   ├── run.py
│   ├── best.pt
│   ├── utils.py
│   └── submission.zip
│
└── notebooks/
    └── quick_error_analysis.ipynb
________________________________________
3. Orden exacto de trabajo
Este es el orden que debes seguir con Cursor:
Paso 1
analyze_dataset.py
Paso 2
prepare_dataset.py
Paso 3
train_yolov8.py
Paso 4
infer_local.py
Paso 5
validate_competition.py
Paso 6
submission/run.py
Paso 7
test_run_local.py
Paso 8
build_submission.py
Ese orden minimiza errores y evita perder tiempo.
________________________________________
4. Prompt maestro para Cursor AI
Pega esto primero en Cursor para establecer el contexto global:
We are implementing a competition solution for AI_NM_2026 Exercise 1: "Object Detection" from NorgesGruppen Data.

Goal:
Build a robust grocery shelf object detection pipeline that trains locally and submits as a .zip containing run.py at the zip root.

Competition constraints:
- Input: shelf images
- Output: JSON list with fields: image_id, category_id, bbox [x, y, w, h], score
- Scoring: 0.7 * detection_mAP@0.5 (class-agnostic) + 0.3 * classification_mAP@0.5 (class-aware)
- Sandbox: Python 3.11, GPU NVIDIA L4 24GB, CUDA 12.4, no network, timeout 300s
- Preinstalled: torch 2.6.0+cu124, torchvision 0.21.0, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless, albumentations, Pillow, numpy, scipy, scikit-learn, pycocotools, ensemble-boxes, timm
- No pip install at runtime
- Use pathlib instead of os
- Avoid blocked imports: os, subprocess, socket, ctypes, builtins
- run.py must be at zip root
- Allowed weights: .pt, .pth, .onnx, .safetensors, .npy
- Weight total max size: 420 MB

Implementation strategy:
1. Start with YOLOv8 fine-tuning using ultralytics==8.1.0
2. Build a reliable first submission
3. Optimize thresholds and image size
4. Only later consider classification refinement or ensemble

Important:
- Preserve exact competition category IDs from annotations.json
- Validate whether the real number of categories is 356 or 357 by reading the file, do not assume
- Dense shelf scenes require high max_det and careful confidence thresholding
- Prioritize robust code, reproducibility, and simple CLI usage

Coding style:
- Production-style Python
- Clear CLI interfaces with argparse
- Strong validation and helpful logging
- Modular helper functions
- Keep files focused and maintainable
________________________________________
5. Prompt exacto para crear analyze_dataset.py
Create a production-quality Python script at src/analyze_dataset.py.

Purpose:
Analyze the competition COCO annotations file and generate a dataset summary.

Requirements:
- Use pathlib, json, collections, statistics, csv, and pandas if available
- CLI arguments:
  --annotations path to annotations.json
  --output_dir directory for reports
- Read the COCO annotations file and compute:
  1. number of images
  2. number of annotations
  3. number of categories
  4. minimum and maximum category_id
  5. whether unknown_product exists
  6. annotation count per category
  7. corrected=true vs corrected=false counts
  8. bbox width/height/area statistics
  9. top 20 most frequent categories
  10. bottom 20 least frequent categories
- Write:
  - category_stats.csv with category_id, category_name, annotation_count
  - dataset_summary.json with global metrics
- Print a concise console summary
- Add validation checks for missing category IDs, invalid boxes, and image references that do not exist in the images list
- Keep the code clean and easy to run

The script should not depend on notebook-only features.
________________________________________
6. Prompt exacto para crear prepare_dataset.py
Create a production-quality Python script at src/prepare_dataset.py.

Purpose:
Convert the competition COCO dataset into YOLO format and create a train/validation split.

Requirements:
- Use pathlib, json, shutil, random, and typing
- CLI arguments:
  --annotations path to annotations.json
  --images_dir path to original images folder
  --output_dir path to YOLO output root
  --val_ratio float default 0.2
  --seed int default 42
- Read images, categories, and annotations from COCO
- Preserve the exact original category IDs from the competition
- Convert COCO bbox [x, y, w, h] into YOLO normalized format
- Validate every annotation:
  - width > 0 and height > 0
  - bbox stays within image dimensions after clamping
  - skip invalid boxes safely and report counts
- Split by image into train/val
- Copy images into:
  images/train
  images/val
- Create label files in:
  labels/train
  labels/val
- Write dataset.yaml for ultralytics with:
  path
  train
  val
  names
- Important:
  Because competition category IDs may not be contiguous in the file, verify and handle this carefully. If IDs are contiguous, map directly. If not, create a deterministic mapping and save a JSON mapping file.
- Save:
  - class_id_mapping.json
  - split_summary.json
- Log skipped annotations and summary counts

Also include helper functions so the logic is easy to test.
________________________________________
7. Prompt exacto para crear train_yolov8.py
Create a clean training script at src/train_yolov8.py using ultralytics YOLO.

Requirements:
- CLI arguments:
  --data path to dataset.yaml
  --model model checkpoint path or model name, default yolov8m.pt
  --imgsz int default 1280
  --epochs int default 80
  --batch int default 8
  --device string default auto
  --project string default experiments
  --name string default exp
  --patience int default 20
  --workers int default 4
  --close_mosaic int default 10
  --seed int default 42
- Use ultralytics==8.1.0 compatible API
- Train a detection model and save best weights
- Print the location of best.pt clearly at the end
- Add defensive checks that dataset.yaml exists
- Prefer reproducible settings where possible
- Keep the code simple and focused

Also add comments indicating which arguments are most important to tune for this competition.
________________________________________
8. Prompt exacto para crear infer_local.py
Create a local inference script at src/infer_local.py.

Purpose:
Run the trained YOLO model on a folder of images and produce a predictions.json in competition format.

Requirements:
- CLI arguments:
  --model path to best.pt
  --input_dir directory with images
  --output path to predictions.json
  --imgsz int default 1280
  --conf float default 0.10
  --iou float default 0.60
  --max_det int default 400
  --device string default auto
- Use pathlib, json, torch, and ultralytics
- Auto-detect CUDA when available
- Process images one by one
- Parse image_id from file names like img_00042.jpg -> 42
- Convert boxes from xyxy to COCO xywh
- Round bbox values to 1 decimal and scores to 3 decimals
- Output JSON list of dicts with:
  image_id, category_id, bbox, score
- Skip non-image files safely
- Add clean logging

Keep the script very close to what we will later use in submission/run.py.
________________________________________
9. Prompt exacto para crear validate_competition.py
Create a Python script at src/validate_competition.py.

Purpose:
Evaluate predictions using the competition scoring logic:
score = 0.7 * detection_mAP50 + 0.3 * classification_mAP50

Requirements:
- CLI arguments:
  --annotations path to ground truth annotations.json
  --predictions path to predictions.json
- Implement:
  1. class-agnostic detection evaluation at IoU 0.5
  2. class-aware classification evaluation at IoU 0.5
  3. final hybrid score
- Use pycocotools if practical, but do not make the code fragile
- If pycocotools is hard to adapt for class-agnostic evaluation, implement a clean custom evaluator
- The script should print:
  - detection_map50
  - classification_map50
  - final_score
- Add validation for malformed predictions
- Predictions use competition format with image_id, category_id, bbox, score

Keep the code readable and trustworthy rather than overly clever.
________________________________________
10. Prompt exacto para crear submission/run.py
Este es probablemente el prompt más importante.
Create a sandbox-safe competition inference script at submission/run.py.

Purpose:
This file will be placed at the root of the submission zip and executed as:
python run.py --input /data/images --output /output/predictions.json

Hard constraints:
- Use pathlib instead of os
- Do not import os, subprocess, socket, ctypes, or builtins
- Use only packages expected in the competition sandbox
- Load a YOLOv8 model from local best.pt in the same folder
- Auto-detect CUDA using torch.cuda.is_available()
- Process images one by one to keep memory usage low
- Write a single JSON array to the output path
- Skip non-image files safely
- Parse image_id from filenames like img_00042.jpg -> 42
- Output records must be:
  {
    "image_id": int,
    "category_id": int,
    "bbox": [x, y, w, h],
    "score": float
  }

CLI:
- --input required
- --output required
- optional:
  --imgsz default 1280
  --conf default 0.10
  --iou default 0.60
  --max_det default 400

Implementation details:
- Use ultralytics YOLO
- Use torch.no_grad()
- Use half precision on CUDA if practical
- Convert xyxy to xywh
- Round bbox to 1 decimal and score to 3 decimals
- Ensure output directory exists
- Keep code robust and minimal
- Include helper functions inside the same file or a tiny local utils.py if needed

Important:
Assume the competition category IDs are already aligned in the trained model. Do not remap category IDs unless a mapping file is explicitly loaded.
________________________________________
11. Prompt exacto para crear test_run_local.py
Create a local smoke test script at src/test_run_local.py.

Purpose:
Test that submission/run.py works locally in the same interface expected by the competition.

Requirements:
- CLI arguments:
  --run_py path to submission/run.py
  --input_dir directory with test images
  --output path to test predictions.json
- Execute run.py through Python in a simple way
- Validate:
  - output file exists
  - output JSON is a list
  - each prediction has image_id, category_id, bbox, score
  - bbox has length 4
  - score is numeric
- Print a short validation summary

Keep it practical and simple.
________________________________________
12. Prompt exacto para crear build_submission.py
Create a packaging script at src/build_submission.py.

Purpose:
Validate and build the final competition zip.

Requirements:
- Use pathlib, zipfile, json, and shutil
- CLI arguments:
  --submission_dir path to submission folder
  --output_zip path to final zip
- Validate:
  - run.py exists at submission root
  - only allowed file types are included
  - total uncompressed size is within 420 MB
  - no nested top-level submission folder is created inside the zip
- Allowed extensions:
  .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy
- Print a file manifest with sizes
- Build the zip so that run.py is at the root of the archive
- Fail loudly with clear messages on invalid files

Keep the implementation concise and reliable.
________________________________________
13. Primer run.py base listo para adaptar
Este ya te sirve como base real. Puedes pegarlo en Cursor y pedir que lo refine, o adaptarlo tú directamente.
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to folder with input images")
    parser.add_argument("--output", required=True, help="Path to output predictions.json")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--max_det", type=int, default=400, help="Maximum detections per image")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def extract_image_id(path: Path) -> int:
    # Expected format: img_00042.jpg -> 42
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse image_id from filename: {path.name}") from exc


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float):
    return [x1, y1, x2 - x1, y2 - y1]


def round_bbox(bbox):
    return [round(float(v), 1) for v in bbox]


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(__file__).resolve().parent / "best.pt"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(model_path))

    image_paths = sorted([p for p in input_dir.iterdir() if is_image_file(p)])
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            image_id = extract_image_id(image_path)

            results = model(
                str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                verbose=False,
            )

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy
                cls = result.boxes.cls
                conf = result.boxes.conf

                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    bbox = xyxy_to_xywh(x1, y1, x2, y2)

                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(cls[i].item()),
                            "bbox": round_bbox(bbox),
                            "score": round(float(conf[i].item()), 3),
                        }
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
________________________________________
14. Versión mejorada de run.py con pequeños extras útiles
Esta versión añade algo de robustez sin complicarlo demasiado.
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing input images")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--max_det", type=int, default=400, help="Maximum detections per image")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def extract_image_id(path: Path) -> int:
    parts = path.stem.split("_")
    if not parts:
        raise ValueError(f"Invalid image filename: {path.name}")
    try:
        return int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Failed to parse image_id from filename: {path.name}") from exc


def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def clamp_non_negative_bbox(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    x = max(0.0, x)
    y = max(0.0, y)
    w = max(0.0, w)
    h = max(0.0, h)
    return [x, y, w, h]


def format_prediction(image_id: int, category_id: int, bbox: List[float], score: float) -> Dict[str, Any]:
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [round(float(v), 1) for v in bbox],
        "score": round(float(score), 3),
    }


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def run_inference(
    model: YOLO,
    image_path: Path,
    image_id: int,
    device: str,
    imgsz: int,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    results = model(
        str(image_path),
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        device=device,
        verbose=False,
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy
        cls = result.boxes.cls
        confs = result.boxes.conf

        for idx in range(len(result.boxes)):
            box_xyxy = xyxy[idx].tolist()
            bbox = xyxy_to_xywh(box_xyxy)
            bbox = clamp_non_negative_bbox(bbox)

            outputs.append(
                format_prediction(
                    image_id=image_id,
                    category_id=int(cls[idx].item()),
                    bbox=bbox,
                    score=float(confs[idx].item()),
                )
            )

    return outputs


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(__file__).resolve().parent / "best.pt"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(path for path in input_dir.iterdir() if is_image_file(path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path)

    all_predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for image_path in image_paths:
            image_id = extract_image_id(image_path)
            preds = run_inference(
                model=model,
                image_path=image_path,
                image_id=image_id,
                device=device,
                imgsz=args.imgsz,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                max_det=args.max_det,
            )
            all_predictions.extend(preds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_predictions, f)

    print(f"Processed {len(image_paths)} images")
    print(f"Generated {len(all_predictions)} predictions")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
________________________________________
15. Recomendación de hiperparámetros iniciales
Empieza así:
Entrenamiento baseline
•	modelo: yolov8m.pt
•	imgsz=1280
•	epochs=80
•	batch=8 o auto según GPU
•	patience=20
•	close_mosaic=10
Inference baseline
•	imgsz=1280
•	conf=0.10
•	iou=0.60
•	max_det=400
Luego prueba:
•	conf=0.08
•	conf=0.12
•	iou=0.55
•	iou=0.65
________________________________________
16. Qué pedirle a Cursor después del primer baseline
Cuando ya tengas entrenamiento + inferencia + zip funcionando, usa este prompt:
Now review the baseline pipeline and suggest the top 5 highest-impact improvements for this specific competition, prioritizing:
1. better detection recall on dense shelf images
2. stronger class discrimination across visually similar products
3. competition-score alignment
4. runtime safety under a 300-second sandbox limit
5. low implementation risk

For each improvement, explain:
- expected benefit
- complexity
- risk
- whether it should be tested before or after first submission
________________________________________
17. Qué haría yo exactamente en tus primeras 3 iteraciones
Iteración 1
Objetivo: zip válido
•	analizar dataset
•	convertir a YOLO
•	entrenar yolov8m
•	crear run.py
•	probar local
•	generar zip
Iteración 2
Objetivo: mejorar score sin añadir complejidad
•	comparar yolov8m vs yolov8l
•	ajustar conf, iou, max_det
•	revisar errores visuales
Iteración 3
Objetivo: mejora competitiva real
•	TTA o ensemble ligero
•	quizá clasificación auxiliar si validación lo justifica
________________________________________
18. El detalle más importante de todos
Antes de entrenar, Cursor debe confirmar leyendo el annotations.json real:
•	número exacto de categorías,
•	rango real de category_id,
•	si existe o no unknown_product,
•	si hay 356 o 357 clases efectivas.
Eso no lo debes asumir desde la documentación.
________________________________________
19. Mi recomendación operativa final
Tu mejor camino ahora mismo es este:
•	montar pipeline limpio,
•	sacar una submission funcional muy rápido,
•	validar localmente con scoring parecido al oficial,
•	y solo después ir a mejoras más sofisticadas.
Eso te da velocidad, control y menos riesgo de romper el submit.
En el siguiente mensaje te puedo preparar el pack 3, todavía más útil: un conjunto de archivos iniciales completos para copiar/pegar, incluyendo:
•	train_yolov8.py
•	infer_local.py
•	build_submission.py
•	test_run_local.py
listos para que los pegues en Cursor.
Ignacio Tejera
Marzo 2026


