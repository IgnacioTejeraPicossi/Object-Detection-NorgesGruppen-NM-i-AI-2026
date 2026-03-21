# NM i AI 2026 — Competition workspace

Working repository for the three **NM i AI 2026** tracks (19–22 March 2026).

---

## Tracks

| # | Name | Folder | Status |
|---|------|--------|--------|
| 1 | **Object Detection** (NorgesGruppen Data) | [`object_detection_ng/`](object_detection_ng/) | **Implemented, tested, and submitted to the platform** |
| 2 | Tripletex AI Accounting Agent | *(separate project)* | *(implemented elsewhere)* |
| 3 | Astar Island | — | Not started |

---

## Exercise 1: what it is and what the solution does

**Goal:** detect and classify grocery products in shelf photos (Norwegian retail data).

**Model output:** JSON predictions in competition format: `image_id`, `category_id`, COCO `bbox` [x, y, w, h], `score`.

**Official score:**  
`0.7 × detection mAP @ IoU 0.5 (class-agnostic) + 0.3 × classification mAP @ IoU 0.5 (correct class)`.

### Status: complete and tested (baseline)

| Check | Result |
|-------|--------|
| Local pipeline | COCO analysis → YOLO conversion → training → inference → hybrid metric validation |
| `submission/run.py` | Meets sandbox rules (`pathlib`, no blocked imports) |
| Local smoke test (`test_run_local.py`) | OK: JSON format, 248 images, runtime &lt; 300 s on reference CPU |
| **Platform submission** | **Done** — evaluation OK in GPU sandbox (NVIDIA L4) |
| **Public leaderboard score (baseline)** | **0.3889** (first submission: YOLOv8s trained on CPU, imgsz 640) |
| Sandbox runtime | ~19 s (well under the 300 s limit) |

The code lives under **`object_detection_ng/`**: scripts in `src/`, weights and `run.py` in `submission/`, and **`submission.zip`** built with `build_submission.py` for upload at  
[Submit — NorgesGruppen Data](https://app.ainm.no/submit/norgesgruppen-data).

### Recommended next improvement

Train **YOLOv8m at 1280 px** on a **Google Colab GPU** using `object_detection_ng/colab_train_yolov8m.ipynb`, replace `submission/best.pt`, and rebuild the zip. That usually gives the largest gain over a CPU 640 baseline.

---

## Detailed documentation (Exercise 1)

- Full technical guide (pipeline, commands, layout): [`object_detection_ng/README.md`](object_detection_ng/README.md)
- Official competition docs: [`docs/`](docs/) (Overview, Submission, Scoring, Examples)

---

## Quick repo layout

```
AI NM i AI 2026/
├── README.md                 ← this file
├── docs/                     ← competition PDFs / Markdown
└── object_detection_ng/      ← Exercise 1 (code + local data)
```
