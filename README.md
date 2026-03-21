# NM i AI 2026 — Competition Workspace

Three exercises for the NM i AI 2026 competition.

## Exercises

| # | Name | Directory | Status |
|---|------|-----------|--------|
| 1 | Object Detection (NorgesGruppen Data) | `object_detection_ng/` | In progress |
| 2 | TBD | — | Pending |
| 3 | TBD | — | Pending |

## Exercise 1: Object Detection

Detect and classify grocery products on store shelves.  
See [`object_detection_ng/README.md`](object_detection_ng/README.md) for full details.

**Quick summary:**
- 248 shelf images, ~22,700 annotations, 357 product categories
- Scoring: `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`
- Sandbox: Python 3.11, NVIDIA L4 GPU, ultralytics 8.1.0, 300s timeout
- Strategy: YOLOv8 fine-tuning → threshold tuning → classification refinement
