"""Evaluate predictions using the competition scoring logic.

Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

- Detection mAP: class-agnostic, IoU >= 0.5 (did you find the products?)
- Classification mAP: class-aware, IoU >= 0.5 AND correct category_id

Usage:
    python src/validate_competition.py \
        --annotations data/raw/NM_NGD_coco_dataset/annotations.json \
        --predictions predictions.json
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DETECTION_WEIGHT = 0.7
CLASSIFICATION_WEIGHT = 0.3


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_predictions(predictions: list[dict]) -> list[str]:
    """Check predictions for common issues. Returns list of warnings."""
    warnings = []
    if not isinstance(predictions, list):
        warnings.append("Predictions is not a list")
        return warnings
    if len(predictions) == 0:
        warnings.append("Predictions list is empty")
        return warnings

    required_keys = {"image_id", "category_id", "bbox", "score"}
    for i, pred in enumerate(predictions):
        missing = required_keys - set(pred.keys())
        if missing:
            warnings.append(f"Prediction {i}: missing keys {missing}")
            continue
        if not isinstance(pred["bbox"], list) or len(pred["bbox"]) != 4:
            warnings.append(f"Prediction {i}: bbox must be [x, y, w, h]")
        if not isinstance(pred["score"], (int, float)):
            warnings.append(f"Prediction {i}: score must be numeric")
        if any(np.isnan(v) or np.isinf(v) for v in pred["bbox"]):
            warnings.append(f"Prediction {i}: bbox contains NaN or Inf")

    return warnings


def compute_map50(coco_gt: COCO, coco_dt, cat_ids: list[int] | None = None) -> float:
    """Compute mAP@0.5 using pycocotools."""
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])
    if cat_ids is not None:
        coco_eval.params.catIds = cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # precision shape: [T, R, K, A, M] — we want mAP across all
    # T=iou thresholds, R=recall thresholds, K=categories, A=area ranges, M=max dets
    precision = coco_eval.eval["precision"]
    if precision.size == 0:
        return 0.0

    # precision[0, :, :, 0, 2] = IoU=0.5, all recall, all cats, area=all, maxDet=100
    ap_per_cat = []
    for k_idx in range(precision.shape[2]):
        p = precision[0, :, k_idx, 0, 2]
        valid = p[p > -1]
        ap_per_cat.append(np.mean(valid) if len(valid) > 0 else 0.0)

    return float(np.mean(ap_per_cat)) if ap_per_cat else 0.0


def compute_detection_map50(gt_data: dict, predictions: list[dict]) -> float:
    """Class-agnostic detection mAP@0.5: all predictions and GT get category_id=1."""
    DUMMY_CAT = 1

    gt_agnostic = copy.deepcopy(gt_data)
    gt_agnostic["categories"] = [{"id": DUMMY_CAT, "name": "object", "supercategory": "object"}]
    for ann in gt_agnostic["annotations"]:
        ann["category_id"] = DUMMY_CAT

    pred_agnostic = []
    for p in predictions:
        pred_agnostic.append({
            "image_id": p["image_id"],
            "category_id": DUMMY_CAT,
            "bbox": p["bbox"],
            "score": p["score"],
        })

    coco_gt = COCO()
    coco_gt.dataset = gt_agnostic
    coco_gt.createIndex()

    if not pred_agnostic:
        return 0.0

    coco_dt = coco_gt.loadRes(pred_agnostic)
    return compute_map50(coco_gt, coco_dt, cat_ids=[DUMMY_CAT])


def compute_classification_map50(gt_data: dict, predictions: list[dict]) -> float:
    """Class-aware classification mAP@0.5: standard COCO eval with original categories."""
    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(gt_data)
    coco_gt.createIndex()

    if not predictions:
        return 0.0

    pred_formatted = []
    for p in predictions:
        pred_formatted.append({
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"],
            "score": p["score"],
        })

    coco_dt = coco_gt.loadRes(pred_formatted)
    cat_ids = sorted(coco_gt.getCatIds())
    return compute_map50(coco_gt, coco_dt, cat_ids=cat_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Competition-aligned prediction evaluation")
    parser.add_argument("--annotations", type=str, required=True, help="Path to GT annotations.json")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.json")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    pred_path = Path(args.predictions)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    gt_data = load_json(ann_path)
    predictions = load_json(pred_path)

    # Validate
    warnings = validate_predictions(predictions)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
            return

    gt_image_ids = {img["id"] for img in gt_data["images"]}
    pred_image_ids = {p["image_id"] for p in predictions}
    covered = pred_image_ids & gt_image_ids
    extra = pred_image_ids - gt_image_ids
    missed = gt_image_ids - pred_image_ids

    print(f"\nPredictions: {len(predictions)} detections across {len(pred_image_ids)} images")
    print(f"Ground truth: {len(gt_data['annotations'])} annotations across {len(gt_image_ids)} images")
    print(f"Images covered: {len(covered)}, missed: {len(missed)}, extra (ignored): {len(extra)}")

    print("\n--- Detection mAP@0.5 (class-agnostic) ---")
    det_map = compute_detection_map50(gt_data, predictions)

    print("\n--- Classification mAP@0.5 (class-aware) ---")
    cls_map = compute_classification_map50(gt_data, predictions)

    final_score = DETECTION_WEIGHT * det_map + CLASSIFICATION_WEIGHT * cls_map

    print("\n" + "=" * 50)
    print("COMPETITION SCORE")
    print("=" * 50)
    print(f"  Detection mAP@0.5:        {det_map:.4f}")
    print(f"  Classification mAP@0.5:   {cls_map:.4f}")
    print(f"  Final score (0.7D+0.3C):  {final_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
