"""Run trained YOLO model locally and produce predictions in competition format.

Usage:
    python src/infer_local.py \
        --model experiments/exp01_yolov8m_1280/weights/best.pt \
        --input_dir data/raw/NM_NGD_coco_dataset/images \
        --output predictions.json \
        --imgsz 1280 \
        --conf 0.10 \
        --iou 0.60 \
        --max_det 400

If category IDs were remapped during training, provide the mapping file:
    --class_mapping data/yolo/class_id_mapping.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local inference with trained YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output predictions JSON")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--max_det", type=int, default=400, help="Max detections per image")
    parser.add_argument("--device", type=str, default="", help="Device ('' for auto, 'cpu', '0')")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--class_mapping", type=str, default=None, help="Path to class_id_mapping.json for ID remapping")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def extract_image_id(path: Path) -> int:
    """Parse image_id from filename like img_00042.jpg -> 42."""
    parts = path.stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot parse image_id from '{path.name}'") from exc


def load_class_mapping(mapping_path: Path) -> dict[int, int] | None:
    """Load YOLO index -> COCO category_id mapping if IDs were remapped."""
    if not mapping_path.exists():
        logger.warning("Mapping file not found: %s", mapping_path)
        return None

    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("contiguous", True):
        return None

    return {int(k): int(v) for k, v in data["yolo_idx_to_coco_id"].items()}


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def run_inference(
    model: YOLO,
    image_paths: list[Path],
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    half: bool,
    yolo_to_coco: dict[int, int] | None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            image_id = extract_image_id(image_path)
            logger.info("Processing %d/%d: %s (image_id=%d)", idx + 1, len(image_paths), image_path.name, image_id)

            results = model(
                str(image_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                half=half,
                verbose=False,
            )

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy
                cls = result.boxes.cls
                confs = result.boxes.conf

                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    bbox = xyxy_to_xywh(x1, y1, x2, y2)
                    bbox = [max(0.0, v) for v in bbox]

                    yolo_cls = int(cls[i].item())
                    category_id = yolo_to_coco[yolo_cls] if yolo_to_coco else yolo_cls

                    predictions.append({
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [round(float(v), 1) for v in bbox],
                        "score": round(float(confs[i].item()), 3),
                    })

            logger.info("  -> %d detections", sum(1 for p in predictions if p["image_id"] == image_id))

    return predictions


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    yolo_to_coco = None
    if args.class_mapping:
        yolo_to_coco = load_class_mapping(Path(args.class_mapping))
        if yolo_to_coco:
            logger.info("Loaded class ID remapping with %d entries", len(yolo_to_coco))
        else:
            logger.info("Category IDs are contiguous, no remapping needed")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = YOLO(str(model_path))
    image_paths = sorted(p for p in input_dir.iterdir() if is_image_file(p))
    logger.info("Found %d images in %s", len(image_paths), input_dir)

    predictions = run_inference(
        model, image_paths, device,
        args.imgsz, args.conf, args.iou, args.max_det, args.half,
        yolo_to_coco,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    logger.info("Wrote %d predictions to %s", len(predictions), output_path)


if __name__ == "__main__":
    main()
