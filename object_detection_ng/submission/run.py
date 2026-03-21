"""Competition submission inference script.

Executed in the sandbox as:
    python run.py --input /data/images --output /output/predictions.json

SANDBOX CONSTRAINTS:
    - Python 3.11, NVIDIA L4 GPU (24GB), CUDA 12.4
    - No network access, 300s timeout, 8GB RAM
    - ultralytics 8.1.0, torch 2.6.0+cu124 pre-installed
    - Blocked imports: os, sys, subprocess, socket, ctypes, builtins,
      importlib, pickle, marshal, shelve, shutil, yaml, requests,
      urllib, http.client, multiprocessing, threading, signal, gc
    - Use pathlib instead of os
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

# ultralytics 8.1.0 + torch >=2.6: weights_only default changed to True
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMGSZ = 1280
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.60
MAX_DET = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing input images")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--max_det", type=int, default=MAX_DET)
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def extract_image_id(path: Path) -> int:
    parts = path.stem.split("_")
    if not parts:
        raise ValueError(f"Invalid filename: {path.name}")
    return int(parts[-1])


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [max(0.0, x1), max(0.0, y1), max(0.0, x2 - x1), max(0.0, y2 - y1)]


def format_pred(image_id: int, cat_id: int, bbox: List[float], score: float) -> Dict[str, Any]:
    return {
        "image_id": int(image_id),
        "category_id": int(cat_id),
        "bbox": [round(float(v), 1) for v in bbox],
        "score": round(float(score), 3),
    }


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(__file__).resolve().parent / "best.pt"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = YOLO(str(model_path))

    image_paths = sorted(p for p in input_dir.iterdir() if is_image(p))
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for img_path in image_paths:
            try:
                image_id = extract_image_id(img_path)
            except (ValueError, IndexError):
                continue

            results = model(
                str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                half=use_cuda,
                verbose=False,
            )

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes_xyxy = result.boxes.xyxy
                classes = result.boxes.cls
                scores = result.boxes.conf

                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                    bbox = xyxy_to_xywh(x1, y1, x2, y2)
                    predictions.append(format_pred(
                        image_id=image_id,
                        cat_id=int(classes[i].item()),
                        bbox=bbox,
                        score=float(scores[i].item()),
                    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"Done: {len(predictions)} predictions from {len(image_paths)} images -> {output_path}")


if __name__ == "__main__":
    main()
