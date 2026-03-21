"""Train YOLOv8 on the competition dataset.

Usage:
    python src/train_yolov8.py \
        --data data/yolo/dataset.yaml \
        --model yolov8m.pt \
        --imgsz 1280 \
        --epochs 80 \
        --batch 8 \
        --project experiments \
        --name exp01_yolov8m_1280

Key tuning parameters for this competition:
    --imgsz     Shelf images are dense; 1280 is strongly recommended over 640.
    --model     Start with yolov8m.pt, upgrade to yolov8l.pt if GPU allows.
    --epochs    80 is a good starting point with patience=20.
    --close_mosaic  Disabling mosaic for last N epochs helps fine-grained localization.
"""

import argparse
from pathlib import Path

import torch

# ultralytics 8.1.0 + torch >=2.6 need weights_only=False for .pt loading
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for grocery shelf detection")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model name or checkpoint path")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size (1280 recommended for shelves)")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (use -1 for auto)")
    parser.add_argument("--device", type=str, default="", help="Device: '' for auto, '0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--project", type=str, default="experiments", help="Project directory for results")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic for last N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    train_args = {
        "data": str(data_path.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "save": True,
        "workers": args.workers,
        "seed": args.seed,
        "close_mosaic": args.close_mosaic,
        "verbose": True,
        "val": True,
    }

    if args.device:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True
    if args.cache:
        train_args["cache"] = True

    print(f"\nTraining config:")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")
    print()

    results = model.train(**train_args)

    # Print results and best weights location
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_path = Path(args.project) / args.name / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if best_path.exists():
        size_mb = best_path.stat().st_size / (1024 * 1024)
        print(f"  Best weights: {best_path}  ({size_mb:.1f} MB)")
    if last_path.exists():
        size_mb = last_path.stat().st_size / (1024 * 1024)
        print(f"  Last weights: {last_path}  ({size_mb:.1f} MB)")
    print(f"\n  Copy best.pt to submission/:")
    print(f"    cp {best_path} submission/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
