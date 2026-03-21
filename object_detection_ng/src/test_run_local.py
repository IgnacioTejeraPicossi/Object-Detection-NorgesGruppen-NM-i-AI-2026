"""Smoke test for submission/run.py — validates it works locally.

Usage:
    python src/test_run_local.py \
        --run_py submission/run.py \
        --input_dir data/raw/NM_NGD_coco_dataset/images \
        --output test_predictions.json

This executes run.py as the sandbox would and validates the output.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def validate_predictions(predictions: list) -> list[str]:
    """Validate prediction format. Returns list of errors."""
    errors = []
    if not isinstance(predictions, list):
        return ["Output is not a JSON array"]

    if len(predictions) == 0:
        errors.append("WARNING: Predictions list is empty")
        return errors

    required_keys = {"image_id", "category_id", "bbox", "score"}

    for i, pred in enumerate(predictions[:100]):
        if not isinstance(pred, dict):
            errors.append(f"[{i}] Not a dict")
            continue

        missing = required_keys - set(pred.keys())
        if missing:
            errors.append(f"[{i}] Missing keys: {missing}")
            continue

        if not isinstance(pred["image_id"], int):
            errors.append(f"[{i}] image_id is not int: {type(pred['image_id'])}")

        if not isinstance(pred["category_id"], int):
            errors.append(f"[{i}] category_id is not int: {type(pred['category_id'])}")

        if not isinstance(pred["bbox"], list) or len(pred["bbox"]) != 4:
            errors.append(f"[{i}] bbox is not [x,y,w,h]: {pred['bbox']}")
        else:
            for j, v in enumerate(pred["bbox"]):
                if not isinstance(v, (int, float)):
                    errors.append(f"[{i}] bbox[{j}] is not numeric: {v}")

        if not isinstance(pred["score"], (int, float)):
            errors.append(f"[{i}] score is not numeric: {pred['score']}")
        elif not (0 <= pred["score"] <= 1.001):
            errors.append(f"[{i}] score out of range [0,1]: {pred['score']}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for submission run.py")
    parser.add_argument("--run_py", type=str, default="submission/run.py", help="Path to run.py")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--output", type=str, default="test_predictions.json", help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    run_py = Path(args.run_py)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found: {run_py}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_count = len([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Testing {run_py} with {image_count} images from {input_dir}")

    cmd = [
        sys.executable, str(run_py),
        "--input", str(input_dir),
        "--output", str(output_path),
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {args.timeout}s")
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.time() - start

    print(f"Exit code: {result.returncode}")
    print(f"Time: {elapsed:.1f}s")

    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout[-2000:]}")
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr[-2000:]}")

    if result.returncode != 0:
        print("\nFAILED: run.py exited with non-zero code")
        return

    if not output_path.exists():
        print(f"\nFAILED: Output file not created: {output_path}")
        return

    with output_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    errors = validate_predictions(predictions)

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"  Predictions: {len(predictions)}")
    print(f"  Unique image_ids: {len(set(p['image_id'] for p in predictions))}")
    print(f"  Unique category_ids: {len(set(p['category_id'] for p in predictions))}")
    print(f"  Inference time: {elapsed:.1f}s / {args.timeout}s budget")
    print(f"  Time per image: {elapsed / max(1, image_count):.2f}s")

    if predictions:
        scores = [p["score"] for p in predictions]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        cat_ids = [p["category_id"] for p in predictions]
        print(f"  Category ID range: [{min(cat_ids)}, {max(cat_ids)}]")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print("\n  ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
