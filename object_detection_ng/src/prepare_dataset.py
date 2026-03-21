"""Convert competition COCO dataset into YOLO format with train/val split.

Usage:
    python src/prepare_dataset.py \
        --annotations data/raw/NM_NGD_coco_dataset/annotations.json \
        --images_dir data/raw/NM_NGD_coco_dataset/images \
        --output_dir data/yolo \
        --val_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_coco(ann_path: Path) -> dict[str, Any]:
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_clamp_bbox(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[list[float], bool]:
    """Validate a COCO bbox [x, y, w, h] and clamp to image bounds.

    Returns (clamped_bbox, is_valid).
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return bbox, False

    x = max(0.0, min(x, img_w - 1))
    y = max(0.0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return [x, y, w, h], False

    return [x, y, w, h], True


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert COCO [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]


def build_category_mapping(categories: list[dict]) -> tuple[dict[int, int], dict[int, str], bool]:
    """Build category ID mapping. If IDs are contiguous 0..N-1, use direct mapping.
    Otherwise create a deterministic remapping.

    Returns (coco_id_to_yolo_idx, coco_id_to_name, ids_are_contiguous).
    """
    cat_ids = sorted(cat["id"] for cat in categories)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    expected_contiguous = list(range(len(cat_ids)))
    is_contiguous = (cat_ids == expected_contiguous)

    if is_contiguous:
        mapping = {cid: cid for cid in cat_ids}
        return mapping, cat_id_to_name, True

    # Non-contiguous: map sorted IDs to 0..N-1
    mapping = {cid: idx for idx, cid in enumerate(cat_ids)}
    return mapping, cat_id_to_name, False


def split_images(
    image_ids: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split image IDs into train and val sets."""
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return sorted(train_ids), sorted(val_ids)


def prepare_dataset(
    ann_path: Path,
    images_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    data = load_coco(ann_path)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    img_id_to_info = {img["id"]: img for img in images}
    coco_to_yolo, cat_id_to_name, contiguous = build_category_mapping(categories)

    print(f"Categories: {len(categories)}")
    print(f"Category IDs contiguous (0..{len(categories)-1}): {contiguous}")

    # Group annotations by image
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Split
    all_image_ids = sorted(img_id_to_info.keys())
    train_ids, val_ids = split_images(all_image_ids, val_ratio, seed)
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")

    # Create output dirs
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    total_labels = 0
    missing_images = 0

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in split_ids:
            img_info = img_id_to_info[img_id]
            img_filename = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            src_path = images_dir / img_filename
            if not src_path.exists():
                missing_images += 1
                print(f"  WARNING: Image not found: {src_path}")
                continue

            dst_img = output_dir / "images" / split_name / img_filename
            shutil.copy2(str(src_path), str(dst_img))

            label_lines = []
            for ann in anns_by_image.get(img_id, []):
                clamped_bbox, valid = validate_and_clamp_bbox(ann["bbox"], img_w, img_h)
                if not valid:
                    skipped += 1
                    continue

                cat_id = ann["category_id"]
                if cat_id not in coco_to_yolo:
                    skipped += 1
                    continue

                yolo_cls = coco_to_yolo[cat_id]
                yolo_bbox = coco_bbox_to_yolo(clamped_bbox, img_w, img_h)

                label_lines.append(
                    f"{yolo_cls} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                    f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                )
                total_labels += 1

            label_stem = Path(img_filename).stem
            label_path = output_dir / "labels" / split_name / f"{label_stem}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    # Build names dict for dataset.yaml (YOLO wants {idx: name})
    nc = len(categories)
    if contiguous:
        names = {cid: cat_id_to_name[cid] for cid in sorted(cat_id_to_name.keys())}
    else:
        names = {coco_to_yolo[cid]: cat_id_to_name[cid] for cid in sorted(cat_id_to_name.keys())}

    # Write dataset.yaml (JSON-compatible format since ultralytics can read YAML)
    yaml_content = f"path: {output_dir.resolve()}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += f"nc: {nc}\n"
    yaml_content += "names:\n"
    for idx in range(nc):
        name = names.get(idx, f"class_{idx}")
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Wrote {yaml_path}")

    # Save mapping file
    mapping_path = output_dir / "class_id_mapping.json"
    mapping_data = {
        "contiguous": contiguous,
        "nc": nc,
        "coco_id_to_yolo_idx": {str(k): v for k, v in coco_to_yolo.items()},
        "yolo_idx_to_coco_id": {str(v): k for k, v in coco_to_yolo.items()},
    }
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Wrote {mapping_path}")

    # Save split summary
    split_summary = {
        "total_images": len(all_image_ids),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "total_labels_written": total_labels,
        "skipped_annotations": skipped,
        "missing_images": missing_images,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
    }
    split_path = output_dir / "split_summary.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    print(f"Wrote {split_path}")

    print(f"\nDone: {total_labels} labels, {skipped} skipped, {missing_images} missing images")
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument("--annotations", type=str, required=True, help="Path to COCO annotations.json")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images folder")
    parser.add_argument("--output_dir", type=str, default="data/yolo", help="YOLO output root")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    prepare_dataset(ann_path, images_dir, output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
