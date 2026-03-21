"""Analyze the competition COCO annotations and generate a dataset summary report.

Usage:
    python src/analyze_dataset.py --annotations data/raw/NM_NGD_coco_dataset/annotations.json --output_dir data/interim
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_bbox_stats(annotations: list[dict]) -> dict[str, Any]:
    widths = []
    heights = []
    areas = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    def _stats(values: list[float], name: str) -> dict[str, float]:
        if not values:
            return {f"{name}_min": 0, f"{name}_max": 0, f"{name}_mean": 0, f"{name}_median": 0}
        return {
            f"{name}_min": round(min(values), 1),
            f"{name}_max": round(max(values), 1),
            f"{name}_mean": round(statistics.mean(values), 1),
            f"{name}_median": round(statistics.median(values), 1),
        }

    stats = {}
    stats.update(_stats(widths, "width"))
    stats.update(_stats(heights, "height"))
    stats.update(_stats(areas, "area"))

    small = sum(1 for a in areas if a < 32 * 32)
    medium = sum(1 for a in areas if 32 * 32 <= a < 96 * 96)
    large = sum(1 for a in areas if a >= 96 * 96)
    stats["small_objects"] = small
    stats["medium_objects"] = medium
    stats["large_objects"] = large

    return stats


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_ids = {img["id"] for img in images}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    cat_ids_in_file = sorted(cat_id_to_name.keys())

    ann_per_cat: Counter = Counter()
    corrected_count = 0
    not_corrected_count = 0
    orphan_annotations = 0
    invalid_bbox_count = 0
    anns_per_image: Counter = Counter()

    for ann in annotations:
        cat_id = ann["category_id"]
        ann_per_cat[cat_id] += 1
        anns_per_image[ann["image_id"]] += 1

        if ann.get("corrected", False):
            corrected_count += 1
        else:
            not_corrected_count += 1

        if ann["image_id"] not in image_ids:
            orphan_annotations += 1

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            invalid_bbox_count += 1

    unknown_cats = [cid for cid, name in cat_id_to_name.items() if "unknown" in name.lower()]
    missing_ids = []
    if cat_ids_in_file:
        full_range = set(range(min(cat_ids_in_file), max(cat_ids_in_file) + 1))
        missing_ids = sorted(full_range - set(cat_ids_in_file))

    zero_annotation_cats = [cid for cid in cat_ids_in_file if ann_per_cat[cid] == 0]

    bbox_stats = compute_bbox_stats(annotations)

    anns_per_img_values = list(anns_per_image.values())
    images_with_no_anns = len(image_ids - set(anns_per_image.keys()))

    summary = {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "min_category_id": min(cat_ids_in_file) if cat_ids_in_file else None,
        "max_category_id": max(cat_ids_in_file) if cat_ids_in_file else None,
        "category_ids_contiguous": len(missing_ids) == 0,
        "missing_category_ids": missing_ids[:20],
        "unknown_product_categories": unknown_cats,
        "corrected_annotations": corrected_count,
        "not_corrected_annotations": not_corrected_count,
        "orphan_annotations": orphan_annotations,
        "invalid_bbox_count": invalid_bbox_count,
        "zero_annotation_categories": len(zero_annotation_cats),
        "images_with_no_annotations": images_with_no_anns,
        "anns_per_image_min": min(anns_per_img_values) if anns_per_img_values else 0,
        "anns_per_image_max": max(anns_per_img_values) if anns_per_img_values else 0,
        "anns_per_image_mean": round(statistics.mean(anns_per_img_values), 1) if anns_per_img_values else 0,
        **bbox_stats,
    }

    cat_stats = []
    for cid in sorted(cat_id_to_name.keys()):
        cat_stats.append({
            "category_id": cid,
            "category_name": cat_id_to_name[cid],
            "annotation_count": ann_per_cat[cid],
        })

    return summary, cat_stats


def print_summary(summary: dict, cat_stats: list[dict]) -> None:
    print("=" * 70)
    print("DATASET ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Images:              {summary['num_images']}")
    print(f"  Annotations:         {summary['num_annotations']}")
    print(f"  Categories:          {summary['num_categories']}")
    print(f"  Category ID range:   {summary['min_category_id']} - {summary['max_category_id']}")
    print(f"  IDs contiguous:      {summary['category_ids_contiguous']}")
    print(f"  Unknown product IDs: {summary['unknown_product_categories']}")
    print(f"  Corrected anns:      {summary['corrected_annotations']}")
    print(f"  Not corrected anns:  {summary['not_corrected_annotations']}")
    print(f"  Orphan annotations:  {summary['orphan_annotations']}")
    print(f"  Invalid bboxes:      {summary['invalid_bbox_count']}")
    print(f"  Zero-ann categories: {summary['zero_annotation_categories']}")
    print(f"  Images w/o anns:     {summary['images_with_no_annotations']}")
    print()
    print("  Annotations per image:")
    print(f"    min={summary['anns_per_image_min']}  max={summary['anns_per_image_max']}  mean={summary['anns_per_image_mean']}")
    print()
    print("  Bbox stats:")
    for key in ["width", "height", "area"]:
        print(f"    {key}: min={summary[f'{key}_min']}  max={summary[f'{key}_max']}  "
              f"mean={summary[f'{key}_mean']}  median={summary[f'{key}_median']}")
    print(f"    Small (<32x32): {summary['small_objects']}  "
          f"Medium (32x32-96x96): {summary['medium_objects']}  "
          f"Large (>=96x96): {summary['large_objects']}")

    if summary["missing_category_ids"]:
        print(f"\n  WARNING: Missing category IDs in range: {summary['missing_category_ids']}")

    sorted_by_count = sorted(cat_stats, key=lambda x: x["annotation_count"], reverse=True)
    print("\n  TOP 20 most frequent categories:")
    for cs in sorted_by_count[:20]:
        print(f"    [{cs['category_id']:>3}] {cs['category_name'][:50]:50s}  count={cs['annotation_count']}")

    print("\n  BOTTOM 20 least frequent categories (non-zero):")
    non_zero = [cs for cs in sorted_by_count if cs["annotation_count"] > 0]
    for cs in reversed(non_zero[-20:]):
        print(f"    [{cs['category_id']:>3}] {cs['category_name'][:50]:50s}  count={cs['annotation_count']}")

    zero_cats = [cs for cs in cat_stats if cs["annotation_count"] == 0]
    if zero_cats:
        print(f"\n  Categories with ZERO annotations ({len(zero_cats)}):")
        for cs in zero_cats[:30]:
            print(f"    [{cs['category_id']:>3}] {cs['category_name']}")

    print("=" * 70)


def save_outputs(summary: dict, cat_stats: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {summary_path}")

    csv_path = output_dir / "category_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category_id", "category_name", "annotation_count"])
        writer.writeheader()
        writer.writerows(cat_stats)
    print(f"Saved category stats to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze competition COCO annotations")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations.json")
    parser.add_argument("--output_dir", type=str, default="data/interim", help="Output directory for reports")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")

    print(f"Loading annotations from {ann_path} ...")
    data = load_annotations(ann_path)

    summary, cat_stats = analyze(data)
    print_summary(summary, cat_stats)
    save_outputs(summary, cat_stats, Path(args.output_dir))


if __name__ == "__main__":
    main()
