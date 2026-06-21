from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch
from PIL import Image

from canteen_checkout.config import DEFAULT_DETECTOR_PATH, DEFAULT_MODEL_PATH, DEFAULT_REGION_DETECTOR_PATH, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.cropping import CropRegion, five_compartment_template
from canteen_checkout.detector import EGG_CLASS, FISH_CLASS, detect_objects, load_yolo_detector, partition_evidence_by_regions
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device
from canteen_checkout.region_detector import detect_food_regions


def denormalize_box(values: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = values
    return int(left * width), int(top * height), int(right * width), int(bottom * height)


def box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0 else 0.0


@torch.no_grad()
def top_predictions(model, class_names: list[str], image_size: int, image: Image.Image, device: torch.device) -> list[dict[str, object]]:
    probabilities = torch.softmax(model(eval_transforms(image_size)(image).unsqueeze(0).to(device)), dim=1)[0]
    values, indices = torch.topk(probabilities, min(3, len(class_names)))
    return [
        {"class_name": class_names[int(index)], "confidence": round(float(value), 4)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]


def draw_overlay(image_path: Path, regions: list[CropRegion], target: Path, source: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for region in regions:
        cv2.rectangle(image, (region.x, region.y), (region.x + region.w, region.y + region.h), (60, 220, 80), 3)
        cv2.putText(image, region.name, (region.x, max(24, region.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 80), 2)
    cv2.putText(image, source, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all three model roles on the official fixed-camera trays.")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data" / "demo" / "uploads")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "configs" / "official_tray_eval.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--region-detector", type=Path, default=DEFAULT_REGION_DETECTOR_PATH)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--region-threshold", type=float, default=0.35)
    parser.add_argument("--egg-threshold", type=float, default=0.15)
    parser.add_argument("--fish-threshold", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "diagnostics" / "official_trays")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "official_tray_evaluation.json")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["images"]
    device = resolve_device()
    classifier, class_names, image_size, _ = load_checkpoint(args.model, device)
    region_model = load_yolo_detector(args.region_detector)
    detector = load_yolo_detector(args.detector)

    results: dict[str, object] = {}
    aggregate = {"production_images": 0, "production_regions": 0, "production_top1_correct": 0, "production_top3_correct": 0}
    for filename, annotation in annotations.items():
        image_path = args.images_dir / filename
        if not image_path.exists():
            results[filename] = {"error": "image_missing"}
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        detected = detect_food_regions(region_model, image_path, detector_path=args.region_detector, confidence=args.region_threshold)
        if detected.regions:
            final_regions = list(detected.regions)
            region_source = "auto_detector"
        else:
            final_regions = five_compartment_template(width, height)
            region_source = "template_fallback"

        tray_evidence = detect_objects(
            detector,
            image_path,
            detector_path=args.detector,
            confidence=min(args.egg_threshold, args.fish_threshold),
            class_thresholds={EGG_CLASS: args.egg_threshold, FISH_CLASS: args.fish_threshold},
        )
        assigned_evidence = partition_evidence_by_regions(tray_evidence, final_regions)
        expected_rows = [
            {"class_name": row["class_name"], "box": denormalize_box(row["box"], width, height)}
            for row in annotation["regions"]
        ]
        crop_rows = []
        for expected in expected_rows:
            left, top, right, bottom = expected["box"]
            crop = image.crop((left, top, right, bottom))
            top3 = top_predictions(classifier, class_names, image_size, crop, device)
            crop_rows.append(
                {
                    "expected": expected["class_name"],
                    "box": expected["box"],
                    "top3": top3,
                    "top1_correct": top3[0]["class_name"] == expected["class_name"],
                    "top3_correct": expected["class_name"] in {row["class_name"] for row in top3},
                }
            )

        if annotation.get("production_orientation"):
            aggregate["production_images"] += 1
            aggregate["production_regions"] += len(final_regions)
            aggregate["production_top1_correct"] += sum(row["top1_correct"] for row in crop_rows)
            aggregate["production_top3_correct"] += sum(row["top3_correct"] for row in crop_rows)
        draw_overlay(image_path, final_regions, args.out / f"{image_path.stem}_regions.jpg", region_source)
        results[filename] = {
            "production_orientation": bool(annotation.get("production_orientation")),
            "region_source": region_source,
            "raw_region_count": detected.raw_detection_count,
            "region_fallback_reason": detected.fallback_reason,
            "final_region_count": len(final_regions),
            "tray_detector_evidence": tray_evidence.as_dict(),
            "assigned_detector_evidence": [row.as_dict() for row in assigned_evidence],
            "classifier_ground_truth_crops": crop_rows,
        }

    production_crops = aggregate["production_images"] * 5
    summary = {
        **aggregate,
        "production_region_recall": round(aggregate["production_regions"] / production_crops, 4) if production_crops else 0.0,
        "production_classifier_top1_accuracy": round(aggregate["production_top1_correct"] / production_crops, 4) if production_crops else 0.0,
        "production_classifier_top3_accuracy": round(aggregate["production_top3_correct"] / production_crops, 4) if production_crops else 0.0,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
