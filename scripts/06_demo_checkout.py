from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import Image

from canteen_checkout.config import BILLS_DIR, CROPPED_DISHES_DIR, DEFAULT_DETECTOR_PATH, DEFAULT_MODEL_PATH
from canteen_checkout.cropping import crop_regions, five_compartment_template, load_regions
from canteen_checkout.detector import detect_objects, empty_evidence, fuse_decision, load_yolo_detector
from canteen_checkout.io_utils import load_prices
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device
from canteen_checkout.pricing import THIT_KHO_TRUNG_CLASS, dish_price


@torch.no_grad()
def predict_crop(model, class_names: list[str], image_size: int, crop_path: Path, device: torch.device) -> tuple[str, float]:
    transform = eval_transforms(image_size)
    image = Image.open(crop_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, idx = torch.max(probs, dim=0)
    return class_names[int(idx)], float(confidence.cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkout demo on one tray image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--use-detector", action="store_true", help="Use YOLO egg/fish detector fusion if the detector exists.")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--regions-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--ignore-region", action="append", default=[], help="Region name to crop but exclude from billing. Can be repeated.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image
    out_dir = args.out_dir or (CROPPED_DISHES_DIR / image_path.stem)

    if args.regions_json:
        regions = load_regions(args.regions_json)
    else:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]
        regions = five_compartment_template(w, h)

    ignored_regions = set(args.ignore_region)
    crop_paths = crop_regions(image_path, regions, out_dir)
    prices = load_prices()

    model = None
    class_names: list[str] = []
    image_size = 224
    device = resolve_device()
    if args.model.exists():
        model, class_names, image_size, _ = load_checkpoint(args.model, device)
        print(f"Loaded model: {args.model}")
    else:
        print(f"Model not found: {args.model}. Bill will mark crops as unknown.")

    detector = None
    detector_loaded = False
    if args.use_detector:
        if args.detector.exists():
            detector = load_yolo_detector(args.detector)
            detector_loaded = True
            print(f"Loaded detector: {args.detector}")
        else:
            print(f"Detector not found: {args.detector}. Fusion disabled.")

    items = []
    total = 0
    for crop_path, region in zip(crop_paths, regions):
        forced_label = region.label or ""
        ignored = region.name in ignored_regions or forced_label in {"ignore", "ignored", "unknown", "other", "extra"}
        if ignored:
            class_name = forced_label or "ignored"
            confidence = 1.0
            uncertain = True
        elif forced_label:
            class_name = forced_label
            confidence = 1.0
            uncertain = False
        elif model is not None:
            class_name, confidence = predict_crop(model, class_names, image_size, crop_path, device)
            uncertain = confidence < args.threshold
        else:
            class_name = "unknown"
            confidence = 0.0
            uncertain = True

        raw_class_name = class_name
        raw_confidence = confidence
        evidence = empty_evidence(args.detector, detector_loaded)
        fusion_reason = "classifier_only"
        final_egg_count = 1 if class_name == THIT_KHO_TRUNG_CLASS else None
        if detector is not None and not ignored and not forced_label:
            evidence = detect_objects(detector, crop_path, detector_path=args.detector, confidence=args.detector_threshold)
            fusion = fuse_decision(
                raw_class_name=raw_class_name,
                raw_confidence=raw_confidence,
                uncertain=uncertain,
                evidence=evidence,
            )
            class_name = fusion.class_name
            confidence = fusion.confidence
            uncertain = fusion.uncertain
            final_egg_count = fusion.egg_count
            fusion_reason = fusion.fusion_reason

        price_row = prices.get(class_name)
        price_info = dish_price(
            class_name,
            prices,
            uncertain=uncertain,
            egg_count=final_egg_count if class_name == THIT_KHO_TRUNG_CLASS else None,
        )
        price_vnd = price_info.total_price_vnd
        display_name = class_name if price_row is None else price_row.display_name
        total += price_vnd
        items.append(
            {
                "crop_path": str(crop_path),
                "region_name": region.name,
                "raw_class_name": raw_class_name,
                "raw_confidence": round(raw_confidence, 4),
                "class_name": class_name,
                "display_name": display_name,
                "confidence": round(confidence, 4),
                "uncertain": uncertain,
                "ignored": ignored,
                "egg_count": price_info.egg_count,
                "fish_count": evidence.fish_count,
                "detections": [detection.as_dict() for detection in evidence.detections],
                "detector_evidence": evidence.as_dict(),
                "fusion_reason": fusion_reason,
                "base_price_vnd": price_info.base_price_vnd,
                "extra_price_vnd": price_info.extra_price_vnd,
                "price_vnd": price_vnd,
            }
        )

    bill = {
        "image_path": str(image_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model) if args.model.exists() else None,
        "detector_path": str(args.detector) if args.use_detector and args.detector.exists() else None,
        "detector_loaded": detector_loaded,
        "threshold": args.threshold,
        "detector_threshold": args.detector_threshold,
        "items": items,
        "total_vnd": total,
    }
    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    bill_path = BILLS_DIR / f"{image_path.stem}_bill.json"
    bill_path.write_text(json.dumps(bill, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Detected dishes:")
    for idx, item in enumerate(items, 1):
        marker = " (ignored)" if item["ignored"] else " (uncertain)" if item["uncertain"] else ""
        egg_note = f", eggs={item['egg_count']}" if item["egg_count"] and item["egg_count"] > 1 else ""
        print(
            f"{idx}. {item['display_name']} - {item['price_vnd']:,} VND "
            f"- conf={item['confidence']:.2f}{egg_note}{marker}"
        )
    print(f"Total: {total:,} VND")
    print(f"Bill JSON: {bill_path}")


if __name__ == "__main__":
    main()
