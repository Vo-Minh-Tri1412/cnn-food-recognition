from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from canteen_checkout.config import DETECTION_DIR, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.data_quality import hamming_distance_hex
from canteen_checkout.food_region_data import (
    DEFAULT_SOURCE_ORDER,
    OPTIONAL_GRID_SOURCE,
    SourceRecord,
    audit_source_root,
    load_coco_records,
    load_yolo_records,
    load_yolov4_records,
    select_representatives,
)


REPORT_FIELDS = [
    "status",
    "reason",
    "dataset",
    "source_split",
    "target_split",
    "source_group",
    "variant_count",
    "source_image",
    "target_image",
    "target_label",
    "boxes_out",
    "black_border_ratio",
    "blur_score",
    "sha256",
    "phash",
]


def relative_or_absolute(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def latest_source_root() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "archive").glob("raw_tray_datasets_*"))
    if not candidates:
        return PROJECT_ROOT / "data" / "archive" / "raw_tray_datasets_20260610_174513"
    return candidates[-1]


def load_dataset(dataset_dir: Path, min_area_ratio: float) -> tuple[list[SourceRecord], int]:
    if dataset_dir.name in {"Khay_thuc_an_2", "Khay_thuc_an_4"}:
        return load_coco_records(dataset_dir, min_area_ratio=min_area_ratio)
    if dataset_dir.name == OPTIONAL_GRID_SOURCE:
        return load_yolov4_records(dataset_dir, min_area_ratio=min_area_ratio)
    return load_yolo_records(dataset_dir, min_area_ratio=min_area_ratio)


def safe_stem(text: str) -> str:
    value = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return value.strip("_") or "image"


def write_dataset_yaml(output_root: Path) -> None:
    payload = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["food_region"],
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def report_row(
    record: SourceRecord,
    *,
    status: str,
    reason: str,
    target_image: Path | None = None,
    target_label: Path | None = None,
) -> dict[str, str]:
    return {
        "status": status,
        "reason": reason,
        "dataset": record.dataset,
        "source_split": record.split,
        "target_split": record.split,
        "source_group": record.source_group,
        "variant_count": str(record.variant_count),
        "source_image": relative_or_absolute(record.image_path),
        "target_image": relative_or_absolute(target_image),
        "target_label": relative_or_absolute(target_label),
        "boxes_out": str(len(record.boxes) if status == "kept" else 0),
        "black_border_ratio": f"{record.black_border_ratio:.6f}",
        "blur_score": f"{record.blur_score:.4f}",
        "sha256": record.sha256,
        "phash": record.phash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned one-class food-region YOLO dataset from archived Roboflow exports.")
    parser.add_argument("--source-root", type=Path, default=latest_source_root())
    parser.add_argument("--out", type=Path, default=DETECTION_DIR / "food_regions")
    parser.add_argument("--include-grid-source", action="store_true", help="Include Khay_thuc_an_3 compartment labels. Disabled by default because empty grids are labeled.")
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "food_region_dataset_report.csv")
    args = parser.parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f"Roboflow tray archive not found: {args.source_root}")
    source_names = list(DEFAULT_SOURCE_ORDER)
    if args.include_grid_source:
        source_names.append(OPTIONAL_GRID_SOURCE)

    discovered: list[SourceRecord] = []
    raw_boxes_by_dataset: dict[str, int] = {}
    loaded_records_by_dataset: dict[str, int] = {}
    for dataset_name in source_names:
        dataset_dir = args.source_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Expected Roboflow source missing: {dataset_dir}")
        records, raw_boxes = load_dataset(dataset_dir, args.min_area_ratio)
        discovered.extend(records)
        raw_boxes_by_dataset[dataset_name] = raw_boxes
        loaded_records_by_dataset[dataset_name] = len(records)

    selected, augmented = select_representatives(discovered)
    priority = {name: index for index, name in enumerate(source_names)}
    selected.sort(key=lambda row: (priority[row.dataset], row.split, row.source_group, row.image_path.as_posix()))

    kept: list[SourceRecord] = []
    rows = [report_row(record, status="skipped", reason="roboflow_augmented_variant") for record in augmented]
    seen_sha: set[str] = set()
    seen_phash: list[str] = []
    for record in selected:
        reason = ""
        if record.sha256 in seen_sha:
            reason = "exact_duplicate"
        elif any(hamming_distance_hex(record.phash, old_hash) <= args.phash_threshold for old_hash in seen_phash):
            reason = "near_duplicate"
        if reason:
            rows.append(report_row(record, status="skipped", reason=reason))
            continue
        kept.append(record)
        seen_sha.add(record.sha256)
        seen_phash.append(record.phash)

    if not args.dry_run:
        if args.out.exists() and args.clear:
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True, exist_ok=True)
        for record in kept:
            stem = safe_stem(f"{record.dataset}_{record.split}_{record.source_group}_{record.sha256[:10]}")
            target_image = args.out / record.split / "images" / f"{stem}{record.image_path.suffix.lower()}"
            target_label = args.out / record.split / "labels" / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.image_path, target_image)
            target_label.write_text("\n".join(box.as_yolo() for box in record.boxes) + "\n", encoding="utf-8")
            rows.append(report_row(record, status="kept", reason="", target_image=target_image, target_label=target_label))
        write_dataset_yaml(args.out)
    else:
        rows.extend(report_row(record, status="kept", reason="dry_run") for record in kept)

    rows.sort(key=lambda row: (row["dataset"], row["source_group"], row["status"], row["source_image"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "source_root": relative_or_absolute(args.source_root),
        "output_root": relative_or_absolute(args.out),
        "included_sources": source_names,
        "excluded_by_default": [] if args.include_grid_source else [OPTIONAL_GRID_SOURCE],
        "min_area_ratio": args.min_area_ratio,
        "phash_threshold": args.phash_threshold,
        "source_audit": audit_source_root(args.source_root),
        "loaded_records_by_dataset": loaded_records_by_dataset,
        "raw_boxes_by_dataset": raw_boxes_by_dataset,
        "representatives_before_dedupe": len(selected),
        "kept_images": len(kept),
        "kept_boxes": sum(len(record.boxes) for record in kept),
        "kept_by_dataset": dict(Counter(record.dataset for record in kept)),
        "kept_by_split": dict(Counter(record.split for record in kept)),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "report": relative_or_absolute(args.report),
    }

    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary_path = args.report.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_report"] = relative_or_absolute(summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
