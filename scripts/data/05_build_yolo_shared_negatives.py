from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from canteen_checkout.config import DETECTION_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT, REPORTS_DIR, REVIEWED_DIR
from canteen_checkout.data_quality import (
    assess_image,
    hamming_distance_hex,
    normalize_image,
    perceptual_hash,
    quality_reasons,
)


SAFE_NEGATIVE_CLASSES = {
    "com_trang": "no_egg_no_fish",
    "dau_hu_sot_ca": "no_egg_no_fish",
    "thit_kho": "no_egg_hard_negative",
    "canh_chua_khong_ca": "no_fish_hard_negative",
    "canh_rau": "no_fish_soup_negative",
    "rau_xao": "no_egg_no_fish",
    "suon_nuong": "no_egg_no_fish",
}

UNSAFE_FOR_EMPTY_YOLO = {
    "ca_hu_kho": "contains fish but has no bbox labels",
    "canh_chua_co_ca": "contains fish but has no bbox labels",
    "thit_kho_trung": "contains egg but has no bbox labels",
    "trung_chien": "contains egg but has no bbox labels",
}

REPORT_FIELDS = [
    "status",
    "reason",
    "source_class",
    "negative_type",
    "source_image",
    "target_split",
    "target_image",
    "target_label",
    "sha256",
    "phash",
    "width",
    "height",
]


@dataclass(frozen=True)
class SeenImage:
    phash: str
    path: Path


def relative_or_absolute(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_base_dataset(base: Path, output: Path) -> None:
    if not base.exists():
        raise FileNotFoundError(f"Base YOLO dataset not found: {base}")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        for folder in ("images", "labels"):
            source_dir = base / split / folder
            target_dir = output / split / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            if not source_dir.exists():
                continue
            for source in sorted(source_dir.iterdir()):
                if source.is_file():
                    shutil.copy2(source, target_dir / source.name)
    data_yaml = base / "data.yaml"
    if data_yaml.exists():
        shutil.copy2(data_yaml, output / "data.yaml")
    else:
        write_yaml(output)


def write_yaml(output: Path) -> None:
    payload = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["egg", "fish"],
    }
    (output / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def index_existing_images(root: Path, threshold: int) -> tuple[set[str], list[SeenImage]]:
    seen_sha: set[str] = set()
    seen_phash: list[SeenImage] = []
    for image_path in image_paths(root):
        image, metrics, reasons = assess_image(image_path)
        if image is None or metrics is None or reasons:
            continue
        seen_sha.add(metrics.sha256)
        seen_phash.append(SeenImage(phash=metrics.phash, path=image_path))
    return seen_sha, seen_phash


def is_near_duplicate(phash: str, seen: list[SeenImage], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, item.phash) <= threshold for item in seen)


def split_for_index(index: int, total: int) -> str:
    valid_start = int(total * 0.8)
    test_start = int(total * 0.9)
    if index < valid_start:
        return "train"
    if index < test_start:
        return "valid"
    return "test"


def count_yolo_dataset(root: Path) -> dict[str, object]:
    split_counts: dict[str, dict[str, int]] = {}
    object_counts = Counter()
    empty_labels = 0
    for split in ("train", "valid", "test"):
        images = image_paths(root / split / "images")
        labels = sorted((root / split / "labels").glob("*.txt")) if (root / split / "labels").exists() else []
        split_counts[split] = {"images": len(images), "labels": len(labels)}
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                if line.startswith("0 "):
                    object_counts["egg"] += 1
                elif line.startswith("1 "):
                    object_counts["fish"] += 1
    return {
        "split_counts": split_counts,
        "object_counts": dict(object_counts),
        "empty_label_images": empty_labels,
    }


def build_shared_negatives(
    *,
    base: Path,
    reviewed: Path,
    output: Path,
    max_per_class: int,
    seed: int,
    phash_threshold: int,
    image_size: int,
    clear: bool,
) -> dict[str, object]:
    if output.exists() and clear:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists and is not empty. Use --clear to rebuild: {output}")

    copy_base_dataset(base, output)
    seen_sha, seen_phash = index_existing_images(output, phash_threshold)
    rows: list[dict[str, str]] = []
    selected_by_class: Counter[str] = Counter()

    rng = random.Random(seed)
    for class_name, negative_type in SAFE_NEGATIVE_CLASSES.items():
        source_dir = reviewed / class_name
        candidates = image_paths(source_dir) if source_dir.exists() else []
        rng.shuffle(candidates)

        selected_for_class: list[tuple[Path, object, object]] = []
        for source in candidates:
            image, metrics, errors = assess_image(source)
            if image is None or metrics is None or errors:
                rows.append(make_row("skipped", ";".join(errors) or "invalid_image", class_name, negative_type, source, None, None, None, None, None))
                continue

            reasons = quality_reasons(metrics)
            if reasons:
                rows.append(make_row("skipped", ";".join(reasons), class_name, negative_type, source, None, None, metrics, None, None))
                continue
            if metrics.sha256 in seen_sha:
                rows.append(make_row("skipped", "exact_duplicate_against_yolo", class_name, negative_type, source, None, None, metrics, None, None))
                continue
            if is_near_duplicate(metrics.phash, seen_phash, phash_threshold):
                rows.append(make_row("skipped", "near_duplicate_against_yolo_or_selected", class_name, negative_type, source, None, None, metrics, None, None))
                continue

            selected_for_class.append((source, image, metrics))
            seen_sha.add(metrics.sha256)
            seen_phash.append(SeenImage(metrics.phash, source))
            if len(selected_for_class) >= max_per_class:
                break

        total = len(selected_for_class)
        for index, (source, image, metrics) in enumerate(selected_for_class):
            split = split_for_index(index, total)
            stem = f"reviewedneg_{class_name}_{metrics.sha256[:12]}"
            target_image = output / split / "images" / f"{stem}.jpg"
            target_label = output / split / "labels" / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            normalized = normalize_image(image, image_size=image_size, mode="pad")
            normalized.save(target_image, "JPEG", quality=92, optimize=True)
            target_label.write_text("", encoding="utf-8")
            selected_by_class[class_name] += 1
            rows.append(make_row("kept", "", class_name, negative_type, source, split, target_image, metrics, target_label, None))

    write_yaml(output)
    return write_reports(rows, base, reviewed, output, max_per_class, seed, phash_threshold, image_size, selected_by_class)


def make_row(
    status: str,
    reason: str,
    source_class: str,
    negative_type: str,
    source_image: Path,
    target_split: str | None,
    target_image: Path | None,
    metrics: object | None,
    target_label: Path | None,
    _unused: object | None,
) -> dict[str, str]:
    return {
        "status": status,
        "reason": reason,
        "source_class": source_class,
        "negative_type": negative_type,
        "source_image": relative_or_absolute(source_image),
        "target_split": target_split or "",
        "target_image": relative_or_absolute(target_image),
        "target_label": relative_or_absolute(target_label),
        "sha256": getattr(metrics, "sha256", ""),
        "phash": getattr(metrics, "phash", ""),
        "width": str(getattr(metrics, "width", "")),
        "height": str(getattr(metrics, "height", "")),
    }


def write_reports(
    rows: list[dict[str, str]],
    base: Path,
    reviewed: Path,
    output: Path,
    max_per_class: int,
    seed: int,
    phash_threshold: int,
    image_size: int,
    selected_by_class: Counter[str],
) -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "egg_fish_shared_negatives_report.csv"
    json_path = REPORTS_DIR / "egg_fish_shared_negatives_report.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    kept_rows = [row for row in rows if row["status"] == "kept"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_yolo_dataset": relative_or_absolute(base),
        "reviewed_root": relative_or_absolute(reviewed),
        "output_root": relative_or_absolute(output),
        "safe_negative_classes": SAFE_NEGATIVE_CLASSES,
        "excluded_classes_for_empty_yolo": UNSAFE_FOR_EMPTY_YOLO,
        "max_per_class": max_per_class,
        "seed": seed,
        "phash_threshold": phash_threshold,
        "image_size": image_size,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "selected_by_class": dict(selected_by_class),
        "selected_total": len(kept_rows),
        "selected_by_split": dict(Counter(row["target_split"] for row in kept_rows)),
        "dataset_counts_after_build": count_yolo_dataset(output),
        "csv_report": relative_or_absolute(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add trusted reviewed classification images as empty-label hard negatives for YOLO egg/fish detection.")
    parser.add_argument("--base", type=Path, default=DETECTION_DIR / "egg_fish")
    parser.add_argument("--reviewed", type=Path, default=REVIEWED_DIR)
    parser.add_argument("--out", type=Path, default=DETECTION_DIR / "egg_fish_shared")
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1412)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    build_shared_negatives(
        base=args.base,
        reviewed=args.reviewed,
        output=args.out,
        max_per_class=args.max_per_class,
        seed=args.seed,
        phash_threshold=args.phash_threshold,
        image_size=args.image_size,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
