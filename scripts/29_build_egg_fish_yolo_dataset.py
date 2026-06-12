from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml
from PIL import Image, ImageOps

from canteen_checkout.config import DETECTION_DIR, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.data_quality import file_sha256, hamming_distance_hex, perceptual_hash


CLASS_MAP = {"egg": 0, "fish": 1}
DATASET_CLASS_MAP = {
    "canh-chua": {"0": "fish", "ca": "fish"},
    "egg-detection": {"0": "egg", "egg": "egg"},
    "trung_thit_kho": {"0": "egg", "a": "egg"},
}
EXCLUDED_DATASETS = {"new_canh_chua"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_FIELDS = [
    "status",
    "reason",
    "dataset",
    "source_split",
    "target_split",
    "source_image",
    "source_label",
    "target_image",
    "target_label",
    "objects_in",
    "objects_out",
    "sha256",
    "phash",
]


@dataclass(frozen=True)
class SourceItem:
    dataset: str
    split: str
    image: Path
    label: Path


def relative_or_absolute(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_for(image: Path) -> Path | None:
    parts = list(image.parts)
    if "images" not in parts:
        return None
    idx = parts.index("images")
    yolo = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
    labeltxt = Path(*parts[:idx], "labelTxt", *parts[idx + 1 :]).with_suffix(".txt")
    if yolo.exists():
        return yolo
    if labeltxt.exists():
        return labeltxt
    return yolo


def load_items(source_root: Path) -> list[SourceItem]:
    items: list[SourceItem] = []
    for dataset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if dataset in EXCLUDED_DATASETS:
            continue
        if dataset not in DATASET_CLASS_MAP:
            continue
        for image in image_paths(dataset_dir):
            label = label_for(image)
            if label is None:
                continue
            rel = image.relative_to(dataset_dir).parts
            split = rel[0] if rel else "train"
            items.append(SourceItem(dataset=dataset, split=split, image=image, label=label))
    return items


def target_split(item: SourceItem) -> str:
    if item.split == "valid":
        return "valid"
    if item.split in {"train", "test"}:
        return item.split
    return "train"


def convert_line(line: str, dataset: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    label_key = parts[0]
    class_name = DATASET_CLASS_MAP[dataset].get(label_key)
    if class_name is None:
        return None
    class_id = CLASS_MAP[class_name]

    values = [float(value) for value in parts[1:]]
    if len(values) == 4:
        xc, yc, width, height = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        xc = (left + right) / 2
        yc = (top + bottom) / 2
        width = right - left
        height = bottom - top
    else:
        return None

    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    if width <= 0 or height <= 0:
        return None
    return f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def convert_label(label: Path, dataset: str) -> tuple[list[str], int]:
    if not label.exists():
        return [], 0
    lines = [line for line in label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    converted = [converted for line in lines if (converted := convert_line(line, dataset)) is not None]
    return converted, len(lines)


def unique_target(base: Path, suffix: str) -> Path:
    candidate = base.with_suffix(suffix)
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        next_candidate = base.with_name(f"{base.name}_{idx:03d}").with_suffix(suffix)
        if not next_candidate.exists():
            return next_candidate
        idx += 1


def image_phash(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return perceptual_hash(image)


def validate_no_near_duplicates(rows: list[dict[str, str]], threshold: int) -> int:
    phashes = [(row["phash"], row["target_image"]) for row in rows if row["status"] == "kept" and row["phash"]]
    near_pairs = 0
    for idx, (left_hash, _left_path) in enumerate(phashes):
        for right_hash, _right_path in phashes[idx + 1 :]:
            if hamming_distance_hex(left_hash, right_hash) <= threshold:
                near_pairs += 1
    return near_pairs


def write_yaml(output_root: Path) -> None:
    payload = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["egg", "fish"],
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_reports(rows: list[dict[str, str]], output_root: Path, source_root: Path, threshold: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "egg_fish_dataset_report.csv"
    json_path = REPORTS_DIR / "egg_fish_dataset_report.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    kept_rows = [row for row in rows if row["status"] == "kept"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": relative_or_absolute(source_root),
        "output_root": relative_or_absolute(output_root),
        "class_map": {"egg": 0, "fish": 1},
        "excluded_datasets": sorted(EXCLUDED_DATASETS),
        "total_rows": len(rows),
        "kept_images": len(kept_rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "kept_by_split": dict(Counter(row["target_split"] for row in kept_rows)),
        "kept_by_dataset": dict(Counter(row["dataset"] for row in kept_rows)),
        "object_counts": {
            "egg": sum(line.startswith("0 ") for row in kept_rows for line in Path(PROJECT_ROOT / row["target_label"]).read_text(encoding="utf-8").splitlines()),
            "fish": sum(line.startswith("1 ") for row in kept_rows for line in Path(PROJECT_ROOT / row["target_label"]).read_text(encoding="utf-8").splitlines()),
        },
        "near_duplicate_pairs_threshold": threshold,
        "near_duplicate_pairs_after_build": validate_no_near_duplicates(rows, threshold),
        "csv_report": relative_or_absolute(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified YOLO egg/fish detector dataset from deduped Roboflow exports.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "download" / "roboflow_yolo_deduped" / "20260612_181500")
    parser.add_argument("--out", type=Path, default=DETECTION_DIR / "egg_fish")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    if args.out.exists() and args.clear:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen_sha: set[str] = set()
    seen_phash: list[tuple[str, Path]] = []
    for item in load_items(args.source):
        image_sha = file_sha256(item.image)
        phash = image_phash(item.image)
        if image_sha in seen_sha:
            status = "skipped"
            reason = "exact_duplicate"
        elif any(hamming_distance_hex(phash, old_phash) <= args.phash_threshold for old_phash, _ in seen_phash):
            status = "skipped"
            reason = "near_duplicate"
        else:
            status = "kept"
            reason = ""

        converted_lines, objects_in = convert_label(item.label, item.dataset)
        if status == "kept":
            split = target_split(item)
            stem = f"{item.dataset}_{item.split}_{item.image.stem}"
            target_image = unique_target(args.out / split / "images" / stem, item.image.suffix.lower())
            target_label = target_image.parent.parent / "labels" / f"{target_image.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.image, target_image)
            target_label.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
            seen_sha.add(image_sha)
            seen_phash.append((phash, item.image))
        else:
            split = target_split(item)
            target_image = None
            target_label = None

        rows.append(
            {
                "status": status,
                "reason": reason,
                "dataset": item.dataset,
                "source_split": item.split,
                "target_split": split,
                "source_image": relative_or_absolute(item.image),
                "source_label": relative_or_absolute(item.label),
                "target_image": relative_or_absolute(target_image),
                "target_label": relative_or_absolute(target_label),
                "objects_in": str(objects_in),
                "objects_out": str(len(converted_lines) if status == "kept" else 0),
                "sha256": image_sha,
                "phash": phash,
            }
        )

    write_yaml(args.out)
    write_reports(rows, args.out, args.source, args.phash_threshold)


if __name__ == "__main__":
    main()
