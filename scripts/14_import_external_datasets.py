from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageOps

from canteen_checkout.config import DISH_CLASSES, DOWNLOADS_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT
from canteen_checkout.data_quality import (
    ImageMetrics,
    blur_score,
    brightness_mean,
    hamming_distance_hex,
    normalize_image,
    perceptual_hash,
    quality_reasons,
)


DIRECT_DATASETS = [
    ("Ca kho to", "ca_hu_kho", "ca_hu_kho", True),
    ("Thit kho", "thit_kho_or_thit_kho_trung", "", True),
    ("Canh chua", "canh_chua_unknown", "", True),
]

ROBOFLOW1_POOL_MAP = {
    "rice": ("com_trang", "com_trang", False),
    "egg": ("trung_chien", "trung_chien", True),
    "omelette": ("trung_chien", "trung_chien", True),
    "soup": ("canh_chua_unknown", "", True),
    "fried vegetable": ("rau_xao_or_canh_rau", "", True),
    "vegetable": ("rau_xao_or_canh_rau", "", True),
    "stir-fried": ("rau_xao_or_canh_rau", "", True),
    "braised dish": ("protein_grid_review", "", True),
    "braised meat": ("protein_grid_review", "", True),
    "meat": ("protein_grid_review", "", True),
    "fried meat": ("protein_grid_review", "", True),
    "fried fish": ("protein_grid_review", "", True),
}

ROBOFLOW3_POOL_MAP = {
    "grain_grid": ("com_trang", "com_trang", False),
    "protein_grid": ("protein_grid_review", "", True),
    "vegetable_grid_1": ("rau_xao_or_canh_rau", "", True),
    "vegetable_grid_2": ("rau_xao_or_canh_rau", "", True),
}

POOL_BASELINE_CLASSES = {
    "ca_hu_kho": ["ca_hu_kho"],
    "thit_kho_or_thit_kho_trung": ["thit_kho", "thit_kho_trung"],
    "canh_chua_unknown": ["canh_chua_co_ca", "canh_chua_khong_ca"],
    "com_trang": ["com_trang"],
    "trung_chien": ["trung_chien"],
    "rau_xao_or_canh_rau": ["rau_xao", "canh_rau"],
    "protein_grid_review": ["thit_kho", "thit_kho_trung", "ca_hu_kho", "suon_nuong"],
    "unknown_food_crops": list(DISH_CLASSES),
}


@dataclass(frozen=True)
class Candidate:
    image: Image.Image
    pool: str
    suggested_class: str
    needs_review: bool
    source_dataset: str
    source_path: Path
    source_split: str
    label_name: str
    annotation_format: str
    method: str
    crop_box: tuple[int, int, int, int] | None = None


def safe_name(text: str, default: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:120] or default


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_sha256(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def image_metrics(image: Image.Image) -> ImageMetrics:
    width, height = image.size
    aspect = max(width / max(height, 1), height / max(width, 1))
    return ImageMetrics(
        width=width,
        height=height,
        aspect_ratio=aspect,
        brightness=brightness_mean(image),
        blur_score=blur_score(image),
        phash=perceptual_hash(image),
        sha256=image_sha256(image),
    )


def is_near_duplicate(phash: str, seen: Iterable[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, old) <= threshold for old in seen)


def unique_output_path(out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.jpg"
    if not path.exists():
        return path
    idx = 1
    while True:
        path = out_dir / f"{stem}_{idx:03d}.jpg"
        if not path.exists():
            return path
        idx += 1


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92, optimize=True)


def expanded_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box).convert("RGB")


def yolo_line_to_box(parts: list[str], image_size: tuple[int, int], pad_ratio: float) -> tuple[int, int, int, int] | None:
    coords = [float(item) for item in parts[1:]]
    width, height = image_size
    if len(coords) == 4:
        cx, cy, w, h = coords
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs) * width, max(xs) * width
        y1, y2 = min(ys) * height, max(ys) * height
    else:
        return None
    return expanded_box((x1, y1, x2, y2), image_size, pad_ratio)


def xyxy_token_to_box(token: str, image_size: tuple[int, int], pad_ratio: float) -> tuple[int, int, int, int] | None:
    parts = token.split(",")
    if len(parts) != 5:
        return None
    x1, y1, x2, y2 = [float(item) for item in parts[:4]]
    return expanded_box((x1, y1, x2, y2), image_size, pad_ratio)


def parse_roboflow_yaml_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*(\d+):\s*(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def parse_classes_file(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    return {idx: line.strip() for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()) if line.strip()}


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(image_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def iter_direct_dataset(downloads_root: Path) -> Iterable[Candidate]:
    for dataset_name, pool, suggested_class, needs_review in DIRECT_DATASETS:
        dataset_root = downloads_root / dataset_name
        for path in list_images(dataset_root):
            try:
                image = open_rgb(path)
            except Exception:
                continue
            yield Candidate(
                image=image,
                pool=pool,
                suggested_class=suggested_class,
                needs_review=needs_review,
                source_dataset=dataset_name,
                source_path=path,
                source_split="",
                label_name=dataset_name,
                annotation_format="folder",
                method="direct_image",
            )


def iter_roboflow_yolo_segments(root: Path, pad_ratio: float) -> Iterable[Candidate]:
    names = parse_roboflow_yaml_names(root / "data.yaml")
    for split in ["train", "valid", "test"]:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image_for_stem(image_dir, label_path.stem)
            if image_path is None:
                continue
            try:
                image = open_rgb(image_path)
            except Exception:
                continue
            for line_idx, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                label_name = names.get(class_id, f"class_{class_id}")
                if label_name not in ROBOFLOW1_POOL_MAP:
                    continue
                box = yolo_line_to_box(parts, image.size, pad_ratio)
                if box is None:
                    continue
                pool, suggested_class, needs_review = ROBOFLOW1_POOL_MAP[label_name]
                yield Candidate(
                    image=crop_box(image, box),
                    pool=pool,
                    suggested_class=suggested_class,
                    needs_review=needs_review,
                    source_dataset=root.name,
                    source_path=image_path,
                    source_split=split,
                    label_name=label_name,
                    annotation_format="yolo_polygon_or_bbox",
                    method=f"roboflow_yolo:{label_name}",
                    crop_box=box,
                )


def iter_roboflow_coco_food(root: Path, pad_ratio: float) -> Iterable[Candidate]:
    for split in ["train", "valid", "test"]:
        split_dir = root / split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {item["id"]: item for item in data.get("images", [])}
        for ann in data.get("annotations", []):
            image_info = images.get(ann.get("image_id"))
            if not image_info:
                continue
            image_path = split_dir / image_info["file_name"]
            if not image_path.exists():
                continue
            try:
                image = open_rgb(image_path)
            except Exception:
                continue
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            box = expanded_box((x, y, x + w, y + h), image.size, pad_ratio)
            if box is None:
                continue
            yield Candidate(
                image=crop_box(image, box),
                pool="unknown_food_crops",
                suggested_class="",
                needs_review=True,
                source_dataset=root.name,
                source_path=image_path,
                source_split=split,
                label_name="food",
                annotation_format="coco_bbox",
                method="roboflow_coco:food",
                crop_box=box,
            )


def iter_roboflow_grid(root: Path, pad_ratio: float) -> Iterable[Candidate]:
    for split in ["train", "valid", "test"]:
        split_dir = root / split
        annotation_path = split_dir / "_annotations.txt"
        names = parse_classes_file(split_dir / "_classes.txt")
        if not annotation_path.exists():
            continue
        for line in annotation_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            image_path = split_dir / parts[0]
            if not image_path.exists():
                continue
            try:
                image = open_rgb(image_path)
            except Exception:
                continue
            for token in parts[1:]:
                class_parts = token.split(",")
                if len(class_parts) != 5:
                    continue
                class_id = int(float(class_parts[4]))
                label_name = names.get(class_id, f"class_{class_id}")
                if label_name not in ROBOFLOW3_POOL_MAP:
                    continue
                box = xyxy_token_to_box(token, image.size, pad_ratio)
                if box is None:
                    continue
                pool, suggested_class, needs_review = ROBOFLOW3_POOL_MAP[label_name]
                yield Candidate(
                    image=crop_box(image, box),
                    pool=pool,
                    suggested_class=suggested_class,
                    needs_review=needs_review,
                    source_dataset=root.name,
                    source_path=image_path,
                    source_split=split,
                    label_name=label_name,
                    annotation_format="yolo_v4_txt_bbox",
                    method=f"roboflow_grid:{label_name}",
                    crop_box=box,
                )


def load_baseline_phashes(baseline_root: Path | None) -> dict[str, list[str]]:
    phashes: dict[str, list[str]] = defaultdict(list)
    if baseline_root is None or not baseline_root.exists():
        return phashes
    for pool, class_names in POOL_BASELINE_CLASSES.items():
        for class_name in class_names:
            for path in list_images(baseline_root / class_name):
                try:
                    image = open_rgb(path)
                    phashes[pool].append(perceptual_hash(image))
                except Exception:
                    continue
    return phashes


def candidate_iterators(downloads_root: Path, pad_ratio: float) -> list[Iterable[Candidate]]:
    return [
        iter_direct_dataset(downloads_root),
        iter_roboflow_yolo_segments(downloads_root / "Khay_thuc_an", pad_ratio),
        iter_roboflow_coco_food(downloads_root / "Khay_thuc_an_2", pad_ratio),
        iter_roboflow_grid(downloads_root / "Khay_thuc_an_3", pad_ratio),
    ]


def row_for_candidate(
    candidate: Candidate,
    *,
    status: str,
    reason: str,
    output_path: str,
    metrics: ImageMetrics | None,
) -> dict[str, str | int | float | bool]:
    box = "" if candidate.crop_box is None else ",".join(str(item) for item in candidate.crop_box)
    return {
        "pool": candidate.pool,
        "suggested_class": candidate.suggested_class,
        "needs_review": candidate.needs_review,
        "status": status,
        "reason": reason,
        "source_dataset": candidate.source_dataset,
        "source_path": relative_or_absolute(candidate.source_path),
        "source_split": candidate.source_split,
        "label_name": candidate.label_name,
        "annotation_format": candidate.annotation_format,
        "method": candidate.method,
        "crop_box": box,
        "output_path": output_path,
        "width": metrics.width if metrics else "",
        "height": metrics.height if metrics else "",
        "aspect_ratio": round(metrics.aspect_ratio, 4) if metrics else "",
        "brightness": round(metrics.brightness, 4) if metrics else "",
        "blur_score": round(metrics.blur_score, 4) if metrics else "",
        "phash": metrics.phash if metrics else "",
        "sha256": metrics.sha256 if metrics else "",
    }


def make_contact_sheets(source_root: Path, out_root: Path, *, thumb_size: int = 180, cols: int = 5, max_images: int = 120) -> None:
    for pool_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        images = list_images(pool_dir)
        if not images:
            continue
        for page_idx in range(0, len(images), max_images):
            chunk = images[page_idx : page_idx + max_images]
            rows = max(1, (len(chunk) + cols - 1) // cols)
            cell_h = thumb_size + 38
            sheet = Image.new("RGB", (cols * thumb_size, rows * cell_h), "white")
            draw = ImageDraw.Draw(sheet)
            for idx, path in enumerate(chunk):
                image = open_rgb(path)
                image.thumbnail((thumb_size, thumb_size))
                x = (idx % cols) * thumb_size
                y = (idx // cols) * cell_h
                sheet.paste(image, (x + (thumb_size - image.width) // 2, y))
                draw.text((x + 4, y + thumb_size + 3), f"{page_idx + idx:04d} {pool_dir.name}", fill=(0, 0, 0))
                draw.text((x + 4, y + thumb_size + 20), path.name[:28], fill=(80, 80, 80))
            out_root.mkdir(parents=True, exist_ok=True)
            suffix = page_idx // max_images + 1
            out_path = out_root / f"{pool_dir.name}_{suffix:02d}.jpg"
            sheet.save(out_path, quality=92)


def process_candidates(args: argparse.Namespace) -> tuple[Counter, Counter, list[dict[str, str | int | float | bool]]]:
    baseline_phashes = load_baseline_phashes(args.baseline)
    external_phashes: dict[str, list[str]] = defaultdict(list)
    status_counts: Counter = Counter()
    pool_counts: Counter = Counter()
    rows: list[dict[str, str | int | float | bool]] = []

    for iterator in candidate_iterators(args.downloads_root, args.crop_pad):
        for candidate in iterator:
            metrics = image_metrics(candidate.image)
            reasons = quality_reasons(
                metrics,
                min_size=args.min_size,
                max_aspect_ratio=args.max_aspect_ratio,
                min_blur_score=args.min_blur_score,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
            )
            pool_baseline = baseline_phashes.get(candidate.pool, [])
            if not reasons and is_near_duplicate(metrics.phash, pool_baseline, args.duplicate_hamming):
                reasons.append("duplicate_baseline")
            if not reasons and is_near_duplicate(metrics.phash, external_phashes[candidate.pool], args.duplicate_hamming):
                reasons.append("duplicate_external")

            if reasons:
                reason = ";".join(reasons)
                status_counts["rejected"] += 1
                if not reason.startswith("duplicate"):
                    rejected_dir = args.out / "rejected" / reasons[0] / candidate.pool
                    output = unique_output_path(rejected_dir, f"{safe_name(candidate.source_dataset)}_{safe_name(candidate.source_path.stem)}_{metrics.sha256[:8]}")
                    if not args.dry_run:
                        save_jpeg(normalize_image(candidate.image, image_size=args.image_size, mode=args.mode), output)
                    output_path = relative_or_absolute(output)
                else:
                    output_path = ""
                rows.append(row_for_candidate(candidate, status="rejected", reason=reason, output_path=output_path, metrics=metrics))
                continue

            external_phashes[candidate.pool].append(metrics.phash)
            status_counts["accepted"] += 1
            pool_counts[candidate.pool] += 1
            out_dir = args.out / "review" / candidate.pool
            output = unique_output_path(out_dir, f"{safe_name(candidate.source_dataset)}_{safe_name(candidate.source_path.stem)}_{metrics.sha256[:8]}")
            if not args.dry_run:
                normalized = normalize_image(candidate.image, image_size=args.image_size, mode=args.mode)
                save_jpeg(normalized, output)
            rows.append(row_for_candidate(candidate, status="accepted", reason="", output_path=relative_or_absolute(output), metrics=metrics))
    return status_counts, pool_counts, rows


def write_manifest(rows: list[dict[str, str | int | float | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_reviewed_dirs(out_root: Path) -> None:
    for class_name in DISH_CLASSES:
        (out_root / "reviewed" / class_name).mkdir(parents=True, exist_ok=True)


def write_review_instructions(out_root: Path) -> None:
    text = """External dataset review workflow

1. Open reports/review_sheets/*.jpg to inspect each review pool.
2. Keep trustworthy class-specific candidates in review/<pool>/, or copy the best files into reviewed/<class_name>/.
3. Ambiguous pools must be sorted manually before training:
   - thit_kho_or_thit_kho_trung -> reviewed/thit_kho or reviewed/thit_kho_trung
   - canh_chua_unknown -> reviewed/canh_chua_co_ca or reviewed/canh_chua_khong_ca
   - rau_xao_or_canh_rau -> reviewed/rau_xao or reviewed/canh_rau
   - protein_grid_review -> reviewed/thit_kho, reviewed/thit_kho_trung, reviewed/ca_hu_kho, or reviewed/suon_nuong
   - unknown_food_crops -> any reviewed/<class_name>, only if visually certain
4. Do not promote ambiguous pool folders directly into data/classification.
"""
    (out_root / "reports" / "review_instructions.txt").write_text(text, encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Import external dish and Roboflow datasets into review pools.")
    parser.add_argument("--downloads-root", type=Path, default=DOWNLOADS_DIR)
    parser.add_argument("--out", type=Path, default=DOWNLOADS_DIR / "external_staging" / f"external_{timestamp}")
    parser.add_argument("--baseline", type=Path, default=DOWNLOADS_DIR / "merge_batches" / "merge_20260609_111108" / "processed")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--crop-pad", type=float, default=0.04)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    parser.add_argument("--min-blur-score", type=float, default=10.0)
    parser.add_argument("--min-brightness", type=float, default=18.0)
    parser.add_argument("--max-brightness", type=float, default=242.0)
    parser.add_argument("--duplicate-hamming", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        (args.out / "review").mkdir(parents=True, exist_ok=True)
        (args.out / "reports").mkdir(parents=True, exist_ok=True)
        ensure_reviewed_dirs(args.out)

    status_counts, pool_counts, rows = process_candidates(args)
    if not args.dry_run:
        manifest = args.out / "reports" / "external_import_manifest.csv"
        write_manifest(rows, manifest)
        make_contact_sheets(args.out / "review", args.out / "reports" / "review_sheets")
        summary = {
            "out": relative_or_absolute(args.out),
            "baseline": relative_or_absolute(args.baseline),
            "status_counts": dict(status_counts),
            "accepted_by_pool": dict(pool_counts),
        }
        (args.out / "reports" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_review_instructions(args.out)

    print("Output:", args.out)
    print("Baseline:", args.baseline)
    print("Status counts:")
    for key, value in sorted(status_counts.items()):
        print(f"{key}: {value}")
    print("Accepted by pool:")
    for key, value in sorted(pool_counts.items()):
        print(f"{key}: {value}")
    print("Dry run:", args.dry_run)


if __name__ == "__main__":
    main()
