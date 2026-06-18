from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .data_quality import assess_image, blur_score


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE_ORDER = ("Khay_thuc_an_4", "Khay_thuc_an_2", "Khay_thuc_an")
OPTIONAL_GRID_SOURCE = "Khay_thuc_an_3"


@dataclass(frozen=True)
class NormalizedBox:
    xc: float
    yc: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_yolo(self) -> str:
        return f"0 {self.xc:.6f} {self.yc:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    split: str
    source_group: str
    image_path: Path
    boxes: tuple[NormalizedBox, ...]
    variant_count: int = 1
    black_border_ratio: float = 0.0
    blur_score: float = 0.0
    sha256: str = ""
    phash: str = ""


def source_group_from_name(filename: str) -> str:
    return filename.split(".rf.", 1)[0]


def canonical_split(value: str) -> str:
    return "valid" if value in {"val", "valid"} else value if value in {"train", "test"} else "train"


def normalized_box_from_xyxy(
    left: float,
    top: float,
    right: float,
    bottom: float,
    image_width: float,
    image_height: float,
    *,
    min_area_ratio: float = 0.005,
) -> NormalizedBox | None:
    if image_width <= 0 or image_height <= 0:
        return None
    left = max(0.0, min(float(image_width), float(left)))
    right = max(0.0, min(float(image_width), float(right)))
    top = max(0.0, min(float(image_height), float(top)))
    bottom = max(0.0, min(float(image_height), float(bottom)))
    if right <= left or bottom <= top:
        return None
    width = (right - left) / image_width
    height = (bottom - top) / image_height
    box = NormalizedBox(
        xc=((left + right) / 2.0) / image_width,
        yc=((top + bottom) / 2.0) / image_height,
        width=width,
        height=height,
    )
    return box if box.area >= min_area_ratio else None


def normalized_box_from_yolo_values(
    values: list[float],
    *,
    min_area_ratio: float = 0.005,
) -> NormalizedBox | None:
    if len(values) == 4:
        xc, yc, width, height = values
        left = xc - width / 2.0
        right = xc + width / 2.0
        top = yc - height / 2.0
        bottom = yc + height / 2.0
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
    else:
        return None
    return normalized_box_from_xyxy(
        left,
        top,
        right,
        bottom,
        1.0,
        1.0,
        min_area_ratio=min_area_ratio,
    )


def parse_yolo_label(path: Path, *, min_area_ratio: float = 0.005) -> tuple[tuple[NormalizedBox, ...], int]:
    if not path.exists():
        return (), 0
    boxes: list[NormalizedBox] = []
    raw_count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        raw_count += 1
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        box = normalized_box_from_yolo_values(values, min_area_ratio=min_area_ratio)
        if box is not None:
            boxes.append(box)
    return tuple(boxes), raw_count


def load_yolo_records(dataset_dir: Path, *, min_area_ratio: float = 0.005) -> tuple[list[SourceRecord], int]:
    records: list[SourceRecord] = []
    raw_boxes = 0
    for image_path in sorted(path for path in dataset_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        parts = image_path.relative_to(dataset_dir).parts
        if len(parts) < 3 or parts[1] != "images":
            continue
        split = canonical_split(parts[0])
        label_path = dataset_dir / parts[0] / "labels" / f"{image_path.stem}.txt"
        boxes, count = parse_yolo_label(label_path, min_area_ratio=min_area_ratio)
        raw_boxes += count
        if not boxes:
            continue
        records.append(
            SourceRecord(
                dataset=dataset_dir.name,
                split=split,
                source_group=source_group_from_name(image_path.name),
                image_path=image_path,
                boxes=boxes,
            )
        )
    return records, raw_boxes


def load_coco_records(dataset_dir: Path, *, min_area_ratio: float = 0.005) -> tuple[list[SourceRecord], int]:
    records: list[SourceRecord] = []
    raw_boxes = 0
    for annotation_path in sorted(dataset_dir.rglob("_annotations.coco.json")):
        split_dir = annotation_path.parent
        split = canonical_split(split_dir.name)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {int(row["id"]): row for row in payload.get("images", [])}
        annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations[int(annotation["image_id"])].append(annotation)
        for image_id, image_row in sorted(images.items(), key=lambda item: str(item[1].get("file_name", ""))):
            image_path = split_dir / str(image_row["file_name"])
            if not image_path.exists():
                continue
            width = float(image_row.get("width") or 0)
            height = float(image_row.get("height") or 0)
            boxes: list[NormalizedBox] = []
            for annotation in annotations.get(image_id, []):
                raw_boxes += 1
                bbox = annotation.get("bbox") or []
                if len(bbox) != 4:
                    continue
                left, top, box_width, box_height = (float(value) for value in bbox)
                box = normalized_box_from_xyxy(
                    left,
                    top,
                    left + box_width,
                    top + box_height,
                    width,
                    height,
                    min_area_ratio=min_area_ratio,
                )
                if box is not None:
                    boxes.append(box)
            if boxes:
                records.append(
                    SourceRecord(
                        dataset=dataset_dir.name,
                        split=split,
                        source_group=source_group_from_name(image_path.name),
                        image_path=image_path,
                        boxes=tuple(boxes),
                    )
                )
    return records, raw_boxes


def load_yolov4_records(dataset_dir: Path, *, min_area_ratio: float = 0.005) -> tuple[list[SourceRecord], int]:
    records: list[SourceRecord] = []
    raw_boxes = 0
    for annotations_path in sorted(dataset_dir.rglob("_annotations.txt")):
        split_dir = annotations_path.parent
        split = canonical_split(split_dir.name)
        for line in annotations_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_path = split_dir / parts[0]
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            boxes: list[NormalizedBox] = []
            for raw_box in parts[1:]:
                values = raw_box.split(",")
                if len(values) != 5:
                    continue
                raw_boxes += 1
                try:
                    left, top, right, bottom = (float(value) for value in values[:4])
                except ValueError:
                    continue
                box = normalized_box_from_xyxy(
                    left,
                    top,
                    right,
                    bottom,
                    width,
                    height,
                    min_area_ratio=min_area_ratio,
                )
                if box is not None:
                    boxes.append(box)
            if boxes:
                records.append(
                    SourceRecord(
                        dataset=dataset_dir.name,
                        split=split,
                        source_group=source_group_from_name(image_path.name),
                        image_path=image_path,
                        boxes=tuple(boxes),
                    )
                )
    return records, raw_boxes


def black_border_ratio(image: Image.Image, band_ratio: float = 0.12, threshold: int = 20) -> float:
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return 1.0
    height, width = gray.shape
    band_y = max(1, int(height * band_ratio))
    band_x = max(1, int(width * band_ratio))
    border = np.zeros_like(gray, dtype=bool)
    border[:band_y, :] = True
    border[-band_y:, :] = True
    border[:, :band_x] = True
    border[:, -band_x:] = True
    values = gray[border]
    return float(np.mean(values <= threshold)) if values.size else 0.0


def enrich_record(record: SourceRecord) -> SourceRecord:
    image, metrics, errors = assess_image(record.image_path)
    if image is None or metrics is None or errors:
        raise ValueError(f"Invalid image: {record.image_path}")
    return replace(
        record,
        black_border_ratio=black_border_ratio(image),
        blur_score=blur_score(image),
        sha256=metrics.sha256,
        phash=metrics.phash,
    )


def select_representatives(records: Iterable[SourceRecord]) -> tuple[list[SourceRecord], list[SourceRecord]]:
    groups: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.dataset, record.source_group)].append(record)
    selected: list[SourceRecord] = []
    rejected: list[SourceRecord] = []
    for key in sorted(groups):
        variants = [enrich_record(record) for record in groups[key]]
        variants.sort(key=lambda row: (row.black_border_ratio, -row.blur_score, row.image_path.as_posix()))
        winner = replace(variants[0], variant_count=len(variants))
        selected.append(winner)
        rejected.extend(replace(row, variant_count=len(variants)) for row in variants[1:])
    return selected, rejected


def audit_source_root(source_root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for dataset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        images = sorted(path for path in dataset_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        groups = Counter(source_group_from_name(path.name) for path in images)
        result[dataset_dir.name] = {
            "images": len(images),
            "source_groups": len(groups),
            "variant_group_sizes": dict(sorted(Counter(groups.values()).items())),
        }
    return result
