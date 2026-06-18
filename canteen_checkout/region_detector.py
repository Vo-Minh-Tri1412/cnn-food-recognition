from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .cropping import CropRegion


FOOD_REGION_CLASS = "food_region"


@dataclass(frozen=True)
class RegionCandidate:
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RegionDetectionResult:
    regions: tuple[CropRegion, ...]
    detector_loaded: bool
    detector_path: str | None
    raw_detection_count: int
    fallback_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_loaded": self.detector_loaded,
            "detector_path": self.detector_path,
            "raw_detection_count": self.raw_detection_count,
            "fallback_reason": self.fallback_reason,
            "regions": [
                {
                    "name": region.name,
                    "x": region.x,
                    "y": region.y,
                    "w": region.w,
                    "h": region.h,
                    "label": region.label or "",
                    "source": region.source,
                    "confidence": round(region.confidence, 4) if region.confidence is not None else None,
                }
                for region in self.regions
            ],
        }


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def regions_from_candidates(
    candidates: list[RegionCandidate],
    *,
    image_width: int,
    image_height: int,
    min_area_ratio: float = 0.005,
    padding_ratio: float = 0.05,
    max_regions: int = 8,
) -> tuple[tuple[CropRegion, ...], str]:
    valid: list[RegionCandidate] = []
    image_area = max(1, image_width * image_height)
    for candidate in candidates:
        left, top, right, bottom = candidate.xyxy
        left = max(0.0, min(float(image_width), left))
        right = max(0.0, min(float(image_width), right))
        top = max(0.0, min(float(image_height), top))
        bottom = max(0.0, min(float(image_height), bottom))
        if right <= left or bottom <= top:
            continue
        if ((right - left) * (bottom - top)) / image_area < min_area_ratio:
            continue
        valid.append(RegionCandidate(candidate.confidence, (left, top, right, bottom)))

    if not valid:
        return (), "no_regions"
    if len(valid) > max_regions:
        return (), "too_many_regions"

    valid.sort(key=lambda row: (((row.xyxy[1] + row.xyxy[3]) / 2.0), ((row.xyxy[0] + row.xyxy[2]) / 2.0)))
    regions: list[CropRegion] = []
    for index, candidate in enumerate(valid, 1):
        left, top, right, bottom = candidate.xyxy
        pad_x = (right - left) * padding_ratio
        pad_y = (bottom - top) * padding_ratio
        left = max(0, int(round(left - pad_x)))
        top = max(0, int(round(top - pad_y)))
        right = min(image_width, int(round(right + pad_x)))
        bottom = min(image_height, int(round(bottom + pad_y)))
        regions.append(
            CropRegion(
                name=f"auto_{index:02d}",
                x=left,
                y=top,
                w=max(1, right - left),
                h=max(1, bottom - top),
                source="auto_detector",
                confidence=candidate.confidence,
            )
        )
    return tuple(regions), ""


def detect_food_regions(
    detector,
    image_path: Path,
    *,
    detector_path: Path | None = None,
    confidence: float = 0.35,
    iou: float = 0.5,
    min_area_ratio: float = 0.005,
    padding_ratio: float = 0.05,
    max_regions: int = 8,
) -> RegionDetectionResult:
    path_value = str(detector_path) if detector_path else None
    if detector is None:
        return RegionDetectionResult((), False, path_value, 0, "model_unavailable")

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    results = detector.predict(
        str(image_path),
        conf=confidence,
        iou=iou,
        max_det=max_regions + 12,
        verbose=False,
    )
    if not results:
        return RegionDetectionResult((), True, path_value, 0, "no_regions")

    result = results[0]
    names = getattr(result, "names", getattr(detector, "names", {}))
    boxes = getattr(result, "boxes", None)
    candidates: list[RegionCandidate] = []
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = _class_name(names, class_id)
            if class_name not in {FOOD_REGION_CLASS, "0"} and class_id != 0:
                continue
            candidates.append(
                RegionCandidate(
                    confidence=float(box.conf[0].item()),
                    xyxy=tuple(float(value) for value in box.xyxy[0].detach().cpu().tolist()),
                )
            )

    regions, fallback_reason = regions_from_candidates(
        candidates,
        image_width=image_width,
        image_height=image_height,
        min_area_ratio=min_area_ratio,
        padding_ratio=padding_ratio,
        max_regions=max_regions,
    )
    return RegionDetectionResult(regions, True, path_value, len(candidates), fallback_reason)
