from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


EGG_CLASS = "egg"
FISH_CLASS = "fish"
EGG_RELATED_CLASSES = {"thit_kho", "thit_kho_trung"}
FISH_RELATED_CLASSES = {"canh_chua_co_ca", "canh_chua_khong_ca"}


@dataclass(frozen=True)
class ObjectDetection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "xyxy": [round(value, 2) for value in self.xyxy],
        }


@dataclass(frozen=True)
class DetectorEvidence:
    egg_count: int
    fish_count: int
    detections: tuple[ObjectDetection, ...]
    detector_loaded: bool
    detector_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_loaded": self.detector_loaded,
            "detector_path": self.detector_path,
            "egg_count": self.egg_count,
            "fish_count": self.fish_count,
            "detections": [detection.as_dict() for detection in self.detections],
        }


@dataclass(frozen=True)
class FusionResult:
    class_name: str
    confidence: float
    uncertain: bool
    egg_count: int | None
    fish_count: int
    fusion_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "uncertain": self.uncertain,
            "egg_count": self.egg_count,
            "fish_count": self.fish_count,
            "fusion_reason": self.fusion_reason,
        }


def load_yolo_detector(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def empty_evidence(detector_path: Path | None = None, loaded: bool = False) -> DetectorEvidence:
    return DetectorEvidence(
        egg_count=0,
        fish_count=0,
        detections=(),
        detector_loaded=loaded,
        detector_path=str(detector_path) if detector_path else None,
    )


def detect_objects(
    detector,
    image_path: Path,
    *,
    detector_path: Path | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
) -> DetectorEvidence:
    if detector is None:
        return empty_evidence(detector_path, loaded=False)

    # Validate the image early so broken crops are reported by PIL instead of
    # bubbling out as less readable backend errors.
    with Image.open(image_path) as image:
        image.verify()

    results = detector.predict(str(image_path), conf=confidence, iou=iou, verbose=False)
    if not results:
        return empty_evidence(detector_path, loaded=True)

    result = results[0]
    detections: list[ObjectDetection] = []
    names = getattr(result, "names", getattr(detector, "names", {}))
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = _class_name(names, class_id)
            conf = float(box.conf[0].item())
            xyxy_values = tuple(float(value) for value in box.xyxy[0].detach().cpu().tolist())
            detections.append(ObjectDetection(class_name=class_name, confidence=conf, xyxy=xyxy_values))

    return DetectorEvidence(
        egg_count=sum(1 for detection in detections if detection.class_name == EGG_CLASS),
        fish_count=sum(1 for detection in detections if detection.class_name == FISH_CLASS),
        detections=tuple(detections),
        detector_loaded=True,
        detector_path=str(detector_path) if detector_path else None,
    )


def fuse_decision(
    *,
    raw_class_name: str,
    raw_confidence: float,
    uncertain: bool,
    evidence: DetectorEvidence,
) -> FusionResult:
    final_class = raw_class_name
    final_confidence = raw_confidence
    final_uncertain = uncertain
    fusion_reason = "classifier_only"
    final_egg_count: int | None = None
    fish_count = evidence.fish_count

    if raw_class_name in EGG_RELATED_CLASSES:
        if evidence.egg_count >= 1:
            final_class = "thit_kho_trung"
            final_confidence = max(raw_confidence, max((d.confidence for d in evidence.detections if d.class_name == EGG_CLASS), default=0.0))
            final_uncertain = False
            final_egg_count = evidence.egg_count
            fusion_reason = "detector_found_egg"
        elif raw_class_name == "thit_kho_trung":
            final_egg_count = 1
            fusion_reason = "classifier_thit_kho_trung_no_detector_egg"
        else:
            fusion_reason = "classifier_thit_kho_no_detector_egg"

    elif raw_class_name in FISH_RELATED_CLASSES:
        if evidence.fish_count >= 1:
            final_class = "canh_chua_co_ca"
            final_confidence = max(raw_confidence, max((d.confidence for d in evidence.detections if d.class_name == FISH_CLASS), default=0.0))
            final_uncertain = False
            fusion_reason = "detector_found_fish"
        elif raw_class_name == "canh_chua_khong_ca":
            fusion_reason = "classifier_khong_ca_no_detector_fish"
        else:
            fusion_reason = "classifier_co_ca_no_detector_fish"

    if final_class == "thit_kho_trung" and final_egg_count is None:
        final_egg_count = 1

    return FusionResult(
        class_name=final_class,
        confidence=final_confidence,
        uncertain=final_uncertain,
        egg_count=final_egg_count,
        fish_count=fish_count,
        fusion_reason=fusion_reason,
    )
