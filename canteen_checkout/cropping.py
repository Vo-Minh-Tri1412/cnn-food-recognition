from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CropRegion:
    name: str
    x: int
    y: int
    w: int
    h: int
    label: str | None = None


def clamp_region(region: CropRegion, image_width: int, image_height: int) -> CropRegion:
    x = max(0, min(region.x, image_width - 1))
    y = max(0, min(region.y, image_height - 1))
    w = max(1, min(region.w, image_width - x))
    h = max(1, min(region.h, image_height - y))
    return CropRegion(region.name, x, y, w, h, region.label)


def five_compartment_template(image_width: int, image_height: int) -> list[CropRegion]:
    """Approximate regions for common HCMUS metal trays.

    This is a starting point for MVP demos. For accurate crops, create a JSON
    file with manual regions from scripts/cli/01_crop_tray.py --interactive.
    """
    w = image_width
    h = image_height
    if w >= h:
        # Landscape image: split into rough tray cells.
        rel_regions = [
            ("top_left", 0.05, 0.06, 0.30, 0.36),
            ("bottom_left", 0.05, 0.48, 0.30, 0.38),
            ("center", 0.38, 0.18, 0.24, 0.50),
            ("top_right", 0.66, 0.06, 0.28, 0.36),
            ("bottom_right", 0.66, 0.48, 0.28, 0.38),
        ]
    else:
        # Portrait image: common phone photos of one vertical tray.
        rel_regions = [
            ("top_left", 0.05, 0.07, 0.36, 0.25),
            ("middle_left", 0.05, 0.34, 0.36, 0.24),
            ("bottom_left", 0.05, 0.61, 0.36, 0.30),
            ("top_right", 0.43, 0.07, 0.52, 0.43),
            ("bottom_right", 0.43, 0.59, 0.52, 0.32),
        ]
    return [
        CropRegion(name, int(rx * w), int(ry * h), int(rw * w), int(rh * h))
        for name, rx, ry, rw, rh in rel_regions
    ]


def load_regions(path: Path) -> list[CropRegion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = data["regions"] if isinstance(data, dict) else data
    return [
        CropRegion(
            name=item.get("name", f"crop_{idx:02d}"),
            x=int(item["x"]),
            y=int(item["y"]),
            w=int(item["w"]),
            h=int(item["h"]),
            label=item.get("label"),
        )
        for idx, item in enumerate(regions)
    ]


def save_regions(path: Path, regions: list[CropRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "regions": [
            {
                "name": r.name,
                "x": r.x,
                "y": r.y,
                "w": r.w,
                "h": r.h,
                **({"label": r.label} if r.label else {}),
            }
            for r in regions
        ]
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def crop_regions(image_path: Path, regions: list[CropRegion], out_dir: Path) -> list[Path]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for idx, region in enumerate(regions):
        r = clamp_region(region, width, height)
        crop = image[r.y : r.y + r.h, r.x : r.x + r.w]
        safe_name = r.name.replace(" ", "_")
        out_path = out_dir / f"{idx:02d}_{safe_name}.jpg"
        cv2.imwrite(str(out_path), crop)
        outputs.append(out_path)
    return outputs


def select_regions_interactive(image_path: Path) -> list[CropRegion]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    rois = cv2.selectROIs("Select dish regions, press ENTER when done", image, showCrosshair=True)
    cv2.destroyAllWindows()
    regions = []
    for idx, (x, y, w, h) in enumerate(rois):
        if w > 0 and h > 0:
            regions.append(CropRegion(name=f"crop_{idx:02d}", x=int(x), y=int(y), w=int(w), h=int(h)))
    return regions
