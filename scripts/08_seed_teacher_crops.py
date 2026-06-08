from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, RAW_TEACHER_TRAYS_DIR
from canteen_checkout.cropping import crop_regions, five_compartment_template
from canteen_checkout.io_utils import list_images


# Curated from the teacher-provided tray contact sheet.
# Regions use the current five_compartment_template names:
# portrait: top_left, middle_left, bottom_left, top_right, bottom_right
# landscape: top_left, bottom_left, center, top_right, bottom_right
CURATED_LABELS: dict[int, dict[str, str]] = {
    2: {
        "top_left": "thit_kho",
        "middle_left": "dau_hu_sot_ca",
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    3: {
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    4: {
        "top_left": "rau_xao",
        "middle_left": "trung_chien",
        "bottom_left": "thit_kho",
        "top_right": "canh_chua_co_ca",
        "bottom_right": "com_trang",
    },
    5: {
        "top_left": "rau_xao",
        "middle_left": "dau_hu_sot_ca",
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    8: {
        "top_left": "rau_xao",
        "middle_left": "thit_kho",
        "top_right": "canh_chua_co_ca",
        "bottom_right": "com_trang",
    },
    17: {
        "top_left": "rau_xao",
        "middle_left": "thit_kho",
        "bottom_left": "ca_hu_kho",
        "top_right": "canh_chua_co_ca",
        "bottom_right": "com_trang",
    },
    27: {
        "top_left": "thit_kho",
        "middle_left": "ca_hu_kho",
        "bottom_left": "rau_xao",
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    28: {
        "top_left": "thit_kho",
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    31: {
        "top_left": "suon_nuong",
    },
    35: {
        "top_left": "thit_kho",
        "top_right": "canh_rau",
        "bottom_right": "com_trang",
    },
    36: {
        "top_left": "thit_kho_trung",
        "bottom_left": "canh_rau",
        "top_right": "canh_chua_khong_ca",
        "bottom_right": "com_trang",
    },
    37: {
        "top_left": "thit_kho",
        "middle_left": "thit_kho",
        "bottom_left": "rau_xao",
        "bottom_right": "com_trang",
    },
    39: {
        "bottom_left": "ca_hu_kho",
        "top_right": "canh_rau",
        "bottom_right": "com_trang",
    },
    41: {
        "top_left": "thit_kho",
        "middle_left": "ca_hu_kho",
        "bottom_left": "rau_xao",
        "top_right": "canh_chua_khong_ca",
    },
    55: {
        "top_left": "com_trang",
        "bottom_left": "canh_rau",
        "bottom_right": "thit_kho",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create initial labeled crops from teacher tray photos.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    if args.clear_existing:
        for class_name in DISH_CLASSES:
            target_dir = CLASSIFICATION_DIR / args.split / class_name
            if target_dir.exists():
                for path in target_dir.glob("teacher_seed_*.jpg"):
                    path.unlink()

    images = list_images(RAW_TEACHER_TRAYS_DIR)
    total = 0
    for image_index, label_map in CURATED_LABELS.items():
        image_path = images[image_index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        regions = five_compartment_template(width, height)
        selected = []
        for region in regions:
            label = label_map.get(region.name)
            if not label:
                continue
            if label not in DISH_CLASSES:
                raise ValueError(f"Unknown label: {label}")
            selected.append(type(region)(region.name, region.x, region.y, region.w, region.h, label))

        temp_dir = CLASSIFICATION_DIR / "_teacher_seed_tmp" / image_path.stem
        crop_paths = crop_regions(image_path, selected, temp_dir)
        for crop_path, region in zip(crop_paths, selected):
            target_dir = CLASSIFICATION_DIR / args.split / region.label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"teacher_seed_{image_index:02d}_{region.name}_{image_path.stem}.jpg"
            shutil.copy2(crop_path, target)
            total += 1

    shutil.rmtree(CLASSIFICATION_DIR / "_teacher_seed_tmp", ignore_errors=True)
    print(f"Created {total} labeled teacher crops in {CLASSIFICATION_DIR / args.split}")


if __name__ == "__main__":
    main()
