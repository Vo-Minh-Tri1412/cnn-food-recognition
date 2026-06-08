from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canteen_checkout.config import CLASS_NAMES_JSON, PROJECT_ROOT
from canteen_checkout.io_utils import (
    copy_teacher_images,
    ensure_project_dirs,
    save_class_names,
    validate_prices_and_classes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare folders and class metadata.")
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=PROJECT_ROOT / "Khay_com",
        help="Folder containing teacher-provided tray images.",
    )
    parser.add_argument("--demo-limit", type=int, default=6)
    args = parser.parse_args()

    ensure_project_dirs()
    save_class_names(CLASS_NAMES_JSON)
    issues = validate_prices_and_classes()
    copied_raw = copied_demo = 0
    if args.teacher_dir.exists():
        copied_raw, copied_demo = copy_teacher_images(args.teacher_dir, args.demo_limit)

    print("Project prepared")
    print(f"class_names: {CLASS_NAMES_JSON}")
    print(f"teacher_source_exists: {args.teacher_dir.exists()}")
    print(f"copied_raw_images: {copied_raw}")
    print(f"copied_demo_images: {copied_demo}")
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("prices/classes: OK")


if __name__ == "__main__":
    main()
