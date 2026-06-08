from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canteen_checkout.config import CLASSIFICATION_DIR, DEMO_TRAYS_DIR, PROJECT_ROOT, RAW_TEACHER_TRAYS_DIR
from canteen_checkout.io_utils import image_size, list_images


def summarize_folder(root: Path) -> list[dict[str, str | int]]:
    rows = []
    images = list_images(root)
    parent_counts = Counter(str(p.parent.relative_to(root)) for p in images) if root.exists() else Counter()
    for parent, count in sorted(parent_counts.items()):
        rows.append({"folder": parent, "image_count": count})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory images in project folders.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "reports" / "inventory.csv")
    args = parser.parse_args()

    roots = {
        "raw_teacher_trays": RAW_TEACHER_TRAYS_DIR,
        "demo_trays": DEMO_TRAYS_DIR,
        "classification": CLASSIFICATION_DIR,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "folder", "image_count"])
        writer.writeheader()
        for section, root in roots.items():
            rows = summarize_folder(root)
            if not rows:
                writer.writerow({"section": section, "folder": ".", "image_count": 0})
            for row in rows:
                writer.writerow({"section": section, **row})

    print(f"Wrote {args.out}")
    for section, root in roots.items():
        images = list_images(root)
        print(f"{section}: {len(images)} images")
        for image_path in images[:3]:
            print(f"  {image_path.name}: {image_size(image_path)}")


if __name__ == "__main__":
    main()
