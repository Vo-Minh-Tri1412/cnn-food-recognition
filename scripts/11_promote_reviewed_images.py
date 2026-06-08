from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, SCRAPED_CANDIDATES_DIR


def images(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote manually reviewed candidate images into data/classification."
    )
    parser.add_argument("--source", type=Path, default=SCRAPED_CANDIDATES_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--class-name", choices=DISH_CLASSES, default=None)
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    args = parser.parse_args()

    classes = [args.class_name] if args.class_name else DISH_CLASSES
    total = 0
    for class_name in classes:
        source_dir = args.source / class_name
        target_dir = CLASSIFICATION_DIR / args.split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in images(source_dir):
            target = target_dir / f"web_{path.name}"
            if target.exists():
                continue
            if args.move:
                shutil.move(str(path), target)
            else:
                shutil.copy2(path, target)
            total += 1
        print(f"{class_name}: promoted to {target_dir}")
    print(f"Total promoted: {total}")


if __name__ == "__main__":
    main()
