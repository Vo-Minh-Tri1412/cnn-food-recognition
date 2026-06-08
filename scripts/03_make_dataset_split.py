from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, IMAGE_EXTENSIONS


def list_class_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.exists():
        return []
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a flat labeled folder into train/val/test.")
    parser.add_argument("--source", type=Path, required=True, help="Folder with one subfolder per class.")
    parser.add_argument("--out", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split in ["train", "val", "test"]:
        for class_name in DISH_CLASSES:
            (args.out / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in DISH_CLASSES:
        images = list_class_images(args.source, class_name)
        rng.shuffle(images)
        n = len(images)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        assignments = {
            "test": images[:n_test],
            "val": images[n_test : n_test + n_val],
            "train": images[n_test + n_val :],
        }
        for split, paths in assignments.items():
            for src in paths:
                dst = args.out / split / class_name / src.name
                if dst.exists():
                    continue
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
        print(f"{class_name}: total={n}, train={len(assignments['train'])}, val={len(assignments['val'])}, test={len(assignments['test'])}")


if __name__ == "__main__":
    main()
