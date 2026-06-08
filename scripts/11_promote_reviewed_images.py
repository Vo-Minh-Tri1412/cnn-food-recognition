from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import (
    CLASSIFICATION_DIR,
    DISH_CLASSES,
    IMAGE_EXTENSIONS,
    PROCESSED_CANDIDATES_DIR,
    REPORTS_DIR,
    SCRAPED_CANDIDATES_DIR,
)
from canteen_checkout.data_quality import assess_image, hamming_distance_hex


def images(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def choose_default_source() -> Path:
    if any(PROCESSED_CANDIDATES_DIR.glob("*/*")):
        return PROCESSED_CANDIDATES_DIR
    return SCRAPED_CANDIDATES_DIR


def is_near_duplicate(phash: str, seen: list[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, previous) <= threshold for previous in seen)


def dedupe_paths(paths: list[Path], threshold: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    kept: list[tuple[Path, str]] = []
    rejected: list[tuple[Path, str]] = []
    seen: list[str] = []
    for path in paths:
        _, metrics, reasons = assess_image(path)
        if metrics is None:
            rejected.append((path, ";".join(reasons) or "invalid_image"))
            continue
        if is_near_duplicate(metrics.phash, seen, threshold):
            rejected.append((path, "near_duplicate"))
            continue
        seen.append(metrics.phash)
        kept.append((path, metrics.phash))
    return kept, rejected


def split_paths(
    paths: list[tuple[Path, str]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")

    shuffled = paths[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    val_count = int(round(n * val_ratio / total_ratio))
    test_count = int(round(n * test_ratio / total_ratio))
    if n >= 3:
        if val_ratio > 0:
            val_count = max(1, val_count)
        if test_ratio > 0:
            test_count = max(1, test_count)
    while val_count + test_count >= n and (val_count > 0 or test_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    train_count = n - val_count - test_count
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def copy_or_move(path: Path, target: Path, *, move: bool, dry_run: bool) -> bool:
    if target.exists():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(path), target)
    else:
        shutil.copy2(path, target)
    return True


def write_report(rows: list[dict[str, str]], path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class_name", "source_path", "target_split", "target_path", "status", "reason", "phash"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote manually reviewed candidate images into data/classification."
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--split", choices=["auto", "train", "val", "test"], default="auto")
    parser.add_argument("--class-name", choices=DISH_CLASSES, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe-threshold", type=int, default=4)
    parser.add_argument("--prefix", default="web")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "promote_reviewed_images_report.csv")
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source or choose_default_source()
    print(f"Source root: {source_root}")
    print(f"Target root: {CLASSIFICATION_DIR}")
    if args.dry_run:
        print("Dry run: no files will be copied or moved.")

    classes = [args.class_name] if args.class_name else DISH_CLASSES
    total = 0
    report_rows: list[dict[str, str]] = []
    for class_name in classes:
        source_dir = source_root / class_name
        source_images = images(source_dir)
        kept, rejected = dedupe_paths(source_images, args.dedupe_threshold)
        for path, reason in rejected:
            report_rows.append(
                {
                    "class_name": class_name,
                    "source_path": str(path),
                    "target_split": "",
                    "target_path": "",
                    "status": "skipped",
                    "reason": reason,
                    "phash": "",
                }
            )

        if args.split == "auto":
            split_map = split_paths(
                kept,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        else:
            split_map = {"train": [], "val": [], "test": []}
            split_map[args.split] = kept

        class_total = 0
        for split_name, split_items in split_map.items():
            target_dir = CLASSIFICATION_DIR / split_name / class_name
            for path, phash in split_items:
                target = target_dir / f"{args.prefix}_{path.stem}{path.suffix.lower()}"
                copied = copy_or_move(path, target, move=args.move, dry_run=args.dry_run)
                status = "promoted" if copied else "skipped"
                reason = "" if copied else "target_exists"
                if copied:
                    class_total += 1
                    total += 1
                report_rows.append(
                    {
                        "class_name": class_name,
                        "source_path": str(path),
                        "target_split": split_name,
                        "target_path": str(target),
                        "status": status,
                        "reason": reason,
                        "phash": phash,
                    }
                )
        print(f"{class_name}: {class_total} promoted, {len(rejected)} skipped by quality/dedupe")
    write_report(report_rows, args.report, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Report: {args.report}")
    print(f"Total promoted: {total}")


if __name__ == "__main__":
    main()
