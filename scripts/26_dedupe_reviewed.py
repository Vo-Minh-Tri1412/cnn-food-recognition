from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import DATA_DIR, DISH_CLASSES, PROJECT_ROOT, REVIEWED_DIR
from canteen_checkout.data_quality import assess_image, hamming_distance_hex
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


REPORT_FIELDS = [
    "status",
    "reason",
    "distance",
    "class_name",
    "path",
    "kept_path",
    "target_path",
    "sha256",
    "phash",
]


@dataclass(frozen=True)
class ReviewedItem:
    class_name: str
    path: Path
    sha256: str
    phash: str


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def class_for_path(root: Path, path: Path) -> str | None:
    rel = path.resolve().relative_to(root.resolve())
    if not rel.parts:
        return None
    class_name = rel.parts[0]
    return class_name if class_name in DISH_CLASSES else None


def load_items(root: Path) -> tuple[list[ReviewedItem], list[dict[str, str]]]:
    items: list[ReviewedItem] = []
    rows: list[dict[str, str]] = []
    for path in list_images(root):
        class_name = class_for_path(root, path)
        if class_name is None:
            continue
        _, metrics, reasons = assess_image(path)
        if metrics is None:
            rows.append(
                {
                    "status": "invalid",
                    "reason": ";".join(reasons) or "invalid_image",
                    "distance": "",
                    "class_name": class_name,
                    "path": relative_or_absolute(path),
                    "kept_path": "",
                    "target_path": "",
                    "sha256": "",
                    "phash": "",
                }
            )
            continue
        items.append(ReviewedItem(class_name=class_name, path=path, sha256=metrics.sha256, phash=metrics.phash))
    return items, rows


def safe_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "unknown"


def unique_destination(folder: Path, filename: str) -> Path:
    target = folder / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    idx = 1
    while True:
        candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def quarantine_target(item: ReviewedItem, reason: str, timestamp: str) -> Path:
    if reason == "duplicate_same_class":
        folder = DATA_DIR / "quarantine" / "duplicates_same_class" / item.class_name
    elif reason == "exact_cross_class_conflict":
        folder = DATA_DIR / "quarantine" / "label_conflicts" / f"exact_{timestamp}" / item.class_name
    elif reason == "near_cross_class_conflict":
        folder = DATA_DIR / "quarantine" / "label_conflicts" / f"near_{timestamp}" / item.class_name
    else:
        folder = DATA_DIR / "quarantine" / reason / item.class_name
    return unique_destination(folder, item.path.name)


def add_move(
    moves: dict[Path, tuple[ReviewedItem, str, str, str, Path]],
    item: ReviewedItem,
    reason: str,
    kept_path: str,
    distance: str,
    timestamp: str,
) -> None:
    if item.path in moves:
        return
    moves[item.path] = (item, reason, kept_path, distance, quarantine_target(item, reason, timestamp))


def plan_moves(items: list[ReviewedItem], same_threshold: int, cross_threshold: int, timestamp: str) -> dict[Path, tuple[ReviewedItem, str, str, str, Path]]:
    moves: dict[Path, tuple[ReviewedItem, str, str, str, Path]] = {}
    by_sha: dict[str, list[ReviewedItem]] = defaultdict(list)
    for item in items:
        by_sha[item.sha256].append(item)

    for group in by_sha.values():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda item: item.path.as_posix())
        classes = {item.class_name for item in group}
        if len(classes) > 1:
            for item in group:
                add_move(moves, item, "exact_cross_class_conflict", "", "0", timestamp)
            continue
        kept = group[0]
        for item in group[1:]:
            add_move(moves, item, "duplicate_same_class", relative_or_absolute(kept.path), "0", timestamp)

    by_class: dict[str, list[ReviewedItem]] = defaultdict(list)
    for item in sorted(items, key=lambda item: (item.class_name, item.sha256, item.path.as_posix())):
        if item.path in moves:
            continue
        by_class[item.class_name].append(item)
    for class_name, class_items in by_class.items():
        kept: list[ReviewedItem] = []
        for item in class_items:
            duplicate_of: tuple[ReviewedItem, int] | None = None
            for prior in kept:
                distance = hamming_distance_hex(item.phash, prior.phash)
                if distance <= same_threshold:
                    duplicate_of = (prior, distance)
                    break
            if duplicate_of is None:
                kept.append(item)
            else:
                prior, distance = duplicate_of
                add_move(moves, item, "duplicate_same_class", relative_or_absolute(prior.path), str(distance), timestamp)

    remaining = [item for item in items if item.path not in moves]
    for left, right in combinations(remaining, 2):
        if left.class_name == right.class_name:
            continue
        distance = hamming_distance_hex(left.phash, right.phash)
        if distance <= cross_threshold:
            add_move(moves, left, "near_cross_class_conflict", relative_or_absolute(right.path), str(distance), timestamp)
            add_move(moves, right, "near_cross_class_conflict", relative_or_absolute(left.path), str(distance), timestamp)
    return moves


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe data/reviewed and quarantine duplicates/conflicts.")
    parser.add_argument("--root", type=Path, default=REVIEWED_DIR)
    parser.add_argument("--same-class-threshold", type=int, default=8)
    parser.add_argument("--cross-class-threshold", type=int, default=4)
    parser.add_argument("--apply", action="store_true", help="Move duplicate/conflict images into data/quarantine.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report or PROJECT_ROOT / "outputs" / "reports" / f"reviewed_dedupe_{timestamp}.csv"
    summary_path = report_path.with_suffix(".summary.json")

    items, rows = load_items(root)
    moves = plan_moves(items, args.same_class_threshold, args.cross_class_threshold, timestamp)
    for _path, (item, reason, kept_path, distance, target) in sorted(moves.items(), key=lambda pair: pair[0].as_posix()):
        rows.append(
            {
                "status": "quarantined" if args.apply else "would_quarantine",
                "reason": reason,
                "distance": distance,
                "class_name": item.class_name,
                "path": relative_or_absolute(item.path),
                "kept_path": kept_path,
                "target_path": relative_or_absolute(target),
                "sha256": item.sha256,
                "phash": item.phash,
            }
        )
        if args.apply and item.path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(target))

    write_report(report_path, rows)
    counts = Counter(row["reason"] for row in rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": relative_or_absolute(root),
        "dry_run": not args.apply,
        "images_scanned": len(items),
        "rows": len(rows),
        "counts": dict(counts),
        "same_class_threshold": args.same_class_threshold,
        "cross_class_threshold": args.cross_class_threshold,
        "report": relative_or_absolute(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary"] = relative_or_absolute(summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
