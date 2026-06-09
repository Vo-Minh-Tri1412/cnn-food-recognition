from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import ARCHIVE_DIR, PROJECT_ROOT


ARTIFACT_TARGETS = [
    "scripts/__pycache__",
    "canteen_checkout/__pycache__",
    "outputs/smoke_dataset",
    "outputs/smoke_model.pt",
    "outputs/bills",
    "outputs/cropped_dishes",
    "outputs/reports/classification_report.txt",
    "outputs/reports/confusion_matrix.png",
    "outputs/reports/smoke_test_summary.json",
    "outputs/reports/training_history.json",
    "outputs/reports/training_history.png",
    "outputs/reports/teacher_contact_sheet_0_29.jpg",
    "outputs/reports/teacher_contact_sheet_30_55.jpg",
    "outputs/reports/teacher_seed_crops_contact_sheet.jpg",
]

OPTIONAL_GLOBS = {
    "review_sheets": [
        "outputs/reports/scraped_review_sheets",
        "data/downloads/external_staging/external_*/reports/review_sheets",
    ],
    "model_assisted": [
        "data/downloads/external_staging/external_*/model_assisted",
        "outputs/reports/model_assisted_filter/run_*",
    ],
    "scrape_batches": [
        "data/downloads/scrape_batches/*",
    ],
}


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def archive_path_for(path: Path, archive_root: Path) -> Path:
    rel = path.relative_to(PROJECT_ROOT)
    return archive_root / rel


def expand_targets(include_groups: list[str]) -> list[Path]:
    targets = [PROJECT_ROOT / item for item in ARTIFACT_TARGETS if (PROJECT_ROOT / item).exists()]
    for group in include_groups:
        for pattern in OPTIONAL_GLOBS[group]:
            targets.extend(p for p in PROJECT_ROOT.glob(pattern) if p.exists())
    unique: dict[str, Path] = {}
    for path in targets:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda p: p.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely clean generated project artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually archive/delete files. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup targets. This is also the default.")
    parser.add_argument("--mode", choices=["archive", "delete"], default="archive")
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--include-review-sheets", action="store_true", help="Also archive/delete generated contact sheet folders.")
    parser.add_argument("--include-model-assisted", action="store_true", help="Also archive/delete generated model-assisted grouping folders.")
    parser.add_argument("--include-scrape-batches", action="store_true", help="Also archive/delete old scrape batch folders.")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("Use either --apply or --dry-run, not both.")
    dry_run = not args.apply

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = args.archive_root or (ARCHIVE_DIR / f"workspace_cleanup_{timestamp}")
    include_groups = []
    if args.include_review_sheets:
        include_groups.append("review_sheets")
    if args.include_model_assisted:
        include_groups.append("model_assisted")
    if args.include_scrape_batches:
        include_groups.append("scrape_batches")
    targets = expand_targets(include_groups)

    print("Cleanup mode:", args.mode)
    print("Dry run:", dry_run)
    print("Targets:")
    total = 0
    for path in targets:
        size = size_bytes(path)
        total += size
        print(f"- {path.relative_to(PROJECT_ROOT)} ({size / 1024 / 1024:.2f} MB)")
    print(f"Total: {total / 1024 / 1024:.2f} MB")

    if dry_run:
        print("No changes made. Re-run with --apply to perform cleanup.")
        return

    if args.mode == "archive":
        archive_root.mkdir(parents=True, exist_ok=True)
    for path in targets:
        if args.mode == "delete":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            continue
        dst = archive_path_for(path, archive_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(path), str(dst))
    print("Cleanup completed.")
    if args.mode == "archive":
        print(f"Archived to: {archive_root}")


if __name__ == "__main__":
    main()
