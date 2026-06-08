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


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def archive_path_for(path: Path, archive_root: Path) -> Path:
    rel = path.relative_to(PROJECT_ROOT)
    return archive_root / rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely clean generated project artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually archive/delete files. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup targets. This is also the default.")
    parser.add_argument("--mode", choices=["archive", "delete"], default="archive")
    parser.add_argument("--archive-root", type=Path, default=None)
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("Use either --apply or --dry-run, not both.")
    dry_run = not args.apply

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = args.archive_root or (ARCHIVE_DIR / f"workspace_cleanup_{timestamp}")
    targets = [PROJECT_ROOT / item for item in ARTIFACT_TARGETS if (PROJECT_ROOT / item).exists()]

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
