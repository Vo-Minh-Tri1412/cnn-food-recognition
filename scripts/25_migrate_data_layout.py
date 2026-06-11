from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import (
    ARCHIVE_DIR,
    DATA_DIR,
    DEMO_TRAYS_DIR,
    EXTRAS_DIR,
    PROJECT_ROOT,
    REVIEW_INBOX_DIR,
    REVIEWED_DIR,
)
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


def quarantine_dir() -> Path:
    return DATA_DIR / "quarantine"


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.name
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def move_path(source: Path, target: Path, *, dry_run: bool, actions: list[dict[str, object]]) -> None:
    if not source.exists():
        return
    target = unique_destination(target)
    actions.append(
        {
            "action": "move",
            "source": relative_or_absolute(source),
            "target": relative_or_absolute(target),
            "images": image_count(source),
        }
    )
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def merge_directory_contents(source: Path, target: Path, *, dry_run: bool, actions: list[dict[str, object]]) -> None:
    if not source.exists():
        return
    for child in sorted(source.iterdir()):
        move_path(child, target / child.name, dry_run=dry_run, actions=actions)


def latest_external_staging() -> Path | None:
    root = DATA_DIR / "downloads" / "external_staging"
    candidates = sorted((p for p in root.glob("external_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def ensure_standard_dirs(*, dry_run: bool, actions: list[dict[str, object]]) -> None:
    for path in [REVIEW_INBOX_DIR, REVIEWED_DIR, EXTRAS_DIR, quarantine_dir(), DEMO_TRAYS_DIR, ARCHIVE_DIR]:
        actions.append({"action": "ensure_dir", "target": relative_or_absolute(path), "images": image_count(path)})
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def migrate(args: argparse.Namespace) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = ARCHIVE_DIR / f"legacy_{timestamp}"
    actions: list[dict[str, object]] = []
    before = snapshot_counts()

    ensure_standard_dirs(dry_run=args.dry_run, actions=actions)

    latest = latest_external_staging()
    if latest is not None:
        merge_directory_contents(latest / "review", REVIEW_INBOX_DIR, dry_run=args.dry_run, actions=actions)
        merge_directory_contents(latest / "reviewed", REVIEWED_DIR, dry_run=args.dry_run, actions=actions)
        merge_directory_contents(latest / "reviewed_extra", EXTRAS_DIR, dry_run=args.dry_run, actions=actions)
        merge_directory_contents(latest / "manual_rejected", quarantine_dir() / "manual_rejected", dry_run=args.dry_run, actions=actions)
        merge_directory_contents(latest / "rejected", quarantine_dir() / "rejected", dry_run=args.dry_run, actions=actions)
        for legacy_name in ["manual_skipped", "model_assisted", "reports"]:
            move_path(latest / legacy_name, archive_root / "external_staging" / latest.name / legacy_name, dry_run=args.dry_run, actions=actions)

    if (DATA_DIR / "demo_trays").exists() and (DATA_DIR / "demo_trays").resolve() != DEMO_TRAYS_DIR.resolve():
        merge_directory_contents(DATA_DIR / "demo_trays", DEMO_TRAYS_DIR, dry_run=args.dry_run, actions=actions)
        move_path(DATA_DIR / "demo_trays", archive_root / "empty_demo_trays", dry_run=args.dry_run, actions=actions)

    for legacy_name in ["downloads", "scraped_candidates", "processed_candidates", "rejected_candidates", "temp_teacher_crops", "raw_teacher_trays"]:
        source = DATA_DIR / legacy_name
        if source.exists():
            move_path(source, archive_root / legacy_name, dry_run=args.dry_run, actions=actions)

    after = snapshot_counts() if not args.dry_run else before
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "archive_root": relative_or_absolute(archive_root),
        "before": before,
        "after": after,
        "actions": actions,
    }
    if not args.dry_run:
        report = PROJECT_ROOT / "outputs" / "reports" / f"data_layout_migration_{timestamp}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["report"] = relative_or_absolute(report)
    return summary


def snapshot_counts() -> dict[str, int]:
    paths = {
        "inbox_review": REVIEW_INBOX_DIR,
        "reviewed": REVIEWED_DIR,
        "extras": EXTRAS_DIR,
        "classification": DATA_DIR / "classification",
        "quarantine": quarantine_dir(),
        "demo": DEMO_TRAYS_DIR,
        "downloads": DATA_DIR / "downloads",
        "legacy_external_review": DATA_DIR / "downloads" / "external_staging" / "external_20260609_115250" / "review",
        "legacy_external_reviewed": DATA_DIR / "downloads" / "external_staging" / "external_20260609_115250" / "reviewed",
    }
    return {key: image_count(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy data folders into the simplified data layout.")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Without this, only prints a dry-run summary.")
    args = parser.parse_args()
    args.dry_run = not args.apply
    print(json.dumps(migrate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
