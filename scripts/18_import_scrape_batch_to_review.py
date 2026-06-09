from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import DISH_CLASSES, DOWNLOADS_DIR, PROJECT_ROOT
from canteen_checkout.data_quality import assess_image, hamming_distance_hex, normalize_image, quality_reasons
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


MANIFEST_FIELDS = [
    "pool",
    "suggested_class",
    "needs_review",
    "status",
    "reason",
    "source_dataset",
    "source_path",
    "source_split",
    "label_name",
    "annotation_format",
    "method",
    "crop_box",
    "output_path",
    "width",
    "height",
    "aspect_ratio",
    "brightness",
    "blur_score",
    "phash",
    "sha256",
]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def latest_external_staging() -> Path:
    root = DOWNLOADS_DIR / "external_staging"
    candidates = sorted((p for p in root.glob("external_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No external staging folders found in {root}")
    return candidates[0]


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def unique_output_path(out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.jpg"
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = out_dir / f"{stem}_{idx:03d}.jpg"
        if not candidate.exists():
            return candidate
        idx += 1


def is_duplicate(phash: str, seen: list[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, old) <= threshold for old in seen)


def load_reference_phashes(roots: list[Path]) -> list[str]:
    phashes: list[str] = []
    for root in roots:
        for path in list_images(root):
            _, metrics, _ = assess_image(path)
            if metrics is not None:
                phashes.append(metrics.phash)
    return phashes


def append_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def manifest_row(
    *,
    pool: str,
    suggested_class: str,
    status: str,
    reason: str,
    source_dataset: str,
    source_path: Path,
    output_path: Path | None,
    metrics,
) -> dict[str, str]:
    return {
        "pool": pool,
        "suggested_class": suggested_class,
        "needs_review": "True",
        "status": status,
        "reason": reason,
        "source_dataset": source_dataset,
        "source_path": relative_or_absolute(source_path),
        "source_split": "",
        "label_name": suggested_class,
        "annotation_format": "scrape_batch",
        "method": "direct_image",
        "crop_box": "",
        "output_path": relative_or_absolute(output_path) if output_path else "",
        "width": str(metrics.width) if metrics else "",
        "height": str(metrics.height) if metrics else "",
        "aspect_ratio": f"{metrics.aspect_ratio:.4f}" if metrics else "",
        "brightness": f"{metrics.brightness:.4f}" if metrics else "",
        "blur_score": f"{metrics.blur_score:.4f}" if metrics else "",
        "phash": metrics.phash if metrics else "",
        "sha256": metrics.sha256 if metrics else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a scrape batch into an external staging review pool.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, default=None)
    parser.add_argument("--class-name", choices=DISH_CLASSES, default="rau_xao")
    parser.add_argument("--pool", default="rau_xao_extra_scrape")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--min-size", type=int, default=180)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--min-blur-score", type=float, default=20.0)
    parser.add_argument("--min-brightness", type=float, default=20.0)
    parser.add_argument("--max-brightness", type=float, default=238.0)
    parser.add_argument("--duplicate-hamming", type=int, default=8)
    parser.add_argument("--dedupe-against", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    staging = args.staging or latest_external_staging()
    source_name = args.source_name or args.source.name
    out_dir = staging / "review" / args.pool
    rejected_dir = staging / "rejected" / "scrape_batch" / args.pool
    manifest_path = staging / "reports" / "external_import_manifest.csv"

    reference_roots = list(args.dedupe_against)
    reference_roots.extend([staging / "reviewed", staging / "review" / args.pool])
    seen_phashes = load_reference_phashes(reference_roots)
    rows: list[dict[str, str]] = []
    counts: Counter = Counter()

    for path in list_images(args.source / args.class_name):
        image, metrics, reasons = assess_image(path)
        if metrics is not None:
            reasons.extend(
                quality_reasons(
                    metrics,
                    min_size=args.min_size,
                    max_aspect_ratio=args.max_aspect_ratio,
                    min_blur_score=args.min_blur_score,
                    min_brightness=args.min_brightness,
                    max_brightness=args.max_brightness,
                )
            )
            if not reasons and is_duplicate(metrics.phash, seen_phashes, args.duplicate_hamming):
                reasons.append("duplicate_reference")

        if reasons or image is None or metrics is None:
            counts["rejected"] += 1
            reason = ";".join(reasons) if reasons else "invalid_image"
            output = rejected_dir / path.name
            if not args.dry_run:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
            rows.append(
                manifest_row(
                    pool=args.pool,
                    suggested_class=args.class_name,
                    status="rejected",
                    reason=reason,
                    source_dataset=source_name,
                    source_path=path,
                    output_path=output,
                    metrics=metrics,
                )
            )
            continue

        counts["accepted"] += 1
        seen_phashes.append(metrics.phash)
        output = unique_output_path(out_dir, f"{source_name}_{path.stem}_{metrics.sha256[:8]}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            normalized = normalize_image(image, image_size=args.image_size, mode=args.mode)
            normalized.save(output, format="JPEG", quality=92, optimize=True)
        rows.append(
            manifest_row(
                pool=args.pool,
                suggested_class=args.class_name,
                status="accepted",
                reason="",
                source_dataset=source_name,
                source_path=path,
                output_path=output,
                metrics=metrics,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": relative_or_absolute(args.source),
        "staging": relative_or_absolute(staging),
        "pool": args.pool,
        "class_name": args.class_name,
        "counts": dict(counts),
    }
    if not args.dry_run:
        append_manifest_rows(manifest_path, rows)
        summary_path = staging / "reports" / f"{args.pool}_import_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Dry run:", args.dry_run)


if __name__ == "__main__":
    main()
