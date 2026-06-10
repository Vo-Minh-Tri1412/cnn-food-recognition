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

from canteen_checkout.config import ARCHIVE_DIR, DISH_CLASSES, DOWNLOADS_DIR, PROJECT_ROOT
from canteen_checkout.data_quality import assess_image, hamming_distance_hex, normalize_image
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


MANIFEST_FIELDS = [
    "timestamp",
    "status",
    "reason",
    "source_dataset",
    "source_path",
    "pool",
    "suggested_class",
    "output_path",
    "duplicate_reference",
    "width",
    "height",
    "phash",
    "sha256",
]


TOP_LEVEL_POOLS = {
    "Ca kho to": ("unreviewed_ca_hu_kho_from_ca_kho_to", "ca_hu_kho"),
    "Canh chua": ("unreviewed_canh_chua_unknown_from_canh_chua", "canh_chua_unknown"),
    "Thit kho": ("unreviewed_thit_kho_unknown_from_thit_kho", "thit_kho_unknown"),
}


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def safe_name(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_"}:
            out.append("_")
    return "_".join("".join(out).split("_")) or "unknown"


def latest_external_staging() -> Path:
    root = DOWNLOADS_DIR / "external_staging"
    candidates = sorted((p for p in root.glob("external_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No external staging folder found in {root}")
    return candidates[0]


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def unique_destination(folder: Path, filename: str, *, create_parent: bool = True) -> Path:
    if create_parent:
        folder.mkdir(parents=True, exist_ok=True)
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


def collect_reference_index(roots: list[Path]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    sha_refs: dict[str, str] = {}
    phash_refs: list[tuple[str, str]] = []
    seen_paths = set()
    resolved_roots = [root.resolve() for root in roots]
    report_dirs = sorted({root.parent / "reports" for root in roots})
    for reports in report_dirs:
        if not reports.exists():
            continue
        for report in sorted(reports.glob("*.csv")):
            try:
                with report.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        rel_path = row.get("output_path") or row.get("path") or ""
                        sha256 = row.get("sha256") or ""
                        phash = row.get("phash") or ""
                        if not rel_path or not sha256 or not phash:
                            continue
                        path = PROJECT_ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
                        resolved = path.resolve()
                        if not path.exists() or not any(resolved.is_relative_to(root) for root in resolved_roots):
                            continue
                        rel = relative_or_absolute(path)
                        sha_refs.setdefault(sha256, rel)
                        phash_refs.append((phash, rel))
                        seen_paths.add(resolved)
            except Exception:
                continue
    for root in roots:
        for path in list_images(root):
            if path.resolve() in seen_paths:
                continue
            _, metrics, _ = assess_image(path)
            if metrics is None:
                continue
            rel = relative_or_absolute(path)
            sha_refs.setdefault(metrics.sha256, rel)
            phash_refs.append((metrics.phash, rel))
    return sha_refs, phash_refs


def find_near_duplicate(phash: str, references: list[tuple[str, str]], threshold: int) -> tuple[str, int] | None:
    best_ref = ""
    best_distance = threshold + 1
    for ref_phash, ref_path in references:
        distance = hamming_distance_hex(phash, ref_phash)
        if distance < best_distance:
            best_distance = distance
            best_ref = ref_path
            if distance == 0:
                break
    if best_distance <= threshold:
        return best_ref, best_distance
    return None


def candidate_sources() -> list[Path]:
    sources: list[Path] = []
    for name in TOP_LEVEL_POOLS:
        root = DOWNLOADS_DIR / name
        if root.exists():
            sources.append(root)
    for processed in sorted((DOWNLOADS_DIR / "merge_batches").glob("*/processed")):
        if processed.is_dir():
            sources.append(processed)
    for raw in sorted((DOWNLOADS_DIR / "scrape_batches").glob("*/raw")):
        if raw.is_dir():
            sources.append(raw)
    return sources


def source_dataset_for(path: Path) -> str:
    rel = path.resolve().relative_to(DOWNLOADS_DIR.resolve())
    parts = rel.parts
    if parts[0] == "merge_batches" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "scrape_batches" and len(parts) >= 2:
        return parts[1]
    return parts[0]


def infer_pool(path: Path) -> tuple[str, str]:
    rel = path.resolve().relative_to(DOWNLOADS_DIR.resolve())
    parts = rel.parts
    top = parts[0]
    if top in TOP_LEVEL_POOLS:
        return TOP_LEVEL_POOLS[top]
    if top == "merge_batches" and "processed" in parts:
        idx = parts.index("processed")
        class_name = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
        if class_name not in DISH_CLASSES:
            class_name = safe_name(class_name)
        return f"unreviewed_merge_{class_name}", class_name
    if top == "scrape_batches" and "raw" in parts:
        idx = parts.index("raw")
        class_name = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
        class_name = safe_name(class_name)
        return f"unreviewed_scrape_{class_name}", class_name
    return f"unreviewed_{safe_name(top)}", "unknown"


def iter_candidates(sources: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for source in sources:
        for path in list_images(source):
            lowered = {part.lower() for part in path.parts}
            if "reports" in lowered:
                continue
            candidates.append(path)
    return sorted(candidates)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
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
    status: str,
    reason: str,
    source_path: Path,
    pool: str,
    suggested_class: str,
    output_path: Path | None,
    duplicate_reference: str,
    metrics,
) -> dict[str, str]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "reason": reason,
        "source_dataset": source_dataset_for(source_path),
        "source_path": relative_or_absolute(source_path),
        "pool": pool,
        "suggested_class": suggested_class,
        "output_path": relative_or_absolute(output_path) if output_path else "",
        "duplicate_reference": duplicate_reference,
        "width": str(metrics.width) if metrics else "",
        "height": str(metrics.height) if metrics else "",
        "phash": metrics.phash if metrics else "",
        "sha256": metrics.sha256 if metrics else "",
    }


def archive_tray_sources(timestamp: str, dry_run: bool) -> list[dict[str, str]]:
    archive_root = ARCHIVE_DIR / f"raw_tray_datasets_{timestamp}"
    rows: list[dict[str, str]] = []
    for source in sorted(DOWNLOADS_DIR.glob("Khay_thuc_an*")):
        if not source.is_dir():
            continue
        target = unique_destination(archive_root, source.name, create_parent=not dry_run)
        rows.append({"source": relative_or_absolute(source), "target": relative_or_absolute(target)})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    return rows


def stage_candidates(args: argparse.Namespace, staging: Path, timestamp: str) -> dict[str, object]:
    sources = candidate_sources()
    candidates = iter_candidates(sources)
    reference_roots = [staging / "reviewed", staging / "review"]
    sha_refs, phash_refs = collect_reference_index(reference_roots)
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()

    for path in candidates:
        pool, suggested_class = infer_pool(path)
        image, metrics, reasons = assess_image(path)
        if image is None or metrics is None:
            counts["invalid"] += 1
            rows.append(
                manifest_row(
                    status="invalid",
                    reason=";".join(reasons) or "invalid_image",
                    source_path=path,
                    pool=pool,
                    suggested_class=suggested_class,
                    output_path=None,
                    duplicate_reference="",
                    metrics=metrics,
                )
            )
            continue

        if metrics.sha256 in sha_refs:
            counts["skipped_exact_duplicate"] += 1
            rows.append(
                manifest_row(
                    status="skipped",
                    reason="exact_duplicate",
                    source_path=path,
                    pool=pool,
                    suggested_class=suggested_class,
                    output_path=None,
                    duplicate_reference=sha_refs[metrics.sha256],
                    metrics=metrics,
                )
            )
            continue

        near = find_near_duplicate(metrics.phash, phash_refs, args.phash_threshold)
        if near:
            ref_path, distance = near
            counts["skipped_near_duplicate"] += 1
            rows.append(
                manifest_row(
                    status="skipped",
                    reason=f"near_duplicate_phash_{distance}",
                    source_path=path,
                    pool=pool,
                    suggested_class=suggested_class,
                    output_path=None,
                    duplicate_reference=ref_path,
                    metrics=metrics,
                )
            )
            continue

        out_dir = staging / "review" / pool
        out_name = f"{source_dataset_for(path)}_{path.stem}_{metrics.sha256[:10]}.jpg"
        out_path = unique_destination(out_dir, safe_name(out_name).replace("_jpg", "") + ".jpg", create_parent=not args.dry_run)
        if not args.dry_run:
            normalized = normalize_image(image, image_size=args.image_size, mode=args.mode)
            normalized.save(out_path, format="JPEG", quality=92, optimize=True)
        sha_refs[metrics.sha256] = relative_or_absolute(out_path)
        phash_refs.append((metrics.phash, relative_or_absolute(out_path)))
        counts["accepted"] += 1
        pool_counts[pool] += 1
        rows.append(
            manifest_row(
                status="accepted",
                reason="",
                source_path=path,
                pool=pool,
                suggested_class=suggested_class,
                output_path=out_path,
                duplicate_reference="",
                metrics=metrics,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "staging": relative_or_absolute(staging),
        "sources": [relative_or_absolute(source) for source in sources],
        "candidate_count": len(candidates),
        "counts": dict(counts),
        "accepted_by_pool": dict(sorted(pool_counts.items())),
        "phash_threshold": args.phash_threshold,
    }
    if not args.dry_run:
        report_dir = staging / "reports"
        write_manifest(report_dir / f"unreviewed_stage_{timestamp}.csv", rows)
        (report_dir / f"unreviewed_stage_{timestamp}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage never-reviewed download sources into the Data IDE review queue.")
    parser.add_argument("--staging", type=Path, default=None, help="External staging folder. Defaults to latest external_*.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-archive-trays", action="store_true", help="Do not move top-level Khay_thuc_an* folders.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = args.staging or latest_external_staging()
    if not staging.is_absolute():
        staging = PROJECT_ROOT / staging

    archived = [] if args.no_archive_trays else archive_tray_sources(timestamp, args.dry_run)
    summary = stage_candidates(args, staging, timestamp)
    summary["archived_tray_sources"] = archived
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
