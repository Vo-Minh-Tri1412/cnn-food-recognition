from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import (
    DISH_CLASSES,
    PROCESSED_CANDIDATES_DIR,
    REJECTED_CANDIDATES_DIR,
    REPORTS_DIR,
    SCRAPED_CANDIDATES_DIR,
)
from canteen_checkout.data_quality import (
    assess_image,
    hamming_distance_hex,
    normalize_image,
    quality_reasons,
)
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def unique_output_path(out_dir: Path, stem: str, suffix: str = ".jpg") -> Path:
    path = out_dir / f"{stem}{suffix}"
    if not path.exists():
        return path
    idx = 1
    while True:
        path = out_dir / f"{stem}_{idx:03d}{suffix}"
        if not path.exists():
            return path
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess scraped image candidates and flag obvious trash.")
    parser.add_argument("--source", type=Path, default=SCRAPED_CANDIDATES_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_CANDIDATES_DIR)
    parser.add_argument("--rejected", type=Path, default=REJECTED_CANDIDATES_DIR)
    parser.add_argument("--class-name", choices=DISH_CLASSES, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--min-size", type=int, default=180)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--min-blur-score", type=float, default=20.0)
    parser.add_argument("--min-brightness", type=float, default=20.0)
    parser.add_argument("--max-brightness", type=float, default=238.0)
    parser.add_argument("--duplicate-hamming", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--move-rejected", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "data_quality_report.csv")
    args = parser.parse_args()

    classes = [args.class_name] if args.class_name else DISH_CLASSES
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    accepted = 0
    rejected = 0

    for class_name in classes:
        seen_hashes: list[str] = []
        for path in list_images(args.source / class_name):
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
                if not reasons:
                    if any(hamming_distance_hex(metrics.phash, old) <= args.duplicate_hamming for old in seen_hashes):
                        reasons.append("duplicate")
                    else:
                        seen_hashes.append(metrics.phash)

            status = "accepted" if not reasons else "rejected"
            reason_text = ";".join(reasons)
            output_path = ""

            if status == "accepted" and image is not None and metrics is not None:
                out_dir = args.out / class_name
                output = unique_output_path(out_dir, f"{path.stem}_{metrics.sha256[:8]}")
                output_path = str(output)
                if not args.dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    normalized = normalize_image(image, image_size=args.image_size, mode=args.mode)
                    normalized.save(output, format="JPEG", quality=92, optimize=True)
                accepted += 1
            else:
                primary_reason = reasons[0] if reasons else "unknown"
                rejected_dir = args.rejected / primary_reason / class_name
                target = rejected_dir / path.name
                output_path = str(target)
                if not args.dry_run:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    if args.move_rejected:
                        shutil.move(str(path), target)
                    else:
                        shutil.copy2(path, target)
                rejected += 1

            rows.append(
                {
                    "class_name": class_name,
                    "file_path": str(path),
                    "status": status,
                    "reasons": reason_text,
                    "width": metrics.width if metrics else "",
                    "height": metrics.height if metrics else "",
                    "aspect_ratio": round(metrics.aspect_ratio, 4) if metrics else "",
                    "brightness": round(metrics.brightness, 4) if metrics else "",
                    "blur_score": round(metrics.blur_score, 4) if metrics else "",
                    "phash": metrics.phash if metrics else "",
                    "sha256": metrics.sha256 if metrics else "",
                    "output_path": output_path,
                }
            )

    if not args.dry_run:
        with args.report.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["class_name"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print("Dry run:", args.dry_run)
    if not args.dry_run:
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
