from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import DETECTION_DIR, OUTPUTS_DIR, PROJECT_ROOT


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package YOLO detection dataset for Colab/Kaggle.")
    parser.add_argument("--source", type=Path, default=DETECTION_DIR / "egg_fish")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "cloud" / "egg_fish_yolo.zip")
    parser.add_argument("--manifest", type=Path, default=OUTPUTS_DIR / "cloud" / "egg_fish_yolo.manifest.json")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Dataset source not found: {args.source}")
    data_yaml = args.source / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    files = sorted(path for path in args.source.rglob("*") if path.is_file())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, Path(args.source.name) / path.relative_to(args.source))

    with zipfile.ZipFile(args.output, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt zip entry: {bad_file}")

    split_counts: dict[str, dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        image_dir = args.source / split / "images"
        label_dir = args.source / split / "labels"
        split_counts[split] = {
            "images": len(list(image_dir.glob("*"))) if image_dir.exists() else 0,
            "labels": len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0,
        }

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": relative_or_absolute(args.source),
        "archive": relative_or_absolute(args.output),
        "archive_sha256": sha256_file(args.output),
        "archive_size_bytes": args.output.stat().st_size,
        "files": len(files),
        "split_counts": split_counts,
        "layout_root": args.source.name,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
