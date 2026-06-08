from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw

from canteen_checkout.config import DISH_CLASSES, SCRAPED_CANDIDATES_DIR


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def make_sheet(class_name: str, images: list[Path], out_path: Path, thumb_size: int, cols: int) -> None:
    rows = max(1, (len(images) + cols - 1) // cols)
    cell_h = thumb_size + 34
    sheet = Image.new("RGB", (cols * thumb_size, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        x = (idx % cols) * thumb_size
        y = (idx // cols) * cell_h
        sheet.paste(image, (x + (thumb_size - image.width) // 2, y))
        draw.text((x + 4, y + thumb_size + 3), f"{idx:03d} {class_name}", fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_size + 18), path.name[:28], fill=(80, 80, 80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review contact sheets for scraped image candidates.")
    parser.add_argument("--source", type=Path, default=SCRAPED_CANDIDATES_DIR)
    parser.add_argument("--out", type=Path, default=Path("outputs/reports/scraped_review_sheets"))
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--class-name", choices=DISH_CLASSES, default=None)
    args = parser.parse_args()

    classes = [args.class_name] if args.class_name else DISH_CLASSES
    for class_name in classes:
        images = list_images(args.source / class_name)
        if not images:
            continue
        out_path = args.out / f"{class_name}.jpg"
        make_sheet(class_name, images, out_path, args.thumb_size, args.cols)
        print(f"{class_name}: {len(images)} images -> {out_path}")


if __name__ == "__main__":
    main()
