from __future__ import annotations

from pathlib import Path


NOTES = """# Dataset download notes

Recommended public sources:

1. 30VNFoods
   - Kaggle slug: quandang/vietnamese-foods
   - GitHub: https://github.com/ds4v/30VNFoods
   - Best use: classification images for Vietnamese dishes.

2. VietFood68
   - Kaggle URL: https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68
   - Best use: broader Vietnamese-food dataset and possible bounding-box work.

3. Roboflow Vietnamese Food Detection
   - URL: https://universe.roboflow.com/nhh/vietnamese-food/dataset/1
   - Best use: object detection experiments.

Expected local layout after downloading/filtering:

data/classification/
  train/
    com_trang/
    dau_hu_sot_ca/
    ...
  val/
  test/

Use scripts/04_crop_tray.py to create labeled crops from teacher tray images
when public datasets do not match the canteen dishes closely enough.
"""


def main() -> None:
    out = Path("data/downloads/README_DATASETS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(NOTES, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
