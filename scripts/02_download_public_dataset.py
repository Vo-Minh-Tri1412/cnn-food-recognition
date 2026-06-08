from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import kagglehub

from canteen_checkout.config import DOWNLOADS_DIR


DATASETS = {
    "30vnfoods": "quandang/vietnamese-foods",
    "vietfood68": "thomasnguyen6868/vietfood68",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public food datasets with kagglehub.")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    slug = DATASETS[args.dataset]
    print(f"Downloading {slug}. This may require Kaggle login/token depending on your setup.")
    path = kagglehub.dataset_download(slug)
    print(f"Downloaded/cache path: {path}")
    print("Next step: inspect class folders and copy matching dish images into data/classification.")


if __name__ == "__main__":
    main()
