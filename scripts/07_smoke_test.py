from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from canteen_checkout.config import DISH_CLASSES, PROJECT_ROOT


def make_synthetic_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    colors = [
        (240, 240, 240),
        (210, 80, 60),
        (120, 70, 30),
        (160, 90, 40),
        (130, 80, 50),
        (230, 160, 60),
        (190, 220, 160),
        (120, 60, 50),
        (110, 170, 80),
        (70, 160, 70),
        (230, 210, 80),
    ]
    for split in ["train", "val", "test"]:
        for idx, class_name in enumerate(DISH_CLASSES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(2 if split == "train" else 1):
                image = Image.new("RGB", (96, 96), colors[idx])
                draw = ImageDraw.Draw(image)
                draw.rectangle((10 + sample_idx * 5, 10, 70, 70), outline=(0, 0, 0), width=3)
                draw.text((8, 74), str(idx), fill=(0, 0, 0))
                image.save(class_dir / f"{class_name}_{sample_idx}.jpg")


def run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def main() -> None:
    python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    smoke_data = PROJECT_ROOT / "outputs" / "smoke_dataset"
    smoke_model = PROJECT_ROOT / "outputs" / "smoke_model.pt"
    make_synthetic_dataset(smoke_data)
    run(
        [
            str(python),
            "scripts/05_train_classifier.py",
            "--data",
            str(smoke_data),
            "--model-out",
            str(smoke_model),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "96",
            "--no-pretrained",
        ]
    )

    demo_images = sorted((PROJECT_ROOT / "data" / "demo_trays").glob("*.*"))
    if demo_images:
        run(
            [
                str(python),
                "scripts/06_demo_checkout.py",
                "--image",
                str(demo_images[0]),
                "--model",
                str(smoke_model),
                "--threshold",
                "0.0",
            ]
        )
    summary = {
        "smoke_dataset": str(smoke_data),
        "smoke_model": str(smoke_model),
        "demo_image_used": str(demo_images[0]) if demo_images else None,
    }
    out = PROJECT_ROOT / "outputs" / "reports" / "smoke_test_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
