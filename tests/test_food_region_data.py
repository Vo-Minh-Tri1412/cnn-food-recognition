from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from canteen_checkout.food_region_data import (
    NormalizedBox,
    SourceRecord,
    load_coco_records,
    normalized_box_from_yolo_values,
    select_representatives,
    source_group_from_name,
)


class FoodRegionDataTests(unittest.TestCase):
    def test_polygon_is_collapsed_to_clipped_box(self) -> None:
        box = normalized_box_from_yolo_values([-0.2, 0.1, 0.8, 0.1, 0.8, 0.9, -0.2, 0.9])
        self.assertIsNotNone(box)
        assert box is not None
        self.assertAlmostEqual(box.xc, 0.4)
        self.assertAlmostEqual(box.yc, 0.5)
        self.assertAlmostEqual(box.width, 0.8)
        self.assertAlmostEqual(box.height, 0.8)

    def test_tiny_box_is_rejected(self) -> None:
        self.assertIsNone(normalized_box_from_yolo_values([0.5, 0.5, 0.04, 0.04]))

    def test_source_group_strips_roboflow_hash(self) -> None:
        self.assertEqual(source_group_from_name("IMG_0800_jpg.rf.abc123.jpg"), "IMG_0800_jpg")

    def test_representative_prefers_clean_border(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean = root / "tray.rf.clean.jpg"
            bordered = root / "tray.rf.bordered.jpg"
            clean_image = Image.new("RGB", (96, 96), "white")
            draw = ImageDraw.Draw(clean_image)
            draw.rectangle((24, 24, 72, 72), fill="navy")
            clean_image.save(clean)
            bordered_image = clean_image.filter(ImageFilter.GaussianBlur(radius=1.5))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle((0, 0, 95, 12), fill="black")
            draw.rectangle((0, 83, 95, 95), fill="black")
            bordered_image.save(bordered)
            box = (NormalizedBox(0.5, 0.5, 0.5, 0.5),)
            records = [
                SourceRecord("Khay_thuc_an_2", "train", "tray", bordered, box),
                SourceRecord("Khay_thuc_an_2", "train", "tray", clean, box),
            ]
            selected, rejected = select_representatives(records)
            self.assertEqual(selected[0].image_path, clean)
            self.assertEqual(selected[0].variant_count, 2)
            self.assertEqual(len(rejected), 1)

    def test_coco_loader_converts_and_clips_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Khay_thuc_an_4" / "train"
            root.mkdir(parents=True)
            Image.new("RGB", (100, 80), "white").save(root / "tray.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "tray.jpg", "width": 100, "height": 80}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [-10, 8, 70, 48]},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
                ],
                "categories": [{"id": 1, "name": "food"}],
            }
            (root / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
            records, raw_boxes = load_coco_records(root.parent)
            self.assertEqual(raw_boxes, 2)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].boxes), 1)
            self.assertAlmostEqual(records[0].boxes[0].width, 0.6)


if __name__ == "__main__":
    unittest.main()
