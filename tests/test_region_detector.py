from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image

from canteen_checkout.region_detector import RegionCandidate, detect_food_regions, regions_from_candidates


class _Value:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self

    def item(self):
        return self.value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Box:
    def __init__(self, confidence: float, xyxy: list[float]):
        self.cls = _Value(0)
        self.conf = _Value(confidence)
        self.xyxy = _Value(xyxy)


class _Detector:
    names = {0: "food_region"}

    def predict(self, *_args, **_kwargs):
        result = type("Result", (), {"names": self.names, "boxes": [_Box(0.91, [10, 12, 70, 60])]})()
        return [result]


class RegionDetectorTests(unittest.TestCase):
    def test_regions_are_filtered_padded_and_sorted(self) -> None:
        regions, fallback = regions_from_candidates(
            [
                RegionCandidate(0.8, (50, 50, 90, 90)),
                RegionCandidate(0.9, (5, 5, 45, 45)),
                RegionCandidate(0.7, (0, 0, 2, 2)),
            ],
            image_width=100,
            image_height=100,
        )
        self.assertEqual(fallback, "")
        self.assertEqual([region.name for region in regions], ["auto_01", "auto_02"])
        self.assertEqual(regions[0].source, "auto_detector")
        self.assertEqual(regions[0].x, 3)
        self.assertEqual(regions[0].y, 3)
        self.assertAlmostEqual(regions[0].confidence or 0, 0.9)

    def test_empty_candidates_trigger_fallback(self) -> None:
        regions, fallback = regions_from_candidates([], image_width=100, image_height=100)
        self.assertEqual(regions, ())
        self.assertEqual(fallback, "no_regions")

    def test_too_many_candidates_trigger_fallback(self) -> None:
        candidates = [RegionCandidate(0.9, (0, 0, 20, 20)) for _ in range(9)]
        regions, fallback = regions_from_candidates(candidates, image_width=100, image_height=100)
        self.assertEqual(regions, ())
        self.assertEqual(fallback, "too_many_regions")

    def test_fake_detector_returns_auto_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "tray.jpg"
            Image.new("RGB", (100, 80), "white").save(image_path)
            result = detect_food_regions(_Detector(), image_path)
            self.assertTrue(result.detector_loaded)
            self.assertEqual(result.fallback_reason, "")
            self.assertEqual(len(result.regions), 1)
            self.assertEqual(result.regions[0].source, "auto_detector")


if __name__ == "__main__":
    unittest.main()
