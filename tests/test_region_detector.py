from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image

from canteen_checkout.cropping import five_compartment_template
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
        boxes = [_Box(0.91, [region.x, region.y, region.x + region.w, region.y + region.h]) for region in five_compartment_template(1000, 600)]
        result = type("Result", (), {"names": self.names, "boxes": boxes})()
        return [result]


class RegionDetectorTests(unittest.TestCase):
    def test_regions_are_filtered_padded_and_sorted(self) -> None:
        expected = five_compartment_template(1000, 600)
        regions, fallback = regions_from_candidates(
            [RegionCandidate(0.9 - index * 0.01, (r.x, r.y, r.x + r.w, r.y + r.h)) for index, r in enumerate(expected)],
            image_width=1000,
            image_height=600,
        )
        self.assertEqual(fallback, "")
        self.assertEqual(len(regions), 5)
        self.assertEqual([region.name for region in regions], [f"auto_{index:02d}" for index in range(1, 6)])
        self.assertEqual(regions[0].source, "auto_detector")
        self.assertLess(regions[0].x, expected[0].x)
        self.assertLess(regions[0].y, expected[0].y)
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

    def test_unexpected_region_counts_trigger_fallback(self) -> None:
        expected = five_compartment_template(1000, 600)
        candidates = [RegionCandidate(0.9, (r.x, r.y, r.x + r.w, r.y + r.h)) for r in expected]
        for count in (3, 4, 6):
            rows = candidates[:count]
            if count == 6:
                rows.append(RegionCandidate(0.5, (450, 250, 550, 350)))
            regions, fallback = regions_from_candidates(rows, image_width=1000, image_height=600)
            self.assertEqual(regions, ())
            self.assertEqual(fallback, f"unexpected_region_count:{count}")

    def test_implausible_five_regions_trigger_fallback(self) -> None:
        candidates = [RegionCandidate(0.9, (10 + index * 5, 10, 200 + index * 5, 200)) for index in range(5)]
        regions, fallback = regions_from_candidates(candidates, image_width=1000, image_height=600)
        self.assertEqual(regions, ())
        self.assertEqual(fallback, "implausible_five_region_layout")

    def test_fake_detector_returns_auto_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "tray.jpg"
            Image.new("RGB", (1000, 600), "white").save(image_path)
            result = detect_food_regions(_Detector(), image_path)
            self.assertTrue(result.detector_loaded)
            self.assertEqual(result.fallback_reason, "")
            self.assertEqual(len(result.regions), 5)
            self.assertEqual(result.regions[0].source, "auto_detector")


if __name__ == "__main__":
    unittest.main()
