from __future__ import annotations

import unittest

from canteen_checkout.cropping import CropRegion
from canteen_checkout.detector import DetectorEvidence, ObjectDetection, partition_evidence_by_regions


class DetectorEvidenceTests(unittest.TestCase):
    def test_tray_detections_are_assigned_once_and_made_crop_local(self) -> None:
        regions = [
            CropRegion("left", 100, 50, 200, 200),
            CropRegion("right", 350, 50, 200, 200),
        ]
        evidence = DetectorEvidence(
            egg_count=1,
            fish_count=1,
            detections=(
                ObjectDetection("egg", 0.8, (140, 90, 220, 180)),
                ObjectDetection("fish", 0.7, (390, 100, 500, 200)),
                ObjectDetection("egg", 0.6, (700, 100, 760, 180)),
            ),
            detector_loaded=True,
            detector_path="detector.pt",
        )

        assigned = partition_evidence_by_regions(evidence, regions)

        self.assertEqual([row.egg_count for row in assigned], [1, 0])
        self.assertEqual([row.fish_count for row in assigned], [0, 1])
        self.assertEqual(assigned[0].detections[0].xyxy, (40, 40, 120, 130))
        self.assertEqual(assigned[1].detections[0].xyxy, (40, 50, 150, 150))


if __name__ == "__main__":
    unittest.main()
