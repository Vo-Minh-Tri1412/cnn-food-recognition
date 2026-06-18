import tempfile
import unittest
from pathlib import Path

from canteen_checkout.config import PROJECT_ROOT
from canteen_checkout.yolo_runtime import resolve_yolo_model_reference, yolo_cache_working_directory


class YoloRuntimeTests(unittest.TestCase):
    def test_simple_model_name_stays_relative_to_cache(self):
        self.assertEqual(resolve_yolo_model_reference("yolo11s.pt"), "yolo11s.pt")

    def test_custom_relative_model_path_is_resolved_from_project(self):
        expected = str((PROJECT_ROOT / "models" / "custom.pt").resolve())
        self.assertEqual(resolve_yolo_model_reference("models/custom.pt"), expected)

    def test_cache_working_directory_is_created_and_restored(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "weights"
            with yolo_cache_working_directory(cache) as resolved:
                self.assertEqual(Path.cwd(), cache.resolve())
                self.assertEqual(resolved, cache.resolve())
            self.assertEqual(Path.cwd(), original)


if __name__ == "__main__":
    unittest.main()
