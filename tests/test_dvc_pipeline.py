import configparser
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DvcPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        cls.params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))

    def test_pipeline_has_expected_stages(self):
        self.assertEqual(
            list(self.pipeline["stages"]),
            [
                "build_classification",
                "audit_classification",
                "package_classification",
                "build_egg_fish",
                "add_egg_fish_negatives",
                "package_egg_fish",
                "build_food_regions",
                "package_food_regions",
            ],
        )

    def test_sources_and_reproducibility_parameters_are_pinned(self):
        self.assertEqual(self.params["classification"]["reviewed_source"], "data/reviewed")
        self.assertEqual(
            self.params["food_regions"]["source"],
            "data/archive/raw_tray_datasets_20260610_174513",
        )
        self.assertEqual(
            self.params["egg_fish"]["source"],
            "data/download/roboflow_yolo_deduped/20260612_181500",
        )
        self.assertEqual(self.params["classification"]["seed"], 42)
        self.assertEqual(self.params["egg_fish"]["hard_negative_max_per_class"], 80)

    def test_delivery_packages_are_not_cached(self):
        stages = self.pipeline["stages"]
        for stage_name in ("package_classification", "package_egg_fish", "package_food_regions"):
            for output in stages[stage_name]["outs"]:
                self.assertIsInstance(output, dict)
                self.assertFalse(next(iter(output.values()))["cache"])

    def test_remote_has_no_credentials(self):
        config = configparser.ConfigParser()
        config.read(ROOT / ".dvc" / "config", encoding="utf-8")
        self.assertEqual(config["core"]["remote"], "gdrive")
        self.assertEqual(config["core"]["site_cache_dir"], ".dvc/tmp/site-cache")
        remote_section = next(section for section in config.sections() if "remote" in section)
        self.assertEqual(
            config[remote_section]["url"],
            "gdrive://root/canteen_checkout/dvc-storage",
        )
        self.assertEqual(set(config[remote_section]), {"url"})
        self.assertIn("/config.local", (ROOT / ".dvc" / ".gitignore").read_text(encoding="utf-8"))

    def test_source_pointers_and_lock_file_exist(self):
        paths = [
            ROOT / "data" / "reviewed.dvc",
            ROOT / "data" / "archive" / "raw_tray_datasets_20260610_174513.dvc",
            ROOT / "data" / "download" / "roboflow_yolo_deduped" / "20260612_181500.dvc",
            ROOT / "dvc.lock",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
