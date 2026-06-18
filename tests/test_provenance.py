import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from canteen_checkout.provenance import sha256_file, write_json, write_run_provenance


class ProvenanceTests(unittest.TestCase):
    def test_write_json_converts_path_and_scalar_like_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            write_json(output, {"path": Path("model.pt"), "metric": ScalarLike(0.75)})
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"path": "model.pt", "metric": 0.75})

    def test_run_provenance_links_git_dvc_data_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.txt").write_text("tracked", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "test"], cwd=root, check=True)

            lock = root / "dvc.lock"
            archive = root / "dataset.zip"
            manifest = root / "dataset.manifest.json"
            model = root / "model.pt"
            lock.write_text("schema: '2.0'", encoding="utf-8")
            archive.write_bytes(b"zip")
            manifest.write_text("{}", encoding="utf-8")
            model.write_bytes(b"weights")

            output = root / "run" / "reports" / "run_provenance.json"
            payload = write_run_provenance(
                output,
                project_root=root,
                model_key="classifier",
                model_path=model,
                dataset_archive=archive,
                dataset_manifest=manifest,
                hyperparameters={"epochs": 8},
                training_seconds=12.3456,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(payload["model_key"], "classifier")
            self.assertEqual(payload["training_seconds"], 12.346)
            self.assertEqual(payload["dvc_lock"]["sha256"], sha256_file(lock))
            self.assertEqual(payload["dataset"]["archive"]["sha256"], sha256_file(archive))
            self.assertEqual(payload["model"]["sha256"], sha256_file(model))
            self.assertEqual(len(payload["git_sha"]), 40)


class ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


if __name__ == "__main__":
    unittest.main()
