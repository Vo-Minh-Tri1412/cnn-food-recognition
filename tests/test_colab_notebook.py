import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "00_colab_kaggle_workflow.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_code_cells_compile_and_have_no_saved_outputs(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_notebook_is_colab_only_and_has_expected_sections(self):
        expected = [
            "## 1. Tổng quan quy trình và chuẩn bị dữ liệu trên máy local",
            "## 2. Chuẩn bị chung trên Google Colab",
            "## 3. Mô hình phân loại 11 món ăn",
            "## 4. Mô hình phát hiện vùng thức ăn `food_region`",
            "## 5. Mô hình phát hiện phụ trợ `egg`/`fish`",
            "## 6. Đưa mô hình và báo cáo về máy local bằng Google Drive Desktop",
        ]
        positions = [self.markdown.index(title) for title in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("Kaggle", self.markdown)
        self.assertNotIn("files.upload", self.code)

    def test_each_model_writes_to_its_own_drive_run(self):
        self.assertIn('tao_run("classifier")', self.code)
        self.assertIn('tao_run("food_region")', self.code)
        self.assertIn('tao_run("egg_fish")', self.code)
        self.assertIn('DRIVE_MODELS / "dish_classifier.pt"', self.code)
        self.assertIn('DRIVE_MODELS / "food_region_detector.pt"', self.code)
        self.assertIn('DRIVE_MODELS / "egg_fish_detector.pt"', self.code)
        self.assertNotIn("Lưu model/report về Google Drive", self.markdown)

    def test_yolo_sections_use_runtime_weight_cache(self):
        self.assertEqual(self.code.count('"--weights-cache", YOLO_CACHE_ROOT'), 2)


if __name__ == "__main__":
    unittest.main()
