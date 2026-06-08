from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_TEACHER_TRAYS_DIR = DATA_DIR / "raw_teacher_trays"
DEMO_TRAYS_DIR = DATA_DIR / "demo_trays"
CLASSIFICATION_DIR = DATA_DIR / "classification"
DOWNLOADS_DIR = DATA_DIR / "downloads"
SCRAPED_CANDIDATES_DIR = DATA_DIR / "scraped_candidates"
PROCESSED_CANDIDATES_DIR = DATA_DIR / "processed_candidates"
REJECTED_CANDIDATES_DIR = DATA_DIR / "rejected_candidates"
TEMP_TEACHER_CROPS_DIR = DATA_DIR / "temp_teacher_crops"
ARCHIVE_DIR = DATA_DIR / "archive"
SCRAPED_MANIFEST_CSV = DATA_DIR / "scraped_manifest.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CROPPED_DISHES_DIR = OUTPUTS_DIR / "cropped_dishes"
BILLS_DIR = OUTPUTS_DIR / "bills"
REPORTS_DIR = OUTPUTS_DIR / "reports"

PRICES_CSV = PROJECT_ROOT / "prices.csv"
CLASS_NAMES_JSON = MODELS_DIR / "class_names.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "dish_classifier.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DISH_CLASSES = [
    "com_trang",
    "dau_hu_sot_ca",
    "ca_hu_kho",
    "thit_kho_trung",
    "thit_kho",
    "canh_chua_co_ca",
    "canh_chua_khong_ca",
    "suon_nuong",
    "canh_rau",
    "rau_xao",
    "trung_chien",
]

DISPLAY_NAMES = {
    "com_trang": "Com trang",
    "dau_hu_sot_ca": "Dau hu sot ca",
    "ca_hu_kho": "Ca hu kho",
    "thit_kho_trung": "Thit kho trung",
    "thit_kho": "Thit kho",
    "canh_chua_co_ca": "Canh chua co ca",
    "canh_chua_khong_ca": "Canh chua khong ca",
    "suon_nuong": "Suon nuong",
    "canh_rau": "Canh rau",
    "rau_xao": "Rau xao",
    "trung_chien": "Trung chien",
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
