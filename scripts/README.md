# Script Layout

The project now keeps only active scripts in purpose-specific folders.

## apps

- `01_demo_checkout_app.py`: browser demo app for tray checkout.
- `02_data_ide.py`: local Data IDE for reviewing and moving images.

## cli

- `01_crop_tray.py`: manual crop helper for tray images.
- `02_demo_checkout.py`: command-line checkout demo.

## data

- `01_build_classification_dataset.py`: build `data/classification` from `data/reviewed`.
- `02_audit_dataset_conflicts.py`: audit invalid images and cross-class duplicates.
- `03_package_classification_dataset.py`: package `classification.zip` for Colab/Kaggle/Drive.
- `04_build_yolo_dataset.py`: build the base YOLO egg/fish dataset from deduped Roboflow exports.
- `05_build_yolo_shared_negatives.py`: add safe reviewed classification images as empty-label YOLO hard negatives.
- `06_package_yolo_dataset.py`: package YOLO dataset zip for cloud training.
- `07_collect_cookpad_candidates.py`: collect class-aware Cookpad candidates into `data/inbox/review` with text filtering, model gate, optional YOLO fish gate, hash-cache dedupe, and `--max-considered` run caps.

## train

- `01_train_classifier.py`: train the 11-class dish classifier; supports focal loss, targeted oversampling, and per-class augmentation overrides.
- `02_train_yolo_detector.py`: train the auxiliary YOLO egg/fish detector.

## debug

- `01_gradcam_debug.py`: generate Grad-CAM samples for classifier debugging.

## cloud

- `01_sync_drive_artifacts.py`: sync packages, models, project files, and full Drive run/report folders through Google Drive Desktop. Use `--pull --apply` to fetch both weights and reports.

## Removed Legacy Scripts

The old crawl/import/migration scripts were removed because the current workflow uses Data IDE plus the clean data contract:

- `00_prepare_project.py`
- `01_inventory_data.py`
- `09_collect_web_images.py`
- `14_import_external_datasets.py`
- `16_model_assisted_filter.py`
- `18_import_scrape_batch_to_review.py`
- `22_search_public_datasets.py`
- `23_stage_unreviewed_sources.py`
- `25_migrate_data_layout.py`
- `26_dedupe_reviewed.py`
- `28_crawl_canh_chua_to_review.py`
