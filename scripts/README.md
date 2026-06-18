# Script Layout

The project now keeps only active scripts in purpose-specific folders.

## DVC Pipeline

`dvc.yaml` connects the active data scripts into eight stages: classification build/audit/package, egg/fish build/hard-negative/package, and food-region build/package. Use the isolated environment from `requirements-dvc.txt`:

```powershell
.\.venv-dvc\Scripts\dvc.exe pull
.\.venv-dvc\Scripts\dvc.exe status
.\.venv-dvc\Scripts\dvc.exe repro
.\.venv-dvc\Scripts\dvc.exe push
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --push-inputs --apply
```

`dvc push` versions data cache. `--push-inputs` transfers the regenerated ZIP/manifest and project files for Colab. Models and `Drive/runs/` remain outside DVC.

Use a personal Google Cloud Desktop OAuth client for the first DVC push. Save `gdrive_client_id` and `gdrive_client_secret` with `dvc remote modify --local`; DVC writes them to ignored `.dvc/config.local` rather than the committed remote config.

## apps

- `01_demo_checkout_app.py`: browser demo app for tray checkout, local SQLite loyalty/vouchers, payment confirmation, and per-item ratings.
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
- `08_build_food_region_dataset.py`: build a clean one-class `food_region` YOLO dataset from immutable archived Roboflow exports; selects representative variants and deduplicates sources.

## train

- `01_train_classifier.py`: train the 11-class dish classifier; supports focal loss, targeted oversampling, per-class augmentation overrides, validation-accuracy early stopping, and `classifier_training_summary.json`.
- `02_train_yolo_detector.py`: train the auxiliary YOLO egg/fish detector; `--weights-cache` keeps downloaded YOLO and AMP-check weights outside the repository root.
- `03_train_food_region_detector.py`: train the automatic tray food-region detector used before dish classification; supports the same isolated weight cache.

The generic `06_package_yolo_dataset.py` packages both egg/fish and food-region datasets by passing explicit `--source`, `--output`, and `--manifest` paths.

## debug

- `01_gradcam_debug.py`: generate Grad-CAM samples for classifier debugging.

## cloud

- `01_sync_drive_artifacts.py`: use `--push-inputs --apply` for dataset/project files and `--pull-results --apply` for canonical models plus the newest run of each model. `--push-inputs` never uploads local model weights.

The Colab notebook writes `run_provenance.json` for all three models and `test_metrics.json` for both YOLO test splits. These files stay under `Drive/runs/<model>/<timestamp>/reports/` and are pulled with the rest of the run.

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
