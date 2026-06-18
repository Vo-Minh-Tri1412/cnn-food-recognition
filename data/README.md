# Data Layout

This project keeps old scrape/import history for traceability, but the working data contract is intentionally small.

## DVC Sources

The source-of-truth snapshots are tracked by these pointer files:

```text
data/reviewed.dvc
data/archive/raw_tray_datasets_20260610_174513.dvc
data/download/roboflow_yolo_deduped/20260612_181500.dvc
```

Restore them with `.\.venv-dvc\Scripts\dvc.exe pull`. Build parameters live in `params.yaml`; run `.\.venv-dvc\Scripts\dvc.exe repro` to regenerate datasets, reports, ZIP files, manifests, and `dvc.lock`. DVC caches generated dataset directories, while cloud ZIP/manifest outputs use `cache: false` because they are reproducible delivery artifacts.

## Clean Working Folders

Use these folders for normal work:

```text
data/reviewed/<class>/                 trusted manually reviewed image pool
data/classification/train|val|test/    generated classifier dataset
data/detection/egg_fish_shared/        generated YOLO egg/fish dataset with reviewed hard negatives
data/detection/food_regions/           generated one-class automatic crop dataset
data/inbox/review/<batch>/             new images waiting for Data IDE review
data/extras/<label>/                   useful images outside the official 11 classes
data/quarantine/<reason>/              rejected, duplicate, or label-conflict images
data/demo/                             demo uploads and tray images
data/archive/raw_tray_datasets_*/      immutable CC BY 4.0 Roboflow tray exports
```

## Ignore During Normal Work

These folders are legacy/raw/history. Do not train directly from them unless a script explicitly says so:

```text
data/archive/
data/download/
data/downloads/
data/inbox/raw_batches/
```

## Cloud Artifacts

The cloud notebook should use zip artifacts from `outputs/cloud/` or Google Drive:

```text
outputs/cloud/classification.zip
outputs/cloud/egg_fish_shared_yolo.zip
outputs/cloud/food_regions_yolo.zip
```

Recommended input-only push to Google Drive for desktop:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --push-inputs --apply
```

The Drive destination is:

```text
MyDrive/canteen_checkout/
  datasets/
  models/
  project_files/
  runs/
    classifier/<timestamp>/
    food_region/<timestamp>/
    egg_fish/<timestamp>/
```

Dataset packages and project files move from local to Drive. Trained models and run reports move from Drive back to local with `--pull-results --apply`; local models are never included in `--push-inputs`.

Run `dvc push` before `--push-inputs --apply`: DVC publishes source/build cache through the Google Drive API, while the sync script publishes Colab-ready packages through Google Drive Desktop. These are separate responsibilities.

If Google blocks the shared `dvc-gdrive` OAuth application, configure a personal Desktop OAuth client with `dvc remote modify --local gdrive gdrive_client_id ...` and `gdrive_client_secret ...`. The resulting `.dvc/config.local` must remain untracked.

## Rule Of Thumb

- Add new untrusted images to `data/inbox/review/`.
- Use Data IDE to move trusted images into `data/reviewed/`.
- Rebuild `data/classification/` from `data/reviewed/`.
- Rebuild `data/detection/egg_fish_shared/` from YOLO positives plus safe reviewed negatives.
- Rebuild `data/detection/food_regions/` with `scripts/data/08_build_food_region_dataset.py`; never edit the Roboflow archive in place.
- Never manually edit `data/classification/` as the long-term source of truth.

## Roboflow Cleanup Contract

- `Khay_thuc_an_4`, `Khay_thuc_an_2`, and `Khay_thuc_an` are normalized to one `food_region` class.
- `Khay_thuc_an_3` is audited but excluded by default because it labels fixed four-compartment grids, including potentially empty regions.
- One representative is selected per Roboflow source group using black-border ratio, blur score, and deterministic filename ordering.
- Exact and perceptual duplicates are removed globally while preserving the higher-priority source.
- Stretch-resized images cannot be restored to their original aspect ratio; the archive remains unchanged for traceability and CC BY 4.0 attribution.
