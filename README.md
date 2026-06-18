# Checkout the Canteen

Local-first canteen checkout project for 11 University of Economics Ho Chi Minh City (UEH) canteen dishes.

The main workflow is now intentionally small:

1. Keep trusted images in `data/reviewed/<class>/`.
2. Generate `data/classification/train|val|test/` from `data/reviewed/`.
3. Build/train the one-class YOLO `food_region` detector for automatic tray crops.
4. Train the dish classifier.
5. Build/train the auxiliary YOLO egg/fish detector.
6. Demo tray checkout with editable auto regions, safe grid fallback, detector fusion, prices, and bill JSON.

## Main Notebook

Use the single root notebook:

```text
00_colab_kaggle_workflow.ipynb
```

It covers setup, Drive/Kaggle data loading, cleaned Roboflow tray data, food-region and egg/fish YOLO training, classifier training, report reading, Grad-CAM, Demo App, Data IDE, and artifact export.

## Clean Data Contract

Only these folders matter for normal work:

```text
data/reviewed/<class>/                 trusted source of truth
data/classification/train|val|test/    generated classifier dataset
data/detection/egg_fish_shared/        generated YOLO dataset with safe hard negatives
data/detection/food_regions/           generated one-class YOLO crop detector dataset
data/archive/raw_tray_datasets_*/      immutable Roboflow source exports
data/inbox/review/<batch>/             images waiting for Data IDE review
data/extras/<label>/                   useful images outside the official classes
data/quarantine/<reason>/              rejected, duplicates, and label conflicts
data/demo/                             demo uploads and tray images
outputs/cloud/*.zip                    cloud-ready dataset packages
models/*.pt                            trained model weights
```

Legacy/raw/history data may still exist under `data/archive/`, `data/download/`, and `data/inbox/raw_batches/`, but do not train directly from those folders.

## Active Scripts

Scripts are grouped by purpose:

```text
scripts/apps/01_demo_checkout_app.py
scripts/apps/02_data_ide.py

scripts/cli/01_crop_tray.py
scripts/cli/02_demo_checkout.py

scripts/data/01_build_classification_dataset.py
scripts/data/02_audit_dataset_conflicts.py
scripts/data/03_package_classification_dataset.py
scripts/data/04_build_yolo_dataset.py
scripts/data/05_build_yolo_shared_negatives.py
scripts/data/06_package_yolo_dataset.py
scripts/data/07_collect_cookpad_candidates.py
scripts/data/08_build_food_region_dataset.py

scripts/train/01_train_classifier.py
scripts/train/02_train_yolo_detector.py
scripts/train/03_train_food_region_detector.py

scripts/debug/01_gradcam_debug.py
scripts/cloud/01_sync_drive_artifacts.py
```

Old one-off scrape/import/migration scripts were removed from the active repo to keep the project readable.

## Quick Commands

Start Data IDE:

```powershell
.\.venv\Scripts\python.exe scripts\apps\02_data_ide.py --host 127.0.0.1 --port 7864
```

Start Demo App:

```powershell
.\.venv\Scripts\python.exe scripts\apps\01_demo_checkout_app.py --host 127.0.0.1 --port 7863
```

The demo requests automatic food regions when `models/food_region_detector.pt` exists. Every box remains editable. Missing, empty, or implausible detector output falls back to the five-compartment grid.

Build classifier dataset:

```powershell
.\.venv\Scripts\python.exe scripts\data\01_build_classification_dataset.py `
  --reviewed-source data\reviewed `
  --old-weight 1 `
  --reviewed-weight 1 `
  --cross-class-hamming 4 `
  --clear `
  --clear-all
```

Audit classifier dataset:

```powershell
.\.venv\Scripts\python.exe scripts\data\02_audit_dataset_conflicts.py --root data\classification --phash-threshold 4
```

Package classifier dataset:

```powershell
.\.venv\Scripts\python.exe scripts\data\03_package_classification_dataset.py
```

Build and package YOLO detector dataset:

```powershell
.\.venv\Scripts\python.exe scripts\data\04_build_yolo_dataset.py --clear
.\.venv\Scripts\python.exe scripts\data\05_build_yolo_shared_negatives.py --clear --max-per-class 80
.\.venv\Scripts\python.exe scripts\data\06_package_yolo_dataset.py `
  --source data\detection\egg_fish_shared `
  --output outputs\cloud\egg_fish_shared_yolo.zip `
  --manifest outputs\cloud\egg_fish_shared_yolo.manifest.json
```

Clean and package the archived Roboflow tray datasets:

```powershell
.\.venv\Scripts\python.exe scripts\data\08_build_food_region_dataset.py --dry-run
.\.venv\Scripts\python.exe scripts\data\08_build_food_region_dataset.py --clear
.\.venv\Scripts\python.exe scripts\data\06_package_yolo_dataset.py `
  --source data\detection\food_regions `
  --output outputs\cloud\food_regions_yolo.zip `
  --manifest outputs\cloud\food_regions_yolo.manifest.json
```

The builder keeps `data/archive/` unchanged. It selects one representative from each Roboflow augmentation group, converts COCO/polygon annotations to `food_region`, drops invalid tiny boxes, and deduplicates across exports. Roboflow Stretch resizing cannot be reversed because the original aspect ratio is no longer present.

Train classifier:

```powershell
.\.venv\Scripts\python.exe scripts\train\01_train_classifier.py `
  --data data\classification `
  --arch efficientnet_b2 `
  --epochs 8 `
  --batch-size 16 `
  --image-size 260 `
  --augmentation light `
  --class-augmentation canh_chua_khong_ca=strong `
  --oversample-class canh_chua_khong_ca:2.5 `
  --label-smoothing 0.05
```

For an experiment after this baseline, add `--loss focal --focal-gamma 2.0`. Do not combine very aggressive oversampling and focal loss unless the confusion matrix still shows poor `canh_chua_khong_ca` recall.

Train YOLO detector:

```powershell
.\.venv\Scripts\python.exe scripts\train\02_train_yolo_detector.py `
  --data data\detection\egg_fish_shared\data.yaml `
  --model yolo11s.pt `
  --epochs 100 `
  --batch 16
```

Train the automatic food-region detector:

```powershell
.\.venv\Scripts\python.exe scripts\train\03_train_food_region_detector.py `
  --data data\detection\food_regions\data.yaml `
  --model yolo11s.pt `
  --epochs 100 `
  --batch 16
```

Use `yolo11n.pt` for smoke tests or lower-latency deployment. With the current 365-image clean dataset, `yolo11s.pt` is the recommended Colab baseline; moving to `yolo11m.pt` is unlikely to fix domain shift and should wait until an independent UEH holdout shows a capacity bottleneck.

Generate Grad-CAM:

```powershell
.\.venv\Scripts\python.exe scripts\debug\01_gradcam_debug.py --data data\classification --split test --max-samples 32
```

Collect Cookpad candidates into Data IDE review:

```powershell
.\.venv\Scripts\python.exe scripts\data\07_collect_cookpad_candidates.py `
  --target-class canh_chua_khong_ca `
  --queries-file configs\cookpad_canh_chua_khong_ca_extra_queries.txt `
  --goal 200 `
  --max-considered 900 `
  --apply
```

Other focused Cookpad query files:

- `configs/cookpad_trung_chien_queries.txt`
- `configs/cookpad_dau_hu_sot_ca_queries.txt`

Publish clean artifacts to Google Drive Desktop:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --publish --apply
```

Pull trained models and the newest report/run folder back from Drive:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --pull --apply
```

Use `--pull-models --apply` only when you intentionally want weights without reports. Full pulled reports are stored under `outputs/cloud/drive_runs/<timestamp>/`.

## Billing Classes

```text
com_trang
dau_hu_sot_ca
ca_hu_kho
thit_kho_trung
thit_kho
canh_chua_co_ca
canh_chua_khong_ca
suon_nuong
canh_rau
rau_xao
trung_chien
```

Prices live in `prices.csv`. The runtime has three model roles: the food-region detector finds billable crops, the dish classifier predicts one of 11 UEH dishes, and the auxiliary egg/fish detector resolves visually ambiguous cases.

## Roboflow Attribution

The archived tray datasets were supplied by Roboflow Universe users under CC BY 4.0. Source project URLs and export details remain in each dataset's `README.dataset.txt` and `README.roboflow.txt`. Generated clean datasets must preserve that attribution.
