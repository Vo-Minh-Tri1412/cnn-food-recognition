# Checkout the Canteen

Local-first canteen checkout project for 11 HCMUS canteen dishes.

The main workflow is now intentionally small:

1. Keep trusted images in `data/reviewed/<class>/`.
2. Generate `data/classification/train|val|test/` from `data/reviewed/`.
3. Train the dish classifier.
4. Build/train the auxiliary YOLO egg/fish detector.
5. Demo tray checkout with crop regions, detector fusion, prices, and bill JSON.

## Main Notebook

Use the single root notebook:

```text
00_colab_kaggle_workflow.ipynb
```

It covers setup, Drive/Kaggle data loading, classifier training, YOLO training, report reading, Grad-CAM, Demo App, Data IDE, and artifact export.

## Clean Data Contract

Only these folders matter for normal work:

```text
data/reviewed/<class>/                 trusted source of truth
data/classification/train|val|test/    generated classifier dataset
data/detection/egg_fish_shared/        generated YOLO dataset with safe hard negatives
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

scripts/train/01_train_classifier.py
scripts/train/02_train_yolo_detector.py

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

Train classifier:

```powershell
.\.venv\Scripts\python.exe scripts\train\01_train_classifier.py `
  --data data\classification `
  --arch efficientnet_b2 `
  --epochs 8 `
  --batch-size 16 `
  --image-size 260 `
  --augmentation strong `
  --label-smoothing 0.05
```

Train YOLO detector:

```powershell
.\.venv\Scripts\python.exe scripts\train\02_train_yolo_detector.py `
  --data data\detection\egg_fish_shared\data.yaml `
  --model yolo11s.pt `
  --epochs 100 `
  --batch 16
```

Generate Grad-CAM:

```powershell
.\.venv\Scripts\python.exe scripts\debug\01_gradcam_debug.py --data data\classification --split test --max-samples 32
```

Publish clean artifacts to Google Drive Desktop:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --publish --apply
```

Pull trained models back from Drive:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --pull-models --apply
```

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

Prices live in `prices.csv`. Detector fusion is auxiliary: YOLO helps distinguish egg/fish cases, but the dish classifier remains the main model.
