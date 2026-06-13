# Data Layout

This project keeps old scrape/import history for traceability, but the working data contract is intentionally small.

## Clean Working Folders

Use these folders for normal work:

```text
data/reviewed/<class>/                 trusted manually reviewed image pool
data/classification/train|val|test/    generated classifier dataset
data/detection/egg_fish_shared/        generated YOLO egg/fish dataset with reviewed hard negatives
data/inbox/review/<batch>/             new images waiting for Data IDE review
data/extras/<label>/                   useful images outside the official 11 classes
data/quarantine/<reason>/              rejected, duplicate, or label-conflict images
data/demo/                             demo uploads and tray images
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
```

Recommended one-shot publish to Google Drive for desktop:

```powershell
.\.venv\Scripts\python.exe scripts\cloud\01_sync_drive_artifacts.py --publish --apply
```

The Drive destination is:

```text
MyDrive/canteen_checkout/
  datasets/
  models/
  project_files/
```

## Rule Of Thumb

- Add new untrusted images to `data/inbox/review/`.
- Use Data IDE to move trusted images into `data/reviewed/`.
- Rebuild `data/classification/` from `data/reviewed/`.
- Rebuild `data/detection/egg_fish_shared/` from YOLO positives plus safe reviewed negatives.
- Never manually edit `data/classification/` as the long-term source of truth.
