# Checkout the Canteen

Local-first implementation for an automated canteen checkout demo.

The current project is set up to:

1. keep teacher-provided tray photos as real demo/test images;
2. train an 11-class dish classifier when labeled dish crops are available;
3. crop dish regions from tray images;
4. classify each crop, look up `prices.csv`, and export a bill JSON.

## Recommended Workflow Now

The fastest path is no longer "open every image and review by hand". Use the model to pre-filter first:

```powershell
.\.venv\Scripts\python.exe scripts\14_import_external_datasets.py
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --epochs 3 --apply
.\.venv\Scripts\python.exe scripts\15_review_external_dataset.py --staging data\downloads\external_staging\external_20260609_115250 --port 7860
```

What this does:

- trains a temporary dataset-filter model from the latest curated merge under `data/downloads/merge_batches/*/processed`;
- predicts every image/crop in the latest external staging review pool;
- groups candidates into `model_assisted/auto_accepted`, `model_assisted/needs_review`, `model_assisted/ambiguous_review`, and `model_assisted/model_rejected`;
- rejects near-duplicates of seed/reviewed images before they can become `auto_accepted`;
- writes `reports/model_suggestions.csv` so the review app can show model prediction, confidence, and decision reason.

When you trust the threshold enough, add `--promote-auto` to copy very confident predictions straight into `reviewed/<class_name>/`:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --skip-train --apply --promote-auto
```

Use this carefully. It is meant for high-confidence candidates only; ambiguous classes still need review.

### Active Learning Loop

After one review pass, retrain the filter from the baseline plus your newly reviewed labels, then regroup the staging data:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --force-train --include-staging-reviewed --epochs 5 --apply
```

Only `reviewed/<class_name>` is used as training seed for the 11-class billing problem. The app also supports `reviewed_extra/<label>` for useful food images outside the assignment, but those extra folders are intentionally ignored by the billing classifier.

Current data policy:

```text
review/<pool>/                 # raw candidates/crops to review
model_assisted/<decision>/     # generated grouping from the filter model
reviewed/<11 class>/           # trusted labels used for training
reviewed_extra/<custom label>/ # useful, outside the current 11-class problem
manual_rejected/<pool>/        # bad or irrelevant images
reports/*.csv|*.json           # manifests, logs, model suggestions
```

Duplicate policy:

- import/preprocess removes obvious duplicates early;
- model-assisted filtering runs a second duplicate gate against seed/reviewed images and previous candidates;
- duplicate images are copied to `model_assisted/model_rejected/duplicate_seed` or `model_assisted/model_rejected/duplicate_candidate`;
- source files are not deleted.

Tune the pHash threshold if needed:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --skip-train --apply --duplicate-hamming 10
```

## Environment

The local environment has already been created at `.venv/` with Python 3.12.

Use this Python for every command:

```powershell
.\.venv\Scripts\python.exe
```

Verified locally:

- PyTorch: `2.11.0+cu128`
- GPU available: `True`
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- OpenCV and Ultralytics import successfully

## Folder Layout

```text
data/
  raw_teacher_trays/      # copied images from Khay_com/
  demo_trays/             # small subset for demo runs
  scraped_candidates/     # web images before manual review
  processed_candidates/   # normalized images after quality filtering
  rejected_candidates/    # suspicious images copied/moved out by reason
  temp_teacher_crops/     # teacher-tray crops kept outside training
  classification/
    train/<class_name>/   # labeled dish crops for training
    val/<class_name>/
    test/<class_name>/
models/
  class_names.json
  dish_classifier.pt      # created after real training
outputs/
  cropped_dishes/
  bills/
  reports/
scripts/
  00_prepare_project.py
  01_inventory_data.py
  02_download_public_dataset.py
  03_make_dataset_split.py
  04_crop_tray.py
  05_train_classifier.py
  06_demo_checkout.py
  07_smoke_test.py
  08_seed_teacher_crops.py
  09_collect_web_images.py
  10_make_review_contact_sheets.py
  11_promote_reviewed_images.py
  12_clean_workspace.py
  13_preprocess_candidates.py
  14_import_external_datasets.py
  15_review_external_dataset.py
  16_model_assisted_filter.py
notebooks/
  00_setup_and_inventory.ipynb
  01_train_classifier.ipynb
  02_demo_checkout.ipynb
```

## The 11 Classes

The implementation currently follows the 11 dishes from the PDF price table:

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

Prices are stored in `prices.csv`.

## Prepare and Inspect Data

```powershell
.\.venv\Scripts\python.exe scripts\00_prepare_project.py
.\.venv\Scripts\python.exe scripts\01_inventory_data.py
```

Notebook equivalent:

```text
notebooks/00_setup_and_inventory.ipynb
```

Current state after preparation:

- `data/raw_teacher_trays/`: 56 tray images
- `data/demo_trays/`: 6 tray images
- `data/classification/`: empty until labeled dish crops are added

## Crop Teacher Tray Images

Fast MVP crop using the built-in five-compartment tray template:

```powershell
.\.venv\Scripts\python.exe scripts\04_crop_tray.py --image data\demo_trays\YOUR_IMAGE.jpg
```

Manual crop with an OpenCV selection window:

```powershell
.\.venv\Scripts\python.exe scripts\04_crop_tray.py --image data\demo_trays\YOUR_IMAGE.jpg --interactive --save-regions configs\my_regions.json
```

If the JSON regions include labels, add crops directly into the training dataset:

```powershell
.\.venv\Scripts\python.exe scripts\04_crop_tray.py --image data\demo_trays\YOUR_IMAGE.jpg --regions-json configs\my_regions.json --add-to-dataset train
```

Use `configs/manual_regions.example.json` as the format reference.

## Seed Training Crops From Teacher Images

After `scripts/00_prepare_project.py` copies `Khay_com/` into `data/raw_teacher_trays/`, create the first curated training crops:

```powershell
.\.venv\Scripts\python.exe scripts\08_seed_teacher_crops.py --split train --clear-existing
```

This creates a small, conservative seed dataset from clear teacher-tray crops. It is useful for domain adaptation, but it is not enough for final training by itself.

## Collect Web Image Candidates

If public Kaggle datasets are not useful enough, collect web image candidates into a review folder:

```powershell
.\.venv\Scripts\python.exe scripts\09_collect_web_images.py --class-name com_trang --per-query 30 --max-downloads-per-class 80
```

The crawler writes `data/scraped_manifest.csv` with class, query, source URL, local path, provider, and download time. It also appends negative terms such as `-logo`, `-icon`, and `-emoji` unless `--raw-query` is used.

When adding a new scrape batch after a merged baseline already exists, seed duplicate checks from that baseline:

```powershell
.\.venv\Scripts\python.exe scripts\09_collect_web_images.py --class-name com_trang --out data\downloads\scrape_batches\NEW_BATCH\raw --manifest data\downloads\scrape_batches\NEW_BATCH\manifest.csv --dedupe-against data\downloads\merge_batches\MERGE_BATCH\processed
```

This skips exact URL repeats and near-duplicate image hashes inside the new batch, while the later preprocess/merge step still performs the stronger final dedupe.

The default provider is DuckDuckGo image search. Bing is available as a fallback:

```powershell
.\.venv\Scripts\python.exe scripts\09_collect_web_images.py --provider bing --class-name com_trang
```

The query file includes teacher-note bias:

```text
canh_rau: cai, rau muong
rau_xao: lagim, cu san, dau que, dau dua
```

Create contact sheets for manual review:

```powershell
.\.venv\Scripts\python.exe scripts\10_make_review_contact_sheets.py
```

Open `outputs/reports/scraped_review_sheets/`, delete bad images from `data/scraped_candidates/<class_name>/`, then preprocess the remaining images:

```powershell
.\.venv\Scripts\python.exe scripts\13_preprocess_candidates.py --dry-run
.\.venv\Scripts\python.exe scripts\13_preprocess_candidates.py
```

This normalizes images into `data/processed_candidates/`, copies suspicious files into `data/rejected_candidates/<reason>/`, and writes `outputs/reports/data_quality_report.csv`.

Promote reviewed, preprocessed images into train/val/test:

```powershell
.\.venv\Scripts\python.exe scripts\11_promote_reviewed_images.py --dry-run
.\.venv\Scripts\python.exe scripts\11_promote_reviewed_images.py
```

Important: do not promote images before visual review. Web search often returns logos, icons, unrelated dishes, or duplicate photos.

## Import External Datasets

When using downloaded dish folders and Roboflow tray datasets, import them into isolated review pools instead of training from them directly:

```powershell
.\.venv\Scripts\python.exe scripts\14_import_external_datasets.py
```

This creates `data/downloads/external_staging/external_<timestamp>/` with:

```text
review/<pool>/              # normalized candidates and Roboflow crops
reviewed/<class_name>/      # empty class folders for manually accepted images
reports/review_sheets/      # paged contact sheets
reports/external_import_manifest.csv
reports/review_instructions.txt
```

Ambiguous pools such as `thit_kho_or_thit_kho_trung`, `canh_chua_unknown`, `protein_grid_review`, and `unknown_food_crops` must be manually copied into the right `reviewed/<class_name>/` folder before promotion.

For faster local review, launch the browser UI:

```powershell
.\.venv\Scripts\python.exe scripts\15_review_external_dataset.py --staging data\downloads\external_staging\external_20260609_115250 --port 7860
```

The app copies accepted images into `reviewed/<class_name>/`, copies rejects into `manual_rejected/`, and logs every action to `reports/review_actions.csv`.

If `scripts/16_model_assisted_filter.py` has already run, the app also shows model prediction, confidence, decision, and reason. Use the left sidebar's model filter to open only `needs_review`, `ambiguous_review`, or `model_rejected` items.

Use the `Extra Folders` panel for images that are useful later but outside the current 11-class price table. These are copied to `reviewed_extra/<label>` and are not promoted into `data/classification` by the normal 11-class workflow.

## Model-Assisted Dataset Filtering

After external import, let the current seed dataset do the first pass:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --epochs 3 --apply
```

Default inputs:

- seed dataset: newest `data/downloads/merge_batches/merge_*/processed`;
- staging dataset: newest `data/downloads/external_staging/external_*`;
- model output: `models/data_filter_classifier.pt`.

Important thresholds:

```powershell
--duplicate-hamming 8
--auto-accept-confidence 0.92
--review-confidence 0.55
--reject-confidence 0.25
--min-margin 0.15
```

Recommended usage:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --force-train --epochs 3 --apply
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --skip-train --apply --promote-auto
```

The first command groups the data for inspection. The second command copies only high-confidence predictions into `reviewed/<class_name>/`. After that, use the review app for ambiguous folders only.

For later passes, include your reviewed labels so the filter improves:

```powershell
.\.venv\Scripts\python.exe scripts\16_model_assisted_filter.py --force-train --include-staging-reviewed --epochs 5 --apply
```

When enough labels are trustworthy, promote reviewed data:

```powershell
.\.venv\Scripts\python.exe scripts\11_promote_reviewed_images.py --source data\downloads\external_staging\external_20260609_115250\reviewed --prefix external --dry-run
.\.venv\Scripts\python.exe scripts\11_promote_reviewed_images.py --source data\downloads\external_staging\external_20260609_115250\reviewed --prefix external
```

## Clean Workspace Artifacts

Archive smoke-test outputs and old demo artifacts without touching source data:

```powershell
.\.venv\Scripts\python.exe scripts\12_clean_workspace.py --dry-run
.\.venv\Scripts\python.exe scripts\12_clean_workspace.py --apply --mode archive
```

Use `--mode delete` only when you are sure the listed files are not needed.

Optional generated cleanup groups:

```powershell
.\.venv\Scripts\python.exe scripts\12_clean_workspace.py --dry-run --include-review-sheets --include-model-assisted
.\.venv\Scripts\python.exe scripts\12_clean_workspace.py --apply --mode archive --include-review-sheets --include-model-assisted
```

Old web scrape batches can also be archived with `--include-scrape-batches`. Keep this off unless you are sure those raw batches are no longer useful.

## Download Public Dataset Notes

Dataset reference notes are in:

```text
data/downloads/README_DATASETS.md
```

Optional downloader:

```powershell
.\.venv\Scripts\python.exe scripts\02_download_public_dataset.py 30vnfoods
.\.venv\Scripts\python.exe scripts\02_download_public_dataset.py vietfood68
```

After download, inspect class folders and copy matching dish images into:

```text
data/classification/train/<class_name>/
data/classification/val/<class_name>/
data/classification/test/<class_name>/
```

## Train the Classifier

Train after `data/classification/` has labeled crops:

```powershell
.\.venv\Scripts\python.exe scripts\05_train_classifier.py --epochs 5 --batch-size 8
```

Notebook equivalent:

```text
notebooks/01_train_classifier.ipynb
```

Outputs:

- `models/dish_classifier.pt`
- `models/class_names.json`
- `outputs/reports/training_history.json`
- `outputs/reports/training_history.png`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix.png`

## Run Checkout Demo

After real training:

```powershell
.\.venv\Scripts\python.exe scripts\06_demo_checkout.py --image data\demo_trays\YOUR_IMAGE.jpg
```

For `thit_kho_trung`, the base price includes one egg. Extra eggs add 6,000 VND each:

```powershell
.\.venv\Scripts\python.exe scripts\06_demo_checkout.py --image data\demo_trays\YOUR_IMAGE.jpg --egg-count 2
```

Notebook equivalent:

```text
notebooks/02_demo_checkout.ipynb
```

## GitHub, Kaggle, and Colab

This project is GitHub-ready. Do not commit `.venv/`, raw data, model weights, or outputs; `.gitignore` already excludes them.

For Kaggle or Colab:

1. Clone the GitHub repo.
2. Install missing packages if needed.
3. Upload or attach dataset folders.
4. Keep the same `data/classification/` layout.
5. Run the notebooks in order.

The portable model artifact is `models/dish_classifier.pt`, but it should travel with `models/class_names.json` and `prices.csv`.

Before real training, this still tests crop + bill JSON, but marks every crop as `unknown`:

```powershell
.\.venv\Scripts\python.exe scripts\06_demo_checkout.py --image data\demo_trays\YOUR_IMAGE.jpg --model models\dish_classifier.pt
```

Bill output:

```text
outputs/bills/<image_name>_bill.json
```

## Smoke Test

This validates the local environment and pipeline using synthetic data. It is not a real dish model.

```powershell
.\.venv\Scripts\python.exe scripts\07_smoke_test.py
```

Verified smoke test result:

- trained 1 epoch on GPU;
- saved and reloaded `outputs/smoke_model.pt`;
- cropped one real teacher tray image;
- exported a bill JSON.

## Important Note

The teacher tray photos are valuable for final testing and demo, but they are not enough for real training until each dish region is cropped and labeled. For the first real model, use public dish datasets plus a small number of manually labeled crops from `Khay_com/`.
