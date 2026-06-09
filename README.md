# Checkout the Canteen

Local-first canteen checkout project for 11 HCMUS canteen dishes.

The current workflow is:

1. collect/import candidate food images into isolated review folders;
2. review and clean data in the Data IDE;
3. rebuild `data/classification/{train,val,test}`;
4. train the dish classifier;
5. demo tray checkout with crop regions, ignore regions, prices, and bill JSON.

## Environment

Use the local virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Main dependencies are PyTorch, torchvision, OpenCV, Pillow, pandas, scikit-learn, matplotlib, tqdm, and Jupyter.

## Core Folders

```text
data/classification/                         # final train/val/test dataset
data/downloads/external_staging/external_*/  # imported public/Roboflow datasets
data/downloads/scrape_batches/               # raw web crawl batches
data/quarantine/                             # rejected or future-use images from Data IDE
models/dish_classifier.pt                    # trained model
models/class_names.json                      # class order
outputs/reports/                             # training/audit/action reports
```

Large data, model weights, and outputs are intentionally ignored by Git.

## The 11 Billing Classes

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

Prices live in `prices.csv`. For `thit_kho_trung`, one egg is included in the base price; extra eggs add 6,000 VND each.

## Data IDE

This is the main review and data-management UI:

```powershell
.\.venv\Scripts\python.exe scripts\21_data_ide.py --host 127.0.0.1 --port 7862
```

Open [http://127.0.0.1:7862](http://127.0.0.1:7862).

Root behavior:

- `classification`: shows `split` and `class` filters.
- `external_review`, `external_reviewed`, `quarantine`: show direct `Folder / pool` filters.

Useful shortcuts:

- `1-9`, `0`: select image in the current row.
- `A`: select the current row.
- `Space`: next row.
- `Shift+Space`: previous row.
- `Enter`: move selected images to the target class.
- `Q`: quarantine selected images.
- `F`: move selected images to future-use quarantine.
- `P`: run the current model on visible images.
- `Esc`: clear selection.

After move/quarantine/future-use, the queue refreshes and processed images leave the current folder.

## Import External Datasets

Put downloaded datasets under `data/downloads/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\14_import_external_datasets.py
```

This creates review pools under:

```text
data/downloads/external_staging/external_<timestamp>/review/
```

Use the Data IDE to move correct images into:

```text
data/downloads/external_staging/external_<timestamp>/reviewed/<class_name>/
```

Ambiguous pools such as `protein_grid_review`, `unknown_food_crops`, `canh_chua_unknown`, or `thit_kho_or_thit_kho_trung` must be reviewed before training.

## Search and Crawl

Search public dataset links before crawling individual images:

```powershell
.\.venv\Scripts\python.exe scripts\22_search_public_datasets.py --provider mixed
```

Crawl image candidates only into raw batch folders:

```powershell
.\.venv\Scripts\python.exe scripts\09_collect_web_images.py `
  --queries configs\green_vegetable_extra_queries.csv `
  --provider mixed `
  --out data\downloads\scrape_batches\new_batch\raw `
  --manifest data\downloads\scrape_batches\new_batch\scraped_manifest.csv `
  --per-query 80 `
  --max-downloads-per-class 300 `
  --dedupe-against data\classification `
  --dedupe-against data\downloads `
  --phash-threshold 6
```

Normalize/import a crawl batch into a review pool:

```powershell
.\.venv\Scripts\python.exe scripts\18_import_scrape_batch_to_review.py `
  --source data\downloads\scrape_batches\new_batch\raw `
  --class-name rau_xao `
  --pool rau_xao_new_batch `
  --target review `
  --dedupe-against data\classification
```

## Build Training Dataset

Rebuild `data/classification` from curated old data and reviewed external data:

```powershell
.\.venv\Scripts\python.exe scripts\17_build_weighted_classification_dataset.py `
  --old-weight 1 `
  --reviewed-weight 1 `
  --cross-class-hamming 4 `
  --clear
```

Current policy: `old` and `reviewed` are trusted equally. Cross-class duplicates are skipped and reported.

Audit the final dataset:

```powershell
.\.venv\Scripts\python.exe scripts\20_audit_dataset_conflicts.py --root data\classification --phash-threshold 4
```

## Train

Recommended local training command:

```powershell
.\.venv\Scripts\python.exe scripts\05_train_classifier.py `
  --arch efficientnet_b0 `
  --epochs 6 `
  --batch-size 8 `
  --lr 0.0001 `
  --label-smoothing 0.05
```

Outputs:

```text
models/dish_classifier.pt
models/class_names.json
outputs/reports/training_history.json
outputs/reports/training_history.png
outputs/reports/classification_report.txt
outputs/reports/confusion_matrix.png
```

## Demo Checkout

Browser demo:

```powershell
.\.venv\Scripts\python.exe scripts\19_demo_checkout_app.py --host 127.0.0.1 --port 7861
```

Open [http://127.0.0.1:7861](http://127.0.0.1:7861).

CLI demo:

```powershell
.\.venv\Scripts\python.exe scripts\06_demo_checkout.py --image data\demo_trays\YOUR_IMAGE.jpg
```

Manual crop helper:

```powershell
.\.venv\Scripts\python.exe scripts\04_crop_tray.py --image data\demo_trays\YOUR_IMAGE.jpg --interactive --save-regions configs\my_regions.json
```

## Notebook

Use the Vietnamese workflow notebook:

```text
notebooks/00_workflow_tieng_viet.ipynb
```

It documents which script to run for data audit, Data IDE, crawl/search, import, build, train, report reading, and demo.
