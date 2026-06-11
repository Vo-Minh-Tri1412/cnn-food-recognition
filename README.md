# Checkout the Canteen

Local-first canteen checkout project for 11 HCMUS canteen dishes.

The current workflow is:

1. collect/import candidate food images into `data/inbox/review`;
2. review and clean data in the Data IDE;
3. keep only trusted images in `data/reviewed/<class>`;
4. regenerate `data/classification/{train,val,test}` from `data/reviewed`;
5. train the dish classifier;
6. demo tray checkout with crop regions, ignore regions, prices, and bill JSON.

## Environment

Use the local virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Main dependencies are PyTorch, torchvision, OpenCV, Pillow, pandas, scikit-learn, matplotlib, tqdm, and Jupyter.

Cloud workflow entrypoint:

```text
00_colab_kaggle_workflow.ipynb
01_colab_demo_checkout.ipynb
02_colab_gradcam_debug.ipynb
03_colab_read_reports.ipynb
```

Open these notebooks from GitHub in Google Colab or upload/import them into Kaggle. The `00` notebook is designed to clone this repo, find a packaged dataset, train, and save model/report artifacts. The other notebooks are smaller control panels for demo inference, Grad-CAM, and reading training reports.

## Core Data Tree

```text
data/inbox/review/          # images waiting for human review
data/reviewed/<class>/      # trusted pool used to build training data
data/extras/<label>/        # useful non-target labels, background, future use
data/quarantine/<reason>/   # rejected, duplicates, and label conflicts
data/classification/        # generated train/val/test dataset
data/demo/                  # demo tray images and uploads
data/archive/               # legacy/raw/frozen data
```

`data/classification` is generated output. Do not treat it as the long-term source of truth; rebuild it from `data/reviewed` when data changes.

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

Start the review and data-management UI:

```powershell
.\.venv\Scripts\python.exe scripts\21_data_ide.py --host 127.0.0.1 --port 7862
```

Open [http://127.0.0.1:7862](http://127.0.0.1:7862).

Source roots:

- `inbox_review`: new images waiting for review.
- `reviewed`: trusted images already accepted.
- `extras`: non-target labels such as background or future-use classes.
- `quarantine`: rejected, duplicate, or conflict images.
- `classification`: generated train/val/test dataset for inspection.
- `data_workspace`: any image folder under `data`.

Target roots:

- `reviewed`: accepted target-class images.
- `inbox_review`: put images back into review.
- `extras`: useful images outside the 11-class task.
- `quarantine`: rejected, duplicate, or uncertain images.

The IDE now lets you choose a source folder and a target root/label directly. After move, quarantine, future-use, or mark-done actions, processed images leave the current queue.

Useful shortcuts:

- `1-9`, `0`: select image in the current row.
- `A`: select the current row.
- `Space`: next row.
- `Shift+Space`: previous row.
- `Enter`: move selected images to the chosen target root/label.
- `Q`: quarantine selected images.
- `F`: move selected images to an extra/future-use label.
- `P`: run the current model on visible images.
- `Esc`: clear selection.

## Dedupe Reviewed

Dry-run first:

```powershell
.\.venv\Scripts\python.exe scripts\26_dedupe_reviewed.py --root data\reviewed
```

Apply safe quarantine:

```powershell
.\.venv\Scripts\python.exe scripts\26_dedupe_reviewed.py --root data\reviewed --apply
```

Policy:

- exact duplicate in the same class: keep one canonical file, quarantine the rest;
- exact or near duplicate across classes: quarantine the conflict group;
- near duplicate in the same class: threshold `8`;
- cross-class conflict: threshold `4`.

Reports are written to `outputs/reports`.

## Import And Review New Data

Put new raw images anywhere under `data/inbox`, for example:

```text
data/inbox/raw_batches/my_new_batch/raw/
```

Normalize/import a batch into the review queue:

```powershell
.\.venv\Scripts\python.exe scripts\18_import_scrape_batch_to_review.py `
  --source data\inbox\raw_batches\my_new_batch\raw `
  --class-name rau_xao `
  --pool rau_xao_my_new_batch `
  --target review `
  --dedupe-against data\reviewed `
  --dedupe-against data\classification
```

Correct images should be moved in the Data IDE into `data/reviewed/<class>`. Ambiguous images should go to `data/extras/<label>` or `data/quarantine/<reason>`.

## Search And Crawl

Search public dataset links before crawling individual images:

```powershell
.\.venv\Scripts\python.exe scripts\22_search_public_datasets.py --provider mixed
```

Crawl image candidates only into raw batch folders:

```powershell
.\.venv\Scripts\python.exe scripts\09_collect_web_images.py `
  --queries configs\green_vegetable_extra_queries.csv `
  --provider mixed `
  --out data\inbox\raw_batches\green_vegetable_batch\raw `
  --manifest outputs\reports\green_vegetable_batch_manifest.csv `
  --per-query 80 `
  --max-downloads-per-class 300 `
  --dedupe-against data\reviewed `
  --dedupe-against data\classification `
  --phash-threshold 6
```

Never crawl directly into `data/reviewed` or `data/classification`.

## Build Training Dataset

Rebuild `data/classification` from trusted reviewed data:

```powershell
.\.venv\Scripts\python.exe scripts\17_build_weighted_classification_dataset.py `
  --reviewed-source data\reviewed `
  --old-weight 1 `
  --reviewed-weight 1 `
  --cross-class-hamming 4 `
  --clear `
  --clear-all
```

Current policy: all trusted reviewed images have equal weight. Cross-class duplicates are skipped and reported.

Audit the final dataset:

```powershell
.\.venv\Scripts\python.exe scripts\20_audit_dataset_conflicts.py --root data\classification --phash-threshold 4
```

## Package For Colab Or Kaggle

Package the generated training dataset into one archive:

```powershell
.\.venv\Scripts\python.exe scripts\27_package_cloud_dataset.py
```

Outputs:

```text
outputs/cloud/classification.zip
outputs/cloud/classification.manifest.json
```

Current dataset size is about 149 MB, so it is reasonable to store on Google Drive or as a Kaggle Dataset. Prefer copying the zip to the cloud runtime and unzipping there before training; do not train directly from many small files on Drive.

Recommended locations:

```text
Google Drive:
MyDrive/canteen_checkout/datasets/classification.zip

Kaggle:
/kaggle/input/<your-dataset>/classification.zip
```

## Train

Recommended local training command:

```powershell
.\.venv\Scripts\python.exe scripts\05_train_classifier.py `
  --arch efficientnet_b0 `
  --augmentation medium `
  --epochs 6 `
  --batch-size 8 `
  --lr 0.0001 `
  --label-smoothing 0.05
```

Supported `--arch` choices:

```text
mobilenet_v3_small
mobilenet_v3_large
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
resnet18
resnet50
convnext_tiny
```

For Colab GPU, start with `efficientnet_b2`. If it runs out of memory, use
`efficientnet_b0` or reduce `--batch-size` to `4`.

Supported `--augmentation` choices:

```text
light
medium
strong
```

The Colab training notebook defaults to `strong`. If validation loss becomes
unstable, switch to `medium`.

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

Demo frontend files:

```text
templates/demo_checkout.html
static/demo_checkout.css
static/demo_checkout.js
```

CLI demo:

```powershell
.\.venv\Scripts\python.exe scripts\06_demo_checkout.py --image data\demo\YOUR_IMAGE.jpg
```

Manual crop helper:

```powershell
.\.venv\Scripts\python.exe scripts\04_crop_tray.py --image data\demo\YOUR_IMAGE.jpg --interactive --save-regions configs\my_regions.json
```

## Notebook

Use the root cloud workflow notebooks:

```text
00_colab_kaggle_workflow.ipynb     # train on Colab/Kaggle/Drive
01_colab_demo_checkout.ipynb       # upload one tray image and run checkout
02_colab_gradcam_debug.ipynb       # inspect weak classes with Grad-CAM
03_colab_read_reports.ipynb        # read classification report/loss/confusion matrix
```

The `00` notebook documents which script to run for local packaging, Google Drive dataset loading, Kaggle input loading, training, report reading, and artifact export. The old `notebooks/` folder now only contains a pointer README.
