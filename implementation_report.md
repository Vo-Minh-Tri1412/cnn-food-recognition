# Implementation Report - Checkout the Canteen

## What Was Implemented

- Created a local Python 3.12 environment in `.venv/`.
- Installed PyTorch CUDA, TorchVision, OpenCV, Ultralytics, pandas, matplotlib, scikit-learn, JupyterLab, and helper packages.
- Verified PyTorch can use the local NVIDIA GeForce RTX 3050 Laptop GPU.
- Standardized project folders under `data/`, `models/`, and `outputs/`.
- Copied 56 teacher tray images from `Khay_com/` into `data/raw_teacher_trays/`.
- Copied 6 demo tray images into `data/demo_trays/`.
- Created `models/class_names.json` for the 11 classes from the PDF.
- Implemented scripts for preparation, inventory, dataset notes/download, crop, train, checkout demo, and smoke test.
- Added three notebook wrappers for GitHub/Kaggle/Colab use:
  - `notebooks/00_setup_and_inventory.ipynb`
  - `notebooks/01_train_classifier.ipynb`
  - `notebooks/02_demo_checkout.ipynb`

## Current Data State

- Teacher tray photos: 56 images.
- Demo tray photos: 6 images.
- Classification dataset: 0 real labeled dish images right now.

This is expected. The teacher images are full tray photos, not labeled dish crops. They should be used as test/demo images first, then manually cropped and labeled only when we want to fine-tune on real canteen data.

## Environment Test Results

The local machine passed the environment smoke test:

- PyTorch version: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- OpenCV version: `4.13.0`
- Ultralytics version: `8.4.61`

The smoke test also:

- created a tiny synthetic 11-class dataset;
- trained MobileNetV3 Small for 1 epoch on GPU;
- saved `outputs/smoke_model.pt`;
- loaded the model;
- cropped one real teacher tray image;
- exported a bill JSON.

The smoke model is only for infrastructure testing and must not be interpreted as a real food recognizer.

## Implemented Pipeline

```text
tray image
  -> crop dish regions
  -> classify each crop with dish_classifier.pt
  -> read price from prices.csv
  -> export console bill and JSON bill
```

The default crop mode uses an approximate five-compartment tray template. For better results, use the interactive crop mode and save a JSON region file.

## GitHub/Kaggle/Colab Portability

The project can be pushed to GitHub now. The core logic is intentionally kept in Python scripts so it is easier to version, test, and reuse across environments. The notebooks call those scripts instead of duplicating the implementation.

Do not commit local-only artifacts:

- `.venv/`
- `data/raw_teacher_trays/`
- `data/demo_trays/`
- `data/classification/`
- `outputs/`
- large model weights

These are already covered by `.gitignore`.

On Kaggle or Colab, clone the repo, attach/upload the dataset, install any missing packages, and run the notebooks in order.

## Next Real Work

1. Build the real classification dataset:
   - collect public dish images from 30VNFoods/VietFood68;
   - manually crop and label teacher images for domain-specific examples;
   - place images under `data/classification/{train,val,test}/<class_name>/`.

2. Train the real classifier:
   - run `scripts/05_train_classifier.py`;
   - inspect accuracy, confusion matrix, and misclassified classes.

3. Run the real demo:
   - run `scripts/06_demo_checkout.py` on 3-5 teacher tray images;
   - check crop quality and bill correctness.

## Known Limitations

- There is no real trained classifier yet because no labeled dish-crop dataset has been added.
- The template crop is approximate and can fail on angled, multi-tray, or partially cropped photos.
- YOLO/object detection is installed but not trained yet; it should be added only after enough bounding-box data exists.
