from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from canteen_checkout.config import CLASSIFICATION_DIR, DEFAULT_MODEL_PATH, DISH_CLASSES, REPORTS_DIR
from canteen_checkout.io_utils import IMAGE_EXTENSIONS, save_class_names
from canteen_checkout.model import build_classifier, eval_transforms, load_checkpoint, resolve_device, save_checkpoint, train_transforms


class FixedClassImageDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.root = root
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def counts_by_class(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def class_weight_tensor(counts: dict[str, int], class_names: list[str], device: torch.device) -> torch.Tensor:
    total = sum(counts.values())
    weights = []
    for class_name in class_names:
        count = max(counts.get(class_name, 0), 1)
        weights.append(total / (len(class_names) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def plot_history(history: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dish classifier.")
    parser.add_argument("--data", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--arch", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--no-weighted-loss", action="store_true", help="Disable class-balanced loss weights.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-empty-val", action="store_true")
    args = parser.parse_args()

    train_ds = FixedClassImageDataset(args.data / "train", DISH_CLASSES, train_transforms(args.image_size))
    val_ds = FixedClassImageDataset(args.data / "val", DISH_CLASSES, eval_transforms(args.image_size))
    test_ds = FixedClassImageDataset(args.data / "test", DISH_CLASSES, eval_transforms(args.image_size))

    print("train counts:", train_ds.counts_by_class())
    print("val counts:", val_ds.counts_by_class())
    print("test counts:", test_ds.counts_by_class())

    if len(train_ds) == 0:
        raise SystemExit("No training images found. Put labeled crops under data/classification/train/<class_name>/")
    if len(val_ds) == 0 and not args.allow_empty_val:
        raise SystemExit("No validation images found. Add val images or pass --allow-empty-val for a smoke run.")

    device = resolve_device()
    print(f"device: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds if len(val_ds) else train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds if len(test_ds) else val_ds if len(val_ds) else train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_classifier(len(DISH_CLASSES), pretrained=not args.no_pretrained, arch=args.arch).to(device)
    loss_weights = None if args.no_weighted_loss else class_weight_tensor(train_ds.counts_by_class(), DISH_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -1.0
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.model_out,
                model,
                DISH_CLASSES,
                args.image_size,
                metadata={"best_val_acc": best_val_acc, "epoch": epoch, "arch": args.arch},
                arch=args.arch,
            )

    save_class_names()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, REPORTS_DIR / "training_history.png")

    if args.model_out.exists():
        model, _, _, checkpoint = load_checkpoint(args.model_out, device)
        print(f"Loaded best checkpoint for test: epoch={checkpoint.get('metadata', {}).get('epoch')}, val_acc={checkpoint.get('metadata', {}).get('best_val_acc')}")
    y_true, y_pred = collect_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, labels=list(range(len(DISH_CLASSES))), target_names=DISH_CLASSES, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DISH_CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISH_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(report)
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
