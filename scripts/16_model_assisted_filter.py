from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canteen_checkout.config import (
    DISH_CLASSES,
    DOWNLOADS_DIR,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from canteen_checkout.data_quality import assess_image, hamming_distance_hex, open_rgb_image
from canteen_checkout.model import (
    build_classifier,
    eval_transforms,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_transforms,
)


POOL_ALLOWED_CLASSES = {
    "ca_hu_kho": ["ca_hu_kho"],
    "com_trang": ["com_trang"],
    "trung_chien": ["trung_chien"],
    "thit_kho_or_thit_kho_trung": ["thit_kho", "thit_kho_trung"],
    "canh_chua_unknown": ["canh_chua_co_ca", "canh_chua_khong_ca"],
    "rau_xao_or_canh_rau": ["rau_xao", "canh_rau"],
    "protein_grid_review": ["thit_kho", "thit_kho_trung", "ca_hu_kho", "suon_nuong", "dau_hu_sot_ca", "trung_chien"],
    "unknown_food_crops": list(DISH_CLASSES),
}

AMBIGUOUS_POOLS = {
    "thit_kho_or_thit_kho_trung",
    "canh_chua_unknown",
    "rau_xao_or_canh_rau",
    "protein_grid_review",
    "unknown_food_crops",
}


@dataclass(frozen=True)
class SeedSample:
    path: Path
    label: int
    phash: str


@dataclass(frozen=True)
class CandidatePrediction:
    path: Path
    pool: str
    suggested_class: str
    top1_class: str
    top1_confidence: float
    top2_class: str
    top2_confidence: float
    margin: float
    decision: str
    reason: str
    target_class: str
    phash: str
    duplicate_of: str
    duplicate_distance: str


class PathImageDataset(Dataset):
    def __init__(self, samples: list[SeedSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = open_rgb_image(sample.path)
        image = ImageOps.exif_transpose(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


class CandidateDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = open_rgb_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def latest_external_staging() -> Path:
    root = DOWNLOADS_DIR / "external_staging"
    candidates = sorted((p for p in root.glob("external_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No external staging folders found in {root}")
    return candidates[0]


def latest_merge_processed() -> Path:
    root = DOWNLOADS_DIR / "merge_batches"
    candidates = sorted(
        (p / "processed" for p in root.glob("merge_*") if (p / "processed").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No processed merge folders found in {root}")
    return candidates[0]


def split_root_has_splits(root: Path) -> bool:
    return any((root / split).is_dir() for split in ["train", "val", "test"])


def seed_images_by_class(root: Path, split: str | None = None) -> dict[str, list[Path]]:
    base = root / split if split and (root / split).is_dir() else root
    result: dict[str, list[Path]] = {}
    for class_name in DISH_CLASSES:
        class_dir = base / class_name
        result[class_name] = list_images(class_dir)
    return result


def is_duplicate(phash: str, seen: list[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, previous) <= threshold for previous in seen)


def make_seed_sample(path: Path, label: int) -> SeedSample | None:
    _, metrics, _ = assess_image(path)
    if metrics is None:
        return None
    return SeedSample(path=path, label=label, phash=metrics.phash)


def dedupe_seed_paths(paths: list[Path], label: int, threshold: int) -> tuple[list[SeedSample], int]:
    kept: list[SeedSample] = []
    seen: list[str] = []
    skipped = 0
    for path in paths:
        sample = make_seed_sample(path, label)
        if sample is None:
            skipped += 1
            continue
        if is_duplicate(sample.phash, seen, threshold):
            skipped += 1
            continue
        seen.append(sample.phash)
        kept.append(sample)
    return kept, skipped


def split_seed_samples(
    root: Path,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    dedupe_threshold: int,
) -> tuple[list[SeedSample], list[SeedSample], list[SeedSample], dict[str, dict[str, int]]]:
    stats: dict[str, dict[str, int]] = {}
    train_samples: list[SeedSample] = []
    val_samples: list[SeedSample] = []
    test_samples: list[SeedSample] = []
    rng = random.Random(seed)

    if split_root_has_splits(root):
        for split_name, target in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            for class_name, paths in seed_images_by_class(root, split_name).items():
                label = DISH_CLASSES.index(class_name)
                kept, skipped = dedupe_seed_paths(paths, label, dedupe_threshold)
                target.extend(kept)
                stats.setdefault(class_name, {"train": 0, "val": 0, "test": 0, "skipped": 0})
                stats[class_name][split_name] += len(kept)
                stats[class_name]["skipped"] += skipped
        return train_samples, val_samples, test_samples, stats

    for class_name, paths in seed_images_by_class(root).items():
        label = DISH_CLASSES.index(class_name)
        kept, skipped = dedupe_seed_paths(paths, label, dedupe_threshold)
        rng.shuffle(kept)
        n = len(kept)
        val_count = int(round(n * val_ratio))
        test_count = int(round(n * test_ratio))
        if n >= 3:
            val_count = max(1, val_count)
            test_count = max(1, test_count)
        while val_count + test_count >= n and (val_count or test_count):
            if val_count >= test_count and val_count:
                val_count -= 1
            elif test_count:
                test_count -= 1
        train_count = n - val_count - test_count
        train_samples.extend(kept[:train_count])
        val_samples.extend(kept[train_count : train_count + val_count])
        test_samples.extend(kept[train_count + val_count :])
        stats[class_name] = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "skipped": skipped,
        }
    return train_samples, val_samples, test_samples, stats


def dedupe_across_splits(
    train_samples: list[SeedSample],
    val_samples: list[SeedSample],
    test_samples: list[SeedSample],
    threshold: int,
) -> tuple[list[SeedSample], list[SeedSample], list[SeedSample], int]:
    seen: list[str] = []
    skipped = 0

    def keep_unique(samples: list[SeedSample]) -> list[SeedSample]:
        nonlocal skipped
        kept: list[SeedSample] = []
        for sample in samples:
            if is_duplicate(sample.phash, seen, threshold):
                skipped += 1
                continue
            seen.append(sample.phash)
            kept.append(sample)
        return kept

    return keep_unique(train_samples), keep_unique(val_samples), keep_unique(test_samples), skipped


def reviewed_seed_root(args) -> Path:
    staging_root = args.staging or latest_external_staging()
    return staging_root / "reviewed"


def seed_roots_from_args(args) -> list[Path]:
    seed_roots = [args.seed_source or latest_merge_processed()]
    seed_roots.extend(args.extra_seed_source or [])
    if args.include_staging_reviewed:
        seed_roots.append(reviewed_seed_root(args))
    return seed_roots


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False, desc="train"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False, desc="eval"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_seed_model(args) -> tuple[Path, dict[str, object]]:
    seed_roots = seed_roots_from_args(args)
    train_samples: list[SeedSample] = []
    val_samples: list[SeedSample] = []
    test_samples: list[SeedSample] = []
    seed_stats: dict[str, object] = {}
    for seed_root in seed_roots:
        if not seed_root.exists():
            print(f"Skipping missing seed source: {seed_root}")
            continue
        source_train, source_val, source_test, source_stats = split_seed_samples(
            seed_root,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            dedupe_threshold=args.seed_dedupe_threshold,
        )
        train_samples.extend(source_train)
        val_samples.extend(source_val)
        test_samples.extend(source_test)
        seed_stats[relative_or_absolute(seed_root)] = source_stats

    train_samples, val_samples, test_samples, cross_source_skipped = dedupe_across_splits(
        train_samples,
        val_samples,
        test_samples,
        args.seed_dedupe_threshold,
    )
    if not train_samples:
        raise SystemExit(f"No seed training images found in: {', '.join(str(path) for path in seed_roots)}")
    if not val_samples:
        val_samples = train_samples

    device = resolve_device()
    print("Seed sources:")
    for seed_root in seed_roots:
        print(f"- {seed_root}")
    print(f"Seed samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    print(f"Device: {device}")

    train_loader = DataLoader(
        PathImageDataset(train_samples, train_transforms(args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PathImageDataset(val_samples, eval_transforms(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_classifier(len(DISH_CLASSES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best_val_acc = -1.0
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
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
                metadata={
                    "purpose": "model_assisted_dataset_filter",
                    "seed_sources": [relative_or_absolute(path) for path in seed_roots],
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
            )

    return args.model_out, {
        "seed_sources": [relative_or_absolute(path) for path in seed_roots],
        "seed_stats": seed_stats,
        "cross_source_duplicates_skipped": cross_source_skipped,
        "history": history,
    }


def load_external_manifest(staging_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = staging_root / "reports" / "external_import_manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    if not manifest_path.exists():
        return rows
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            output_path = row.get("output_path", "")
            if not output_path:
                continue
            path = Path(output_path)
            resolved = path if path.is_absolute() else PROJECT_ROOT / path
            rows[str(resolved.resolve())] = row
    return rows


def candidate_paths(staging_root: Path) -> list[Path]:
    review_root = staging_root / "review"
    if not review_root.exists():
        raise FileNotFoundError(f"Review folder not found: {review_root}")
    return sorted(p for p in review_root.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def nearest_duplicate(phash: str, references: list[tuple[str, Path]], threshold: int) -> tuple[Path, int] | None:
    best_path: Path | None = None
    best_distance: int | None = None
    for old_phash, old_path in references:
        distance = hamming_distance_hex(phash, old_phash)
        if distance <= threshold and (best_distance is None or distance < best_distance):
            best_path = old_path
            best_distance = distance
    if best_path is None or best_distance is None:
        return None
    return best_path, best_distance


def seed_phash_references(args) -> list[tuple[str, Path]]:
    references: list[tuple[str, Path]] = []
    for seed_root in seed_roots_from_args(args):
        if not seed_root.exists():
            continue
        for path in list_images(seed_root):
            _, metrics, _ = assess_image(path)
            if metrics is not None:
                references.append((metrics.phash, path))
    return references


def duplicate_overrides(
    paths: list[Path],
    args,
) -> tuple[dict[str, tuple[str, str, str]], dict[str, int], dict[str, str]]:
    if args.disable_duplicate_gate:
        return {}, {"seed_references": 0, "duplicate_seed": 0, "duplicate_candidate": 0}, {}

    seed_references = seed_phash_references(args)
    candidate_references: list[tuple[str, Path]] = []
    overrides: dict[str, tuple[str, str, str]] = {}
    phashes: dict[str, str] = {}
    counts = {"seed_references": len(seed_references), "duplicate_seed": 0, "duplicate_candidate": 0}

    for path in paths:
        _, metrics, _ = assess_image(path)
        if metrics is None:
            continue
        resolved = str(path.resolve())
        phashes[resolved] = metrics.phash
        seed_duplicate = nearest_duplicate(metrics.phash, seed_references, args.duplicate_hamming)
        if seed_duplicate is not None:
            duplicate_path, distance = seed_duplicate
            overrides[resolved] = ("duplicate_seed", relative_or_absolute(duplicate_path), str(distance))
            counts["duplicate_seed"] += 1
            continue
        candidate_duplicate = nearest_duplicate(metrics.phash, candidate_references, args.duplicate_hamming)
        if candidate_duplicate is not None:
            duplicate_path, distance = candidate_duplicate
            overrides[resolved] = ("duplicate_candidate", relative_or_absolute(duplicate_path), str(distance))
            counts["duplicate_candidate"] += 1
            continue
        candidate_references.append((metrics.phash, path))
    return overrides, counts, phashes


def decide_prediction(
    *,
    pool: str,
    suggested_class: str,
    top1_class: str,
    top1_confidence: float,
    top2_class: str,
    top2_confidence: float,
    auto_accept_confidence: float,
    review_confidence: float,
    reject_confidence: float,
    min_margin: float,
    strict_pool: bool,
) -> tuple[str, str, str]:
    allowed = POOL_ALLOWED_CLASSES.get(pool, list(DISH_CLASSES))
    is_allowed = top1_class in allowed
    matches_suggested = not suggested_class or suggested_class == top1_class
    margin = top1_confidence - top2_confidence

    if strict_pool and not is_allowed:
        if top1_confidence < review_confidence:
            return "model_rejected", "outside_pool_and_low_confidence", ""
        return "ambiguous_review", "outside_pool_hint", ""

    if top1_confidence < reject_confidence:
        return "model_rejected", "low_confidence", ""

    if pool in AMBIGUOUS_POOLS and margin < min_margin:
        return "ambiguous_review", "small_margin_in_ambiguous_pool", ""

    if top1_confidence >= auto_accept_confidence and margin >= min_margin and is_allowed and matches_suggested:
        return "auto_accepted", "high_confidence", top1_class

    if top1_confidence >= auto_accept_confidence and margin >= min_margin and is_allowed and pool in AMBIGUOUS_POOLS:
        return "auto_accepted", "high_confidence_resolved_ambiguous_pool", top1_class

    if top1_confidence < review_confidence:
        return "needs_review", "medium_or_low_confidence", ""

    if is_allowed:
        return "needs_review", "plausible_but_not_auto", ""

    return "ambiguous_review", "model_disagrees_with_pool", ""


@torch.no_grad()
def predict_candidates(args, model_path: Path) -> tuple[list[CandidatePrediction], dict[str, object]]:
    staging_root = args.staging or latest_external_staging()
    manifest = load_external_manifest(staging_root)
    paths = candidate_paths(staging_root)
    duplicate_map, duplicate_counts, candidate_phashes = duplicate_overrides(paths, args)
    device = resolve_device()
    model, class_names, image_size, checkpoint = load_checkpoint(model_path, device)
    transform = eval_transforms(image_size)
    loader = DataLoader(CandidateDataset(paths, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions: list[CandidatePrediction] = []
    print(f"Staging: {staging_root}")
    print(f"Candidates: {len(paths)}")
    print(f"Model: {model_path}")
    if not args.disable_duplicate_gate:
        print(f"Duplicate seed references: {duplicate_counts['seed_references']}")
        print(f"Duplicate gate: seed={duplicate_counts['duplicate_seed']}, candidate={duplicate_counts['duplicate_candidate']}")

    for images, path_strings in tqdm(loader, desc="predict"):
        images = images.to(device)
        probabilities = torch.softmax(model(images), dim=1).cpu()
        values, indices = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1)
        for i, path_string in enumerate(path_strings):
            path = Path(path_string)
            pool = path.parent.name
            meta = manifest.get(str(path.resolve()), {})
            suggested_class = meta.get("suggested_class", "")
            top1_idx = int(indices[i, 0].item())
            top2_idx = int(indices[i, 1].item()) if probabilities.shape[1] > 1 else top1_idx
            top1_class = class_names[top1_idx]
            top2_class = class_names[top2_idx]
            top1_confidence = float(values[i, 0].item())
            top2_confidence = float(values[i, 1].item()) if probabilities.shape[1] > 1 else 0.0
            decision, reason, target_class = decide_prediction(
                pool=pool,
                suggested_class=suggested_class,
                top1_class=top1_class,
                top1_confidence=top1_confidence,
                top2_class=top2_class,
                top2_confidence=top2_confidence,
                auto_accept_confidence=args.auto_accept_confidence,
                review_confidence=args.review_confidence,
                reject_confidence=args.reject_confidence,
                min_margin=args.min_margin,
                strict_pool=not args.loose_pool,
            )
            duplicate_reason, duplicate_of, duplicate_distance = duplicate_map.get(str(path.resolve()), ("", "", ""))
            if duplicate_reason:
                decision = "model_rejected"
                reason = duplicate_reason
                target_class = ""
            predictions.append(
                CandidatePrediction(
                    path=path,
                    pool=pool,
                    suggested_class=suggested_class,
                    top1_class=top1_class,
                    top1_confidence=top1_confidence,
                    top2_class=top2_class,
                    top2_confidence=top2_confidence,
                    margin=top1_confidence - top2_confidence,
                    decision=decision,
                    reason=reason,
                    target_class=target_class,
                    phash=candidate_phashes.get(str(path.resolve()), ""),
                    duplicate_of=duplicate_of,
                    duplicate_distance=duplicate_distance,
                )
            )

    summary = {
        "staging": relative_or_absolute(staging_root),
        "model": relative_or_absolute(model_path),
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "decision_counts": dict(Counter(row.decision for row in predictions)),
        "reason_counts": dict(Counter(row.reason for row in predictions)),
        "pool_counts": {pool: dict(counter) for pool, counter in decision_counts_by_pool(predictions).items()},
        "duplicate_gate": duplicate_counts,
    }
    return predictions, summary


def decision_counts_by_pool(predictions: list[CandidatePrediction]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in predictions:
        counts[row.pool][row.decision] += 1
    return counts


def unique_destination(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        next_candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        idx += 1


def copy_predictions(
    predictions: list[CandidatePrediction],
    *,
    staging_root: Path,
    apply: bool,
    promote_auto: bool,
    keep_existing_output: bool,
) -> Counter:
    counts: Counter = Counter()
    assisted_root = staging_root / "model_assisted"
    if apply and assisted_root.exists() and not keep_existing_output:
        shutil.rmtree(assisted_root)
    for row in predictions:
        if row.decision == "auto_accepted":
            folder = assisted_root / "auto_accepted" / row.target_class
            filename = f"{row.pool}_{row.path.name}"
        elif row.decision == "model_rejected":
            folder = assisted_root / "model_rejected" / row.reason / row.pool
            filename = row.path.name
        elif row.decision == "ambiguous_review":
            folder = assisted_root / "ambiguous_review" / row.pool
            filename = row.path.name
        else:
            folder = assisted_root / "needs_review" / row.pool
            filename = row.path.name

        if apply:
            target = unique_destination(folder, filename)
            shutil.copy2(row.path, target)
            if promote_auto and row.decision == "auto_accepted" and row.target_class:
                reviewed_target = unique_destination(staging_root / "reviewed" / row.target_class, f"auto_{row.pool}_{row.path.name}")
                shutil.copy2(row.path, reviewed_target)
        counts[row.decision] += 1
    return counts


def write_prediction_report(predictions: list[CandidatePrediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "pool",
        "suggested_class",
        "top1_class",
        "top1_confidence",
        "top2_class",
        "top2_confidence",
        "margin",
        "decision",
        "reason",
        "target_class",
        "phash",
        "duplicate_of",
        "duplicate_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "source_path": relative_or_absolute(row.path),
                    "pool": row.pool,
                    "suggested_class": row.suggested_class,
                    "top1_class": row.top1_class,
                    "top1_confidence": f"{row.top1_confidence:.6f}",
                    "top2_class": row.top2_class,
                    "top2_confidence": f"{row.top2_confidence:.6f}",
                    "margin": f"{row.margin:.6f}",
                    "decision": row.decision,
                    "reason": row.reason,
                    "target_class": row.target_class,
                    "phash": row.phash,
                    "duplicate_of": row.duplicate_of,
                    "duplicate_distance": row.duplicate_distance,
                }
            )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seed classifier and use it to pre-filter external dataset candidates.")
    parser.add_argument("--staging", type=Path, default=None, help="External staging folder. Defaults to newest data/downloads/external_staging/external_*.")
    parser.add_argument("--seed-source", type=Path, default=None, help="Class-labeled seed dataset. Defaults to newest data/downloads/merge_batches/merge_*/processed.")
    parser.add_argument("--extra-seed-source", type=Path, action="append", default=[], help="Additional class-labeled seed dataset. Can be repeated.")
    parser.add_argument("--include-staging-reviewed", action="store_true", help="Also train from staging/reviewed/<11 dish classes>. reviewed_extra is ignored.")
    parser.add_argument("--model-out", type=Path, default=MODELS_DIR / "data_filter_classifier.pt")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if --model-out already exists.")
    parser.add_argument("--skip-train", action="store_true", help="Use existing --model-out without training.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed-dedupe-threshold", type=int, default=4)
    parser.add_argument("--duplicate-hamming", type=int, default=8, help="pHash Hamming threshold for seed/candidate duplicate gate before auto accept.")
    parser.add_argument("--disable-duplicate-gate", action="store_true", help="Do not reject near-duplicates of seed or earlier candidate images.")
    parser.add_argument("--auto-accept-confidence", type=float, default=0.92)
    parser.add_argument("--review-confidence", type=float, default=0.55)
    parser.add_argument("--reject-confidence", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--loose-pool", action="store_true", help="Allow predictions outside the source pool hints.")
    parser.add_argument("--apply", action="store_true", help="Copy predictions into staging/model_assisted. Default only writes reports.")
    parser.add_argument("--promote-auto", action="store_true", help="Also copy auto accepted images into staging/reviewed/<class>.")
    parser.add_argument("--keep-existing-output", action="store_true", help="Do not clear staging/model_assisted before copying a new grouped run.")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR / "model_assisted_filter")
    args = parser.parse_args()

    if args.promote_auto and not args.apply:
        parser.error("--promote-auto requires --apply")
    if args.skip_train and not args.model_out.exists():
        parser.error(f"--skip-train requested but model does not exist: {args.model_out}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_report_dir = args.report_dir / f"run_{timestamp}"
    model_path = args.model_out
    training_payload: dict[str, object] = {}

    if not args.skip_train and (args.force_train or not model_path.exists()):
        model_path, training_payload = train_seed_model(args)
    else:
        print(f"Using existing model: {model_path}")

    predictions, summary = predict_candidates(args, model_path)
    staging_root = args.staging or latest_external_staging()
    copy_counts = copy_predictions(
        predictions,
        staging_root=staging_root,
        apply=args.apply,
        promote_auto=args.promote_auto,
        keep_existing_output=args.keep_existing_output,
    )

    summary.update(
        {
            "apply": args.apply,
            "promote_auto": args.promote_auto,
            "copy_counts": dict(copy_counts),
            "thresholds": {
                "duplicate_hamming": args.duplicate_hamming,
                "auto_accept_confidence": args.auto_accept_confidence,
                "review_confidence": args.review_confidence,
                "reject_confidence": args.reject_confidence,
                "min_margin": args.min_margin,
            },
            "training": training_payload,
            "outputs": {
                "report_dir": relative_or_absolute(run_report_dir),
                "staging_model_assisted": relative_or_absolute(staging_root / "model_assisted"),
            },
        }
    )
    write_prediction_report(predictions, run_report_dir / "model_suggestions.csv")
    write_json(run_report_dir / "summary.json", summary)
    write_prediction_report(predictions, staging_root / "reports" / "model_suggestions.csv")
    write_json(staging_root / "reports" / "model_assisted_summary.json", summary)

    latest_dir = args.report_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_report(predictions, latest_dir / "model_suggestions.csv")
    write_json(latest_dir / "summary.json", summary)

    print("Decision counts:", summary["decision_counts"])
    print("Report:", run_report_dir)
    if args.apply:
        print("Copied grouped candidates to:", staging_root / "model_assisted")
        if args.promote_auto:
            print("Auto accepted images also copied to:", staging_root / "reviewed")
    else:
        print("No images copied. Re-run with --apply after checking the report.")


if __name__ == "__main__":
    main()
