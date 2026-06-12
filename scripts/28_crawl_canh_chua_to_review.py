from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import torch
from ddgs import DDGS
from PIL import Image, ImageOps

from canteen_checkout.config import (
    CLASSIFICATION_DIR,
    DATA_DIR,
    DEFAULT_MODEL_PATH,
    DISH_CLASSES,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    REPORTS_DIR,
    REVIEW_INBOX_DIR,
    REVIEWED_DIR,
)
from canteen_checkout.data_quality import (
    ImageMetrics,
    blur_score,
    brightness_mean,
    hamming_distance_hex,
    normalize_image,
    perceptual_hash,
    quality_reasons,
)
from canteen_checkout.io_utils import list_images
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device


CANH_CHUA_CLASSES = {"canh_chua_co_ca", "canh_chua_khong_ca"}
NEGATIVE_TERMS = ["-logo", "-icon", "-emoji", "-clipart", "-vector", "-cartoon"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}
MANIFEST_FIELDS = [
    "status",
    "reason",
    "suggested_class",
    "target_pool",
    "query",
    "provider",
    "source_url",
    "raw_path",
    "output_path",
    "deleted_raw",
    "top1_class",
    "top1_confidence",
    "top2_class",
    "top2_confidence",
    "margin",
    "duplicate_of",
    "duplicate_distance",
    "sha256",
    "phash",
    "width",
    "height",
    "aspect_ratio",
    "brightness",
    "blur_score",
]


@dataclass(frozen=True)
class QueryRow:
    class_name: str
    query: str


@dataclass(frozen=True)
class ReferenceImage:
    phash: str
    path: Path


@dataclass(frozen=True)
class Prediction:
    top1_class: str = ""
    top1_confidence: float = 0.0
    top2_class: str = ""
    top2_confidence: float = 0.0

    @property
    def margin(self) -> float:
        return self.top1_confidence - self.top2_confidence


def relative_or_absolute(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_queries(path: Path) -> list[QueryRow]:
    rows: list[QueryRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            class_name = row["class_name"].strip()
            query = row["query"].strip()
            if class_name not in CANH_CHUA_CLASSES:
                raise ValueError(f"Query class must be one of {sorted(CANH_CHUA_CLASSES)}: {class_name}")
            if query:
                rows.append(QueryRow(class_name=class_name, query=query))
    return rows


def refined_query(query: str, strict_phrase: bool) -> str:
    text = query.strip()
    if strict_phrase and not (text.startswith('"') and text.endswith('"')):
        text = f'"{text}"'
    tokens = set(text.split())
    missing = [term for term in NEGATIVE_TERMS if term not in tokens]
    return " ".join([text, *missing]).strip()


def find_bing_image_urls(query: str, max_urls: int) -> list[str]:
    url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1"
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    html = response.text
    urls: list[str] = []
    for pattern in [r'"murl":"(.*?)"', r"mediaurl=([^&\"']+)"]:
        for match in re.finditer(pattern, html):
            image_url = unquote(match.group(1)).encode("utf-8").decode("unicode_escape")
            image_url = image_url.replace("&amp;", "&")
            if image_url.startswith("http") and image_url not in urls:
                urls.append(image_url)
            if len(urls) >= max_urls:
                return urls
    return urls


def find_duckduckgo_image_urls(query: str, max_urls: int) -> list[str]:
    urls: list[str] = []
    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=max_urls, safesearch="moderate", region="vn-vi"):
            image_url = result.get("image")
            if image_url and image_url.startswith("http") and image_url not in urls:
                urls.append(image_url)
    return urls


def find_image_urls(provider: str, query: str, max_urls: int) -> list[tuple[str, str]]:
    if provider == "duckduckgo":
        return [(url, "duckduckgo") for url in find_duckduckgo_image_urls(query, max_urls)]
    if provider == "bing":
        return [(url, "bing") for url in find_bing_image_urls(query, max_urls)]

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for provider_name, finder in [("duckduckgo", find_duckduckgo_image_urls), ("bing", find_bing_image_urls)]:
        try:
            for url in finder(query, max_urls):
                if url not in seen:
                    seen.add(url)
                    results.append((url, provider_name))
        except Exception as exc:
            print(f"  {provider_name} search failed: {exc}")
    return results


def safe_suffix(content_type: str, url: str) -> str:
    content_type = content_type.lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    suffix = Path(url.split("?")[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def safe_stem(text: str) -> str:
    chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "image"


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    candidate = folder / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def image_metrics(image: Image.Image, sha256: str) -> ImageMetrics:
    width, height = image.size
    aspect_ratio = max(width / max(height, 1), height / max(width, 1))
    return ImageMetrics(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        brightness=brightness_mean(image),
        blur_score=blur_score(image),
        phash=perceptual_hash(image),
        sha256=sha256,
    )


def decode_image(content: bytes, min_size: int) -> tuple[Image.Image | None, ImageMetrics | None, str]:
    sha256 = hashlib.sha256(content).hexdigest()
    try:
        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        metrics = image_metrics(image, sha256)
        if min(metrics.width, metrics.height) < min_size:
            return image, metrics, "too_small"
        return image, metrics, ""
    except Exception as exc:
        return None, None, f"invalid_image:{type(exc).__name__}"


def load_reference_images(roots: list[Path]) -> tuple[list[ReferenceImage], set[str]]:
    references: list[ReferenceImage] = []
    shas: set[str] = set()
    for root in roots:
        paths = list_images(root)
        if paths:
            print(f"Reference root: {relative_or_absolute(root)} ({len(paths)} images)")
        for path in paths:
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    references.append(ReferenceImage(perceptual_hash(image), path))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                shas.add(digest)
            except Exception:
                continue
    return references, shas


def nearest_duplicate(phash: str, references: list[ReferenceImage], threshold: int) -> tuple[ReferenceImage, int] | None:
    best: tuple[ReferenceImage, int] | None = None
    for reference in references:
        distance = hamming_distance_hex(phash, reference.phash)
        if distance <= threshold and (best is None or distance < best[1]):
            best = (reference, distance)
    return best


def save_jpeg_bytes(image: Image.Image, path: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@torch.no_grad()
def predict_image(model, class_names: list[str], transform, image: Image.Image, device: torch.device) -> Prediction:
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    values, indices = torch.topk(probs, k=min(2, len(class_names)))
    top1_idx = int(indices[0].item())
    top2_idx = int(indices[1].item()) if len(indices) > 1 else top1_idx
    return Prediction(
        top1_class=class_names[top1_idx],
        top1_confidence=float(values[0].item()),
        top2_class=class_names[top2_idx],
        top2_confidence=float(values[1].item()) if len(values) > 1 else 0.0,
    )


def decide_pool(
    prediction: Prediction,
    *,
    top1_threshold: float,
    top2_threshold: float,
    ambiguous_margin: float,
) -> tuple[str, str, str]:
    top1_is_canh = prediction.top1_class in CANH_CHUA_CLASSES
    top2_is_canh = prediction.top2_class in CANH_CHUA_CLASSES
    if not top1_is_canh and not top2_is_canh:
        return "model_rejected", "model_not_canh_chua", ""
    if top1_is_canh and top2_is_canh and prediction.margin < ambiguous_margin:
        return "accepted_unknown", "canh_classes_small_margin", "canh_chua_unknown"
    if top1_is_canh and prediction.top1_confidence >= top1_threshold:
        return "accepted", "model_top1_canh_chua", prediction.top1_class
    best_canh_confidence = max(
        prediction.top1_confidence if top1_is_canh else 0.0,
        prediction.top2_confidence if top2_is_canh else 0.0,
    )
    if best_canh_confidence >= top2_threshold:
        return "accepted_unknown", "possible_canh_chua_low_or_second_rank", "canh_chua_unknown"
    return "model_rejected", "canh_chua_confidence_below_threshold", ""


def append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def manifest_row(
    *,
    status: str,
    reason: str,
    suggested_class: str,
    target_pool: str,
    query: str,
    provider: str,
    source_url: str,
    raw_path: Path | None,
    output_path: Path | None,
    deleted_raw: bool,
    prediction: Prediction | None,
    duplicate_of: Path | None,
    duplicate_distance: str,
    metrics: ImageMetrics | None,
) -> dict[str, str]:
    prediction = prediction or Prediction()
    return {
        "status": status,
        "reason": reason,
        "suggested_class": suggested_class,
        "target_pool": target_pool,
        "query": query,
        "provider": provider,
        "source_url": source_url,
        "raw_path": relative_or_absolute(raw_path),
        "output_path": relative_or_absolute(output_path),
        "deleted_raw": str(deleted_raw),
        "top1_class": prediction.top1_class,
        "top1_confidence": f"{prediction.top1_confidence:.6f}" if prediction.top1_class else "",
        "top2_class": prediction.top2_class,
        "top2_confidence": f"{prediction.top2_confidence:.6f}" if prediction.top2_class else "",
        "margin": f"{prediction.margin:.6f}" if prediction.top1_class else "",
        "duplicate_of": relative_or_absolute(duplicate_of),
        "duplicate_distance": duplicate_distance,
        "sha256": metrics.sha256 if metrics else "",
        "phash": metrics.phash if metrics else "",
        "width": str(metrics.width) if metrics else "",
        "height": str(metrics.height) if metrics else "",
        "aspect_ratio": f"{metrics.aspect_ratio:.4f}" if metrics else "",
        "brightness": f"{metrics.brightness:.4f}" if metrics else "",
        "blur_score": f"{metrics.blur_score:.4f}" if metrics else "",
    }


def delete_if_batch_file(path: Path | None, batch_root: Path, apply: bool) -> bool:
    if path is None or not path.exists():
        return False
    try:
        path.resolve().relative_to(batch_root.resolve())
    except ValueError:
        raise RuntimeError(f"Refusing to delete outside batch root: {path}")
    if not apply:
        return False
    path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl canh chua images, model-filter them, dedupe, and stage survivors for review.")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "configs" / "canh_chua_extra_queries.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--provider", choices=["duckduckgo", "bing", "mixed"], default="mixed")
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--max-downloads-per-class", type=int, default=120)
    parser.add_argument("--min-size", type=int, default=180)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--strict-phrase", action="store_true")
    parser.add_argument("--min-blur-score", type=float, default=20.0)
    parser.add_argument("--min-brightness", type=float, default=20.0)
    parser.add_argument("--max-brightness", type=float, default=238.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--crawl-phash-threshold", type=int, default=6)
    parser.add_argument("--post-phash-threshold", type=int, default=8)
    parser.add_argument("--top1-threshold", type=float, default=0.50)
    parser.add_argument("--top2-threshold", type=float, default=0.28)
    parser.add_argument("--ambiguous-margin", type=float, default=0.18)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--apply", action="store_true", help="Actually download, delete rejected raw files, and stage accepted images.")
    parser.add_argument("--keep-rejected-raw", action="store_true", help="Keep rejected raw files instead of deleting them from the new batch.")
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = DATA_DIR / "inbox" / "raw_batches" / f"canh_chua_{timestamp}"
    raw_root = batch_root / "raw"
    review_roots = {
        "canh_chua_co_ca": REVIEW_INBOX_DIR / f"canh_chua_co_ca_scrape_{timestamp}",
        "canh_chua_khong_ca": REVIEW_INBOX_DIR / f"canh_chua_khong_ca_scrape_{timestamp}",
        "canh_chua_unknown": REVIEW_INBOX_DIR / f"canh_chua_unknown_scrape_{timestamp}",
    }
    manifest_path = REPORTS_DIR / f"canh_chua_crawl_{timestamp}_manifest.csv"
    summary_path = REPORTS_DIR / f"canh_chua_crawl_{timestamp}_summary.json"
    queries = read_queries(args.queries)

    print("Apply:", args.apply)
    print("Queries:", relative_or_absolute(args.queries), len(queries))
    for row in queries:
        print(f"  [{row.class_name}] {refined_query(row.query, args.strict_phrase)}")
    print("Batch root:", relative_or_absolute(batch_root))
    print("Review outputs:")
    for pool, folder in review_roots.items():
        print(f"  {pool}: {relative_or_absolute(folder)}")
    print("Manifest:", relative_or_absolute(manifest_path))
    print("Summary:", relative_or_absolute(summary_path))
    print("Model:", relative_or_absolute(args.model))
    print(
        "Thresholds:",
        {
            "crawl_phash": args.crawl_phash_threshold,
            "post_phash": args.post_phash_threshold,
            "top1": args.top1_threshold,
            "top2": args.top2_threshold,
            "ambiguous_margin": args.ambiguous_margin,
        },
    )

    if not args.apply:
        print("Dry-run only. Re-run with --apply to crawl and stage images.")
        return

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    reference_roots = [REVIEWED_DIR, CLASSIFICATION_DIR, REVIEW_INBOX_DIR]
    references, reference_shas = load_reference_images(reference_roots)
    print(f"Reference phashes: {len(references)}")

    device = resolve_device()
    model, class_names, model_image_size, checkpoint = load_checkpoint(args.model, device)
    transform = eval_transforms(model_image_size)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint.get('metadata', {})}")

    raw_root.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    downloaded_by_class: Counter[str] = Counter()
    seen_urls: set[str] = set()
    batch_references: list[ReferenceImage] = []
    batch_shas: set[str] = set()

    for query_row in queries:
        if downloaded_by_class[query_row.class_name] >= args.max_downloads_per_class:
            continue
        search_query = refined_query(query_row.query, args.strict_phrase)
        print(f"Searching [{query_row.class_name}] {search_query}")
        try:
            urls = find_image_urls(args.provider, search_query, args.per_query)
        except Exception as exc:
            print(f"  search failed: {exc}")
            urls = []
        counts["urls_found"] += len(urls)
        print(f"  found {len(urls)} urls")

        for idx, (url, provider_used) in enumerate(urls):
            if downloaded_by_class[query_row.class_name] >= args.max_downloads_per_class:
                break
            if url in seen_urls:
                counts["duplicate_url"] += 1
                continue
            seen_urls.add(url)

            raw_path: Path | None = None
            output_path: Path | None = None
            deleted_raw = False
            prediction: Prediction | None = None
            duplicate_path: Path | None = None
            duplicate_distance = ""
            try:
                response = requests.get(url, headers=HEADERS, timeout=25)
                response.raise_for_status()
                image, metrics, decode_reason = decode_image(response.content, args.min_size)
                if image is None or metrics is None:
                    counts[decode_reason or "invalid_image"] += 1
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason=decode_reason or "invalid_image",
                            suggested_class=query_row.class_name,
                            target_pool="",
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=None,
                            output_path=None,
                            deleted_raw=False,
                            prediction=None,
                            duplicate_of=None,
                            duplicate_distance="",
                            metrics=None,
                        )
                    )
                    continue

                if metrics.sha256 in reference_shas or metrics.sha256 in batch_shas:
                    counts["duplicate_sha"] += 1
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason="duplicate_sha",
                            suggested_class=query_row.class_name,
                            target_pool="",
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=None,
                            output_path=None,
                            deleted_raw=False,
                            prediction=None,
                            duplicate_of=None,
                            duplicate_distance="0",
                            metrics=metrics,
                        )
                    )
                    continue

                duplicate = nearest_duplicate(metrics.phash, references + batch_references, args.crawl_phash_threshold)
                if duplicate is not None:
                    duplicate_reference, distance = duplicate
                    counts["duplicate_phash_crawl"] += 1
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason="duplicate_phash_crawl",
                            suggested_class=query_row.class_name,
                            target_pool="",
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=None,
                            output_path=None,
                            deleted_raw=False,
                            prediction=None,
                            duplicate_of=duplicate_reference.path,
                            duplicate_distance=str(distance),
                            metrics=metrics,
                        )
                    )
                    continue

                suffix = safe_suffix(response.headers.get("content-type", ""), url)
                raw_dir = raw_root / query_row.class_name
                raw_path = unique_path(raw_dir, f"{query_row.class_name}_{idx:03d}_{metrics.sha256[:12]}", suffix)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(response.content)
                downloaded_by_class[query_row.class_name] += 1
                counts["downloaded"] += 1

                reasons = quality_reasons(
                    metrics,
                    min_size=args.min_size,
                    max_aspect_ratio=args.max_aspect_ratio,
                    min_blur_score=args.min_blur_score,
                    min_brightness=args.min_brightness,
                    max_brightness=args.max_brightness,
                )
                if reasons:
                    counts["quality_rejected"] += 1
                    deleted_raw = delete_if_batch_file(raw_path, batch_root, apply=not args.keep_rejected_raw)
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason=";".join(reasons),
                            suggested_class=query_row.class_name,
                            target_pool="",
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=raw_path,
                            output_path=None,
                            deleted_raw=deleted_raw,
                            prediction=None,
                            duplicate_of=None,
                            duplicate_distance="",
                            metrics=metrics,
                        )
                    )
                    continue

                prediction = predict_image(model, class_names, transform, image, device)
                status, reason, target_pool = decide_pool(
                    prediction,
                    top1_threshold=args.top1_threshold,
                    top2_threshold=args.top2_threshold,
                    ambiguous_margin=args.ambiguous_margin,
                )
                if status == "model_rejected":
                    counts["model_rejected"] += 1
                    deleted_raw = delete_if_batch_file(raw_path, batch_root, apply=not args.keep_rejected_raw)
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason=reason,
                            suggested_class=query_row.class_name,
                            target_pool="",
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=raw_path,
                            output_path=None,
                            deleted_raw=deleted_raw,
                            prediction=prediction,
                            duplicate_of=None,
                            duplicate_distance="",
                            metrics=metrics,
                        )
                    )
                    continue

                normalized = normalize_image(image, image_size=args.image_size, mode=args.mode)
                normalized_phash = perceptual_hash(normalized)
                duplicate = nearest_duplicate(normalized_phash, references + batch_references, args.post_phash_threshold)
                if duplicate is not None:
                    duplicate_reference, distance = duplicate
                    counts["duplicate_phash_post"] += 1
                    deleted_raw = delete_if_batch_file(raw_path, batch_root, apply=not args.keep_rejected_raw)
                    rows.append(
                        manifest_row(
                            status="rejected",
                            reason="duplicate_phash_post",
                            suggested_class=query_row.class_name,
                            target_pool=target_pool,
                            query=query_row.query,
                            provider=provider_used,
                            source_url=url,
                            raw_path=raw_path,
                            output_path=None,
                            deleted_raw=deleted_raw,
                            prediction=prediction,
                            duplicate_of=duplicate_reference.path,
                            duplicate_distance=str(distance),
                            metrics=metrics,
                        )
                    )
                    continue

                review_dir = review_roots[target_pool]
                output_path = unique_path(review_dir, f"canh_chua_{safe_stem(target_pool)}_{metrics.sha256[:12]}", ".jpg")
                output_sha = save_jpeg_bytes(normalized, output_path)
                batch_shas.add(metrics.sha256)
                batch_shas.add(output_sha)
                batch_references.append(ReferenceImage(metrics.phash, raw_path))
                batch_references.append(ReferenceImage(normalized_phash, output_path))
                counts["accepted"] += 1
                counts[f"accepted_{target_pool}"] += 1
                rows.append(
                    manifest_row(
                        status="accepted",
                        reason=reason,
                        suggested_class=query_row.class_name,
                        target_pool=target_pool,
                        query=query_row.query,
                        provider=provider_used,
                        source_url=url,
                        raw_path=raw_path,
                        output_path=output_path,
                        deleted_raw=False,
                        prediction=prediction,
                        duplicate_of=None,
                        duplicate_distance="",
                        metrics=metrics,
                    )
                )
                print(f"  accepted -> {target_pool}: {output_path.name} ({prediction.top1_class} {prediction.top1_confidence:.2f})")
            except Exception as exc:
                counts["download_or_process_error"] += 1
                if raw_path is not None:
                    deleted_raw = delete_if_batch_file(raw_path, batch_root, apply=not args.keep_rejected_raw)
                rows.append(
                    manifest_row(
                        status="rejected",
                        reason=f"error:{type(exc).__name__}",
                        suggested_class=query_row.class_name,
                        target_pool="",
                        query=query_row.query,
                        provider=provider_used,
                        source_url=url,
                        raw_path=raw_path,
                        output_path=output_path,
                        deleted_raw=deleted_raw,
                        prediction=prediction,
                        duplicate_of=duplicate_path,
                        duplicate_distance=duplicate_distance,
                        metrics=None,
                    )
                )
        time.sleep(args.sleep)

    append_manifest(manifest_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_root": relative_or_absolute(batch_root),
        "raw_root": relative_or_absolute(raw_root),
        "review_outputs": {pool: relative_or_absolute(path) for pool, path in review_roots.items()},
        "manifest": relative_or_absolute(manifest_path),
        "model": relative_or_absolute(args.model),
        "provider": args.provider,
        "queries": [query_row.__dict__ for query_row in queries],
        "counts": dict(counts),
        "downloaded_by_class": dict(downloaded_by_class),
        "thresholds": {
            "crawl_phash_threshold": args.crawl_phash_threshold,
            "post_phash_threshold": args.post_phash_threshold,
            "top1_threshold": args.top1_threshold,
            "top2_threshold": args.top2_threshold,
            "ambiguous_margin": args.ambiguous_margin,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
