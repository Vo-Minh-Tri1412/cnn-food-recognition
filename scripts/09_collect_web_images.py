from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from ddgs import DDGS
from PIL import Image, ImageOps

from canteen_checkout.config import (
    DISH_CLASSES,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SCRAPED_CANDIDATES_DIR,
    SCRAPED_MANIFEST_CSV,
)
from canteen_checkout.data_quality import hamming_distance_hex, perceptual_hash


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}
NEGATIVE_TERMS = ["-logo", "-icon", "-emoji", "-clipart", "-vector", "-cartoon"]


def read_queries(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            class_name = row["class_name"].strip()
            query = row["query"].strip()
            if class_name not in DISH_CLASSES:
                raise ValueError(f"Unknown class in query CSV: {class_name}")
            if query:
                rows.append((class_name, query))
    return rows


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def append_manifest_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "class_name",
        "query",
        "source_url",
        "image_url",
        "file_path",
        "provider",
        "download_time",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def refine_query(query: str, *, strict_phrase: bool = False) -> str:
    query = query.strip()
    if strict_phrase and not (query.startswith('"') and query.endswith('"')):
        query = f'"{query}"'
    tokens = set(query.split())
    missing_terms = [term for term in NEGATIVE_TERMS if term not in tokens]
    return " ".join([query, *missing_terms]).strip()


def existing_short_digests(out_dir: Path) -> set[str]:
    digests: set[str] = set()
    if not out_dir.exists():
        return digests
    for path in out_dir.glob("*.*"):
        match = re.search(r"_([0-9a-f]{16})$", path.stem)
        if match:
            digests.add(match.group(1))
    return digests


def list_class_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    search_root = class_dir if class_dir.exists() else root
    if not search_root.exists():
        return []
    return sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def existing_phashes(*, out_dir: Path, against_roots: list[Path], class_name: str) -> list[str]:
    phashes: list[str] = []
    paths = sorted(p for p in out_dir.glob("*.*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for root in against_roots:
        paths.extend(list_class_images(root, class_name))
    for path in paths:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                phashes.append(perceptual_hash(image))
        except Exception:
            continue
    return phashes


def is_near_phash(phash: str, seen_phashes: list[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, seen) <= threshold for seen in seen_phashes)


def find_bing_image_urls(query: str, max_urls: int) -> list[str]:
    url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1"
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    html = response.text
    urls = []
    patterns = [
        r'"murl":"(.*?)"',
        r"mediaurl=([^&\"']+)",
    ]
    for pattern in patterns:
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


def safe_suffix(content_type: str, url: str) -> str:
    content_type = content_type.lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".jpg"


def validate_image(path: Path, min_size: int) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            w, h = image.size
        return w >= min_size and h >= min_size
    except Exception:
        return False


def decode_response_image(content: bytes, min_size: int) -> tuple[Image.Image, str] | None:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            w, h = image.size
            if w < min_size or h < min_size:
                return None
            return image, perceptual_hash(image)
    except Exception:
        return None


def download_url(
    url: str,
    out_dir: Path,
    prefix: str,
    min_size: int,
    seen_digests: set[str],
    seen_phashes: list[str],
    phash_threshold: int,
) -> Path | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()[:16]
        if digest in seen_digests:
            return None
        decoded = decode_response_image(response.content, min_size)
        if decoded is None:
            return None
        _, phash = decoded
        if is_near_phash(phash, seen_phashes, phash_threshold):
            return None
        suffix = safe_suffix(response.headers.get("content-type", ""), url)
        out_path = out_dir / f"{prefix}_{digest}{suffix}"
        if out_path.exists():
            seen_digests.add(digest)
            return None
        out_path.write_bytes(response.content)
        seen_digests.add(digest)
        seen_phashes.append(phash)
        return out_path
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect image candidates from web search for manual review.")
    parser.add_argument("--queries", type=Path, default=Path("configs/search_queries.csv"))
    parser.add_argument("--out", type=Path, default=SCRAPED_CANDIDATES_DIR)
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--max-downloads-per-class", type=int, default=80)
    parser.add_argument("--min-size", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--class-name", choices=DISH_CLASSES, default=None)
    parser.add_argument("--provider", choices=["duckduckgo", "bing"], default="duckduckgo")
    parser.add_argument("--manifest", type=Path, default=SCRAPED_MANIFEST_CSV)
    parser.add_argument("--strict-phrase", action="store_true", help="Quote each query phrase before adding negative terms.")
    parser.add_argument("--raw-query", action="store_true", help="Do not append negative terms to search queries.")
    parser.add_argument(
        "--dedupe-against",
        type=Path,
        action="append",
        default=[],
        help="Seed perceptual duplicate checks from one or more existing dataset roots.",
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    queries = read_queries(args.queries)
    if args.class_name:
        queries = [(class_name, query) for class_name, query in queries if class_name == args.class_name]

    counts: dict[str, int] = {}
    digest_cache: dict[str, set[str]] = {}
    phash_cache: dict[str, list[str]] = {}
    seen_urls: dict[str, set[str]] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for class_name, query in queries:
        search_query = query if args.raw_query else refine_query(query, strict_phrase=args.strict_phrase)
        class_dir = args.out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        counts.setdefault(class_name, len(list(class_dir.glob("*.*"))))
        digest_cache.setdefault(class_name, existing_short_digests(class_dir))
        phash_cache.setdefault(
            class_name,
            existing_phashes(out_dir=class_dir, against_roots=args.dedupe_against, class_name=class_name),
        )
        seen_urls.setdefault(class_name, set())
        if counts[class_name] >= args.max_downloads_per_class:
            continue
        print(f"Searching [{class_name}] {search_query}")
        provider_used = args.provider
        try:
            if args.provider == "duckduckgo":
                urls = find_duckduckgo_image_urls(search_query, args.per_query)
                if not urls:
                    print("  duckduckgo returned no urls; trying bing fallback")
                    urls = find_bing_image_urls(search_query, args.per_query)
                    provider_used = "bing_fallback"
            else:
                urls = find_bing_image_urls(search_query, args.per_query)
        except Exception as exc:
            print(f"  search failed: {exc}")
            if args.provider == "duckduckgo":
                print("  trying bing fallback")
                try:
                    urls = find_bing_image_urls(search_query, args.per_query)
                    provider_used = "bing_fallback"
                except Exception as fallback_exc:
                    print(f"  fallback failed: {fallback_exc}")
                    urls = []
            else:
                urls = []
        print(f"  found {len(urls)} urls")
        for idx, url in enumerate(urls):
            if counts[class_name] >= args.max_downloads_per_class:
                break
            if url in seen_urls[class_name]:
                continue
            seen_urls[class_name].add(url)
            out_path = download_url(
                url,
                class_dir,
                f"{class_name}_{idx:03d}",
                args.min_size,
                digest_cache[class_name],
                phash_cache[class_name],
                args.phash_threshold,
            )
            if out_path:
                counts[class_name] += 1
                append_manifest_row(
                    args.manifest,
                    {
                        "class_name": class_name,
                        "query": query,
                        "source_url": url,
                        "image_url": url,
                        "file_path": relative_or_absolute(out_path),
                        "provider": provider_used,
                        "download_time": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                print(f"  saved {out_path.name}")
        time.sleep(args.sleep)

    print("Candidate counts:")
    for class_name in DISH_CLASSES:
        count = len(list((args.out / class_name).glob("*.*"))) if (args.out / class_name).exists() else 0
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
