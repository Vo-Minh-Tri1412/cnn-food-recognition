from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from ddgs import DDGS
from PIL import Image

from canteen_checkout.config import DISH_CLASSES, SCRAPED_CANDIDATES_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}


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


def download_url(url: str, out_dir: Path, prefix: str, min_size: int) -> Path | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()[:16]
        suffix = safe_suffix(response.headers.get("content-type", ""), url)
        out_path = out_dir / f"{prefix}_{digest}{suffix}"
        if out_path.exists():
            return None
        out_path.write_bytes(response.content)
        if not validate_image(out_path, min_size):
            out_path.unlink(missing_ok=True)
            return None
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
    args = parser.parse_args()

    queries = read_queries(args.queries)
    if args.class_name:
        queries = [(class_name, query) for class_name, query in queries if class_name == args.class_name]

    counts: dict[str, int] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for class_name, query in queries:
        class_dir = args.out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        counts.setdefault(class_name, len(list(class_dir.glob("*.*"))))
        if counts[class_name] >= args.max_downloads_per_class:
            continue
        print(f"Searching [{class_name}] {query}")
        try:
            if args.provider == "duckduckgo":
                urls = find_duckduckgo_image_urls(query, args.per_query)
                if not urls:
                    print("  duckduckgo returned no urls; trying bing fallback")
                    urls = find_bing_image_urls(query, args.per_query)
            else:
                urls = find_bing_image_urls(query, args.per_query)
        except Exception as exc:
            print(f"  search failed: {exc}")
            if args.provider == "duckduckgo":
                print("  trying bing fallback")
                try:
                    urls = find_bing_image_urls(query, args.per_query)
                except Exception as fallback_exc:
                    print(f"  fallback failed: {fallback_exc}")
                    urls = []
            else:
                urls = []
        print(f"  found {len(urls)} urls")
        for idx, url in enumerate(urls):
            if counts[class_name] >= args.max_downloads_per_class:
                break
            out_path = download_url(url, class_dir, f"{class_name}_{idx:03d}", args.min_size)
            if out_path:
                counts[class_name] += 1
                print(f"  saved {out_path.name}")
        time.sleep(args.sleep)

    print("Candidate counts:")
    for class_name in DISH_CLASSES:
        count = len(list((args.out / class_name).glob("*.*"))) if (args.out / class_name).exists() else 0
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
