from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from ddgs import DDGS

from canteen_checkout.config import PROJECT_ROOT, REPORTS_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}

DEFAULT_QUERIES = [
    "Vietnamese food dataset Kaggle",
    "Vietnamese food classification dataset GitHub",
    "canh chua dataset Vietnamese food",
    "thit kho dataset Vietnamese food",
    "Vietnamese canteen tray food dataset",
    "Roboflow canteen food tray rice egg vegetable dataset",
    "Vietnamese rice tray food detection dataset",
    "VietFood dataset Vietnamese cuisine",
    "30VNFoods dataset GitHub",
    "Vietnamese vegetable soup food dataset",
    "Vietnamese stir fried vegetables dataset",
]

FIELDS = [
    "search_time",
    "query",
    "rank",
    "title",
    "url",
    "snippet",
    "source_hint",
    "related_classes",
    "status",
    "note",
]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def source_hint(url: str) -> str:
    lowered = url.lower()
    if "kaggle.com" in lowered:
        return "kaggle"
    if "github.com" in lowered:
        return "github"
    if "roboflow.com" in lowered:
        return "roboflow"
    if "huggingface.co" in lowered:
        return "huggingface"
    if "zenodo.org" in lowered:
        return "zenodo"
    return "web"


def related_classes(text: str) -> str:
    text = text.lower()
    hits = []
    rules = {
        "canh_rau": ["vegetable soup", "canh rau", "canh cải", "rau muống"],
        "rau_xao": ["stir fried", "rau xào", "lagim", "đậu que", "củ sắn"],
        "canh_chua_co_ca": ["canh chua", "sour fish soup"],
        "thit_kho": ["thit kho", "thịt kho", "caramelized pork"],
        "thit_kho_trung": ["egg", "trứng", "thịt kho trứng"],
        "com_trang": ["rice", "cơm"],
    }
    for class_name, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            hits.append(class_name)
    return ";".join(hits)


def read_queries(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUERIES
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            query = (row.get("query") or "").strip()
            if query:
                rows.append(query)
    return rows


def ddgs_search(query: str, max_results: int) -> list[dict[str, str]]:
    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results, region="vn-vi", safesearch="moderate"):
            url = item.get("href") or item.get("url") or ""
            if not url:
                continue
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": url,
                    "snippet": item.get("body") or "",
                }
            )
    return results


def bing_search(query: str, max_results: int) -> list[dict[str, str]]:
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    # Keep the fallback intentionally conservative. DDGS is the primary provider;
    # this fallback records the result page when result parsing is unavailable.
    return [{"title": f"Bing results for {query}", "url": url, "snippet": response.text[:300]}][:max_results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search public dataset links before image crawling.")
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "public_dataset_search_manifest.csv")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--provider", choices=["ddgs", "bing", "mixed"], default="mixed")
    args = parser.parse_args()

    seen_urls: set[str] = set()
    rows: list[dict[str, str]] = []
    for query in read_queries(args.queries):
        providers = ["ddgs", "bing"] if args.provider == "mixed" else [args.provider]
        rank = 0
        for provider in providers:
            try:
                results = ddgs_search(query, args.max_results) if provider == "ddgs" else bing_search(query, args.max_results)
            except Exception as exc:
                print(f"{provider} failed for {query}: {exc}")
                continue
            for result in results:
                if result["url"] in seen_urls:
                    continue
                seen_urls.add(result["url"])
                rank += 1
                text = f"{result['title']} {result['snippet']} {result['url']}"
                rows.append(
                    {
                        "search_time": datetime.now().isoformat(timespec="seconds"),
                        "query": query,
                        "rank": str(rank),
                        "title": result["title"],
                        "url": result["url"],
                        "snippet": result["snippet"],
                        "source_hint": source_hint(result["url"]),
                        "related_classes": related_classes(text),
                        "status": "candidate",
                        "note": "",
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {relative_or_absolute(args.out)}")


if __name__ == "__main__":
    main()
