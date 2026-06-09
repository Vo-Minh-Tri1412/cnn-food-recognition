from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import DISH_CLASSES, DISPLAY_NAMES, DOWNLOADS_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT


ACTION_FIELDS = [
    "timestamp",
    "item_id",
    "pool",
    "source_path",
    "action",
    "class_name",
    "output_path",
]

MODEL_DECISIONS = ["auto_accepted", "needs_review", "ambiguous_review", "model_rejected"]
DEFAULT_EXTRA_LABELS = ["mon_khac", "future_use", "khay_background"]


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    pool: str
    path: Path
    rel_path: str
    filename: str
    suggested_class: str
    needs_review: bool
    method: str
    source_dataset: str
    label_name: str
    model_class: str
    model_confidence: str
    model_decision: str
    model_reason: str
    duplicate_of: str
    duplicate_distance: str


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def latest_external_staging() -> Path:
    root = DOWNLOADS_DIR / "external_staging"
    candidates = sorted((p for p in root.glob("external_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No external staging folders found in {root}")
    return candidates[0]


def stable_item_id(staging_root: Path, path: Path) -> str:
    rel = path.resolve().relative_to(staging_root.resolve()).as_posix()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def load_manifest(staging_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = staging_root / "reports" / "external_import_manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    if not manifest_path.exists():
        return rows
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            output_path = row.get("output_path") or ""
            if not output_path:
                continue
            path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path).resolve()
            rows[str(path)] = row
    return rows


def load_model_suggestions(staging_root: Path) -> dict[str, dict[str, str]]:
    suggestions_path = staging_root / "reports" / "model_suggestions.csv"
    rows: dict[str, dict[str, str]] = {}
    if not suggestions_path.exists():
        return rows
    with suggestions_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_path = row.get("source_path") or ""
            if not source_path:
                continue
            path = Path(source_path)
            resolved = path if path.is_absolute() else PROJECT_ROOT / path
            rows[str(resolved.resolve())] = row
    return rows


def list_review_items(staging_root: Path) -> list[ReviewItem]:
    manifest = load_manifest(staging_root)
    model_suggestions = load_model_suggestions(staging_root)
    review_root = staging_root / "review"
    items: list[ReviewItem] = []
    if not review_root.exists():
        return items
    for pool_dir in sorted(p for p in review_root.iterdir() if p.is_dir()):
        for path in sorted(pool_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            meta = manifest.get(str(path.resolve()), {})
            model_meta = model_suggestions.get(str(path.resolve()), {})
            item_id = stable_item_id(staging_root, path)
            model_class = model_meta.get("top1_class", "")
            suggested_class = meta.get("suggested_class", "")
            if model_class and not suggested_class and model_meta.get("decision") != "model_rejected":
                suggested_class = model_class
            items.append(
                ReviewItem(
                    item_id=item_id,
                    pool=pool_dir.name,
                    path=path,
                    rel_path=relative_or_absolute(path),
                    filename=path.name,
                    suggested_class=suggested_class,
                    needs_review=str(meta.get("needs_review", "true")).lower() == "true",
                    method=meta.get("method", ""),
                    source_dataset=meta.get("source_dataset", ""),
                    label_name=meta.get("label_name", ""),
                    model_class=model_class,
                    model_confidence=model_meta.get("top1_confidence", ""),
                    model_decision=model_meta.get("decision", ""),
                    model_reason=model_meta.get("reason", ""),
                    duplicate_of=model_meta.get("duplicate_of", ""),
                    duplicate_distance=model_meta.get("duplicate_distance", ""),
                )
            )
    return items


def ensure_review_dirs(staging_root: Path) -> None:
    for class_name in DISH_CLASSES:
        (staging_root / "reviewed" / class_name).mkdir(parents=True, exist_ok=True)
    (staging_root / "reviewed_extra").mkdir(parents=True, exist_ok=True)
    (staging_root / "manual_rejected").mkdir(parents=True, exist_ok=True)
    (staging_root / "reports").mkdir(parents=True, exist_ok=True)


def extra_labels_path(staging_root: Path) -> Path:
    return staging_root / "reports" / "extra_labels.json"


def slugify_label(value: str) -> str:
    value = value.strip().lower().replace("đ", "d")
    value = re.sub(r"[^a-z0-9_ -]+", "", value)
    value = re.sub(r"[\s-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Extra folder name cannot be empty")
    return value[:80]


def load_extra_labels(staging_root: Path) -> list[str]:
    path = extra_labels_path(staging_root)
    labels: list[str] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                labels = [slugify_label(str(item)) for item in payload if str(item).strip()]
        except Exception:
            labels = []
    if not labels:
        labels = DEFAULT_EXTRA_LABELS[:]
    return sorted(dict.fromkeys(labels))


def save_extra_labels(staging_root: Path, labels: list[str]) -> None:
    path = extra_labels_path(staging_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(dict.fromkeys(labels)), indent=2, ensure_ascii=False), encoding="utf-8")


def read_action_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_action(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in ACTION_FIELDS})


def action_state(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    state: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row.get("item_id", "")
        if not item_id:
            continue
        if row.get("action") == "undo":
            state.pop(item_id, None)
        else:
            state[item_id] = row
    return state


def unique_destination(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def count_images_by_folder(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    counts: dict[str, int] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[folder.name] = sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return counts


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class ReviewStore:
    def __init__(self, staging_root: Path):
        self.staging_root = staging_root.resolve()
        ensure_review_dirs(self.staging_root)
        save_extra_labels(self.staging_root, load_extra_labels(self.staging_root))
        self.action_log = self.staging_root / "reports" / "review_actions.csv"
        self.reload()

    def reload(self) -> dict[str, object]:
        self.items = list_review_items(self.staging_root)
        self.items_by_id = {item.item_id: item for item in self.items}
        return {"ok": True, "items": len(self.items), "pools": self.pools()}

    def rows(self) -> list[dict[str, str]]:
        return read_action_rows(self.action_log)

    def state(self) -> dict[str, dict[str, str]]:
        return action_state(self.rows())

    def pools(self) -> list[dict[str, object]]:
        state = self.state()
        result = []
        for pool in sorted({item.pool for item in self.items}):
            total = sum(1 for item in self.items if item.pool == pool)
            done = sum(1 for item in self.items if item.pool == pool and item.item_id in state)
            result.append({"name": pool, "total": total, "done": done, "remaining": total - done})
        return result

    def filtered_items(self, pool: str, include_done: bool, model_decision: str = "") -> list[ReviewItem]:
        state = self.state()
        return [
            item
            for item in self.items
            if (not pool or item.pool == pool)
            and (not model_decision or item.model_decision == model_decision)
            and (include_done or item.item_id not in state)
        ]

    def stats(self) -> dict[str, object]:
        state = self.state()
        by_action = {}
        by_class = {}
        by_extra = {}
        for row in state.values():
            action = row.get("action", "")
            by_action[action] = by_action.get(action, 0) + 1
            if action == "label":
                class_name = row.get("class_name", "")
                by_class[class_name] = by_class.get(class_name, 0) + 1
            if action == "label_extra":
                class_name = row.get("class_name", "")
                by_extra[class_name] = by_extra.get(class_name, 0) + 1
        reviewed_counts = count_images_by_folder(self.staging_root / "reviewed")
        extra_counts = count_images_by_folder(self.staging_root / "reviewed_extra")
        rejected_counts = count_images_by_folder(self.staging_root / "manual_rejected")
        return {
            "total": len(self.items),
            "done": len(state),
            "remaining": len(self.items) - len(state),
            "by_action": by_action,
            "by_class": by_class,
            "by_extra": by_extra,
            "library": {
                "reviewed_total": sum(reviewed_counts.values()),
                "extra_total": sum(extra_counts.values()),
                "rejected_total": sum(rejected_counts.values()),
                "reviewed_by_class": reviewed_counts,
                "extra_by_label": extra_counts,
                "rejected_by_pool": rejected_counts,
            },
        }

    def model_decisions(self) -> list[dict[str, object]]:
        state = self.state()
        result = []
        known_decisions = sorted({item.model_decision for item in self.items if item.model_decision})
        for decision in [name for name in MODEL_DECISIONS if name in known_decisions]:
            total = sum(1 for item in self.items if item.model_decision == decision)
            done = sum(1 for item in self.items if item.model_decision == decision and item.item_id in state)
            result.append({"name": decision, "total": total, "done": done, "remaining": total - done})
        return result

    def serialize_item(self, item: ReviewItem) -> dict[str, object]:
        return {
            "id": item.item_id,
            "pool": item.pool,
            "filename": item.filename,
            "path": item.rel_path,
            "suggested_class": item.suggested_class,
            "needs_review": item.needs_review,
            "method": item.method,
            "source_dataset": item.source_dataset,
            "label_name": item.label_name,
            "model_class": item.model_class,
            "model_confidence": item.model_confidence,
            "model_decision": item.model_decision,
            "model_reason": item.model_reason,
            "duplicate_of": item.duplicate_of,
            "duplicate_distance": item.duplicate_distance,
            "image_url": f"/media/{item.item_id}",
        }

    def api_state(self, pool: str, index: int, include_done: bool, model_decision: str = "") -> dict[str, object]:
        items = self.filtered_items(pool, include_done, model_decision)
        if items:
            index = max(0, min(index, len(items) - 1))
            current = self.serialize_item(items[index])
        else:
            index = 0
            current = None
        start = max(0, index - 6)
        end = min(len(items), index + 18)
        return {
            "staging_root": relative_or_absolute(self.staging_root),
            "pools": self.pools(),
            "classes": [{"name": name, "display_name": DISPLAY_NAMES.get(name, name)} for name in DISH_CLASSES],
            "extra_labels": self.extra_labels(),
            "model_decisions": self.model_decisions(),
            "selected_pool": pool,
            "selected_model_decision": model_decision,
            "index": index,
            "count": len(items),
            "current": current,
            "nearby": [self.serialize_item(item) for item in items[start:end]],
            "stats": self.stats(),
            "include_done": include_done,
        }

    def extra_labels(self) -> list[str]:
        return load_extra_labels(self.staging_root)

    def add_extra_label(self, label: str) -> dict[str, object]:
        slug = slugify_label(label)
        labels = self.extra_labels()
        if slug not in labels:
            labels.append(slug)
            save_extra_labels(self.staging_root, labels)
        (self.staging_root / "reviewed_extra" / slug).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "label": slug, "labels": self.extra_labels()}

    def action(self, item_id: str, action: str, class_name: str = "") -> dict[str, object]:
        if item_id not in self.items_by_id:
            raise ValueError("Unknown item id")
        if action not in {"label", "label_extra", "reject", "skip"}:
            raise ValueError("Unsupported action")
        if action == "label" and class_name not in DISH_CLASSES:
            raise ValueError("Invalid class name")
        if action == "label_extra":
            class_name = self.add_extra_label(class_name)["label"]

        item = self.items_by_id[item_id]
        output = ""
        if action == "label":
            target_dir = self.staging_root / "reviewed" / class_name
            target = unique_destination(target_dir, f"{item.pool}_{item.filename}")
            shutil.copy2(item.path, target)
            output = relative_or_absolute(target)
        elif action == "label_extra":
            target_dir = self.staging_root / "reviewed_extra" / class_name
            target = unique_destination(target_dir, f"{item.pool}_{item.filename}")
            shutil.copy2(item.path, target)
            output = relative_or_absolute(target)
        elif action == "reject":
            target_dir = self.staging_root / "manual_rejected" / item.pool
            target = unique_destination(target_dir, item.filename)
            shutil.copy2(item.path, target)
            output = relative_or_absolute(target)

        append_action(
            self.action_log,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "pool": item.pool,
                "source_path": item.rel_path,
                "action": action,
                "class_name": class_name,
                "output_path": output,
            },
        )
        return {"ok": True, "output_path": output}

    def bulk_action(self, item_ids: list[str], action: str, class_name: str = "") -> dict[str, object]:
        state = self.state()
        results = []
        for item_id in item_ids:
            if item_id in state:
                results.append({"id": item_id, "ok": True, "skipped": True, "reason": "already_done"})
                continue
            try:
                result = self.action(item_id, action, class_name)
                results.append({"id": item_id, **result})
            except Exception as exc:
                results.append({"id": item_id, "ok": False, "error": str(exc)})
        return {"ok": True, "count": len(results), "results": results}

    def undo(self, item_id: str) -> dict[str, object]:
        state = self.state()
        if not item_id:
            active_ids = set(state)
            for row in reversed(self.rows()):
                candidate_id = row.get("item_id", "")
                if candidate_id in active_ids and row.get("action") != "undo":
                    item_id = candidate_id
                    break
        current = state.get(item_id)
        if not current:
            return {"ok": True, "undone": False}
        output_path = current.get("output_path") or ""
        if output_path:
            path = PROJECT_ROOT / output_path if not Path(output_path).is_absolute() else Path(output_path)
            if is_inside(path, self.staging_root) and path.exists():
                path.unlink()
        append_action(
            self.action_log,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item_id,
                "pool": current.get("pool", ""),
                "source_path": current.get("source_path", ""),
                "action": "undo",
                "class_name": current.get("class_name", ""),
                "output_path": output_path,
            },
        )
        return {"ok": True, "undone": True}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Canteen Dataset Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #151922;
      --muted: #657083;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --danger: #b42318;
      --warn: #9a5b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    .app {
      display: grid;
      grid-template-columns: 260px minmax(420px, 1fr) 320px;
      height: 100vh;
      min-height: 640px;
    }
    aside, main {
      min-width: 0;
      min-height: 0;
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 14px;
      overflow: auto;
    }
    .right {
      border-right: 0;
      border-left: 1px solid var(--line);
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    main {
      display: grid;
      grid-template-rows: 58px 1fr 128px;
      overflow: hidden;
    }
    h1, h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
    }
    .meta, .small {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .pool {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 9px 10px;
      margin: 8px 0;
      border-radius: 6px;
      cursor: pointer;
      text-align: left;
    }
    .pool.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .badge {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .viewer {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
      overflow: hidden;
    }
    .image-wrap {
      width: 100%;
      height: 100%;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      background: #eef1f5;
      border-radius: 8px;
      overflow: hidden;
    }
    .image-wrap img {
      display: block;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .thumbs {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      overflow-x: auto;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    .thumb {
      width: 92px;
      height: 92px;
      flex: 0 0 auto;
      border: 2px solid transparent;
      border-radius: 6px;
      object-fit: cover;
      background: #e5e9f0;
      cursor: pointer;
    }
    .thumb.active { border-color: var(--accent); }
    .thumb-item {
      position: relative;
      flex: 0 0 auto;
    }
    .thumb-item.selected .thumb {
      border-color: var(--ok);
      box-shadow: 0 0 0 2px rgba(46, 160, 67, 0.18);
    }
    .thumb-select {
      position: absolute;
      top: 6px;
      left: 6px;
      width: 18px;
      height: 18px;
    }
    .class-btn, .cmd {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      min-height: 36px;
      padding: 8px 10px;
      margin: 7px 0;
      text-align: left;
      cursor: pointer;
      overflow-wrap: anywhere;
    }
    .class-btn.suggested {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .cmd-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }
    .extra-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      margin: 8px 0 12px;
    }
    input {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      min-width: 0;
      font: inherit;
    }
    .cmd {
      text-align: center;
      margin: 0;
    }
    .reject { color: var(--danger); }
    .skip { color: var(--warn); }
    .empty {
      color: var(--muted);
      padding: 24px;
      text-align: center;
    }
    .field {
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
    }
    @media (max-width: 1020px) {
      .app { grid-template-columns: 1fr; height: auto; min-height: 100vh; }
      aside, .right { border: 0; border-bottom: 1px solid var(--line); }
      main { height: 76vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Dataset Review</h1>
      <div id="root" class="meta"></div>
      <button id="reloadFiles" class="cmd">Reload Files</button>
      <div class="stats">
        <div class="stat"><div class="small">Done</div><strong id="done">0</strong></div>
        <div class="stat"><div class="small">Left</div><strong id="left">0</strong></div>
        <div class="stat"><div class="small">Reviewed</div><strong id="reviewedCount">0</strong></div>
        <div class="stat"><div class="small">Extra</div><strong id="extraCount">0</strong></div>
        <div class="stat"><div class="small">Rejected</div><strong id="rejectedCount">0</strong></div>
      </div>
      <div id="pools"></div>
      <h1>Model Filter</h1>
      <div id="decisionFilters"></div>
    </aside>
    <main>
      <header>
        <div>
          <h2 id="title">Loading</h2>
          <div id="subtitle" class="meta"></div>
        </div>
        <div id="counter" class="badge"></div>
      </header>
      <section class="viewer">
        <div id="imageWrap" class="image-wrap"><div class="empty">No image</div></div>
      </section>
      <section id="thumbs" class="thumbs"></section>
    </main>
    <aside class="right">
      <h2>Classes</h2>
      <div id="classes"></div>
      <h2>Bulk</h2>
      <div class="cmd-row">
        <button id="selectVisible" class="cmd">Select Visible</button>
        <button id="clearSelection" class="cmd">Clear</button>
      </div>
      <div class="cmd-row">
        <button id="bulkReject" class="cmd reject">Reject Selected</button>
        <button id="bulkSkip" class="cmd skip">Skip Selected</button>
      </div>
      <button id="skipVisible" class="cmd skip">Skip Visible</button>
      <div class="meta" id="selectedCount">Selected: 0</div>
      <h2>Extra Folders</h2>
      <div id="extraLabels"></div>
      <div class="extra-row">
        <input id="extraInput" type="text" placeholder="mon_ngoai_de" />
        <button id="addExtra" class="cmd skip">Add</button>
      </div>
      <div class="cmd-row">
        <button id="reject" class="cmd reject">Reject</button>
        <button id="skip" class="cmd skip">Skip</button>
      </div>
      <button id="undo" class="cmd">Undo</button>
      <div class="field">
        <div class="small">Pool</div>
        <strong id="poolName"></strong>
      </div>
      <div class="field">
        <div class="small">Suggested</div>
        <strong id="suggested"></strong>
      </div>
      <div class="field">
        <div class="small">Model</div>
        <strong id="model"></strong>
        <div id="modelDecision" class="meta"></div>
      </div>
      <div class="field">
        <div class="small">Duplicate</div>
        <div id="duplicate" class="meta"></div>
      </div>
      <div class="field">
        <div class="small">Source</div>
        <div id="source" class="meta"></div>
      </div>
      <div class="field">
        <div class="small">File</div>
        <div id="file" class="meta"></div>
      </div>
    </aside>
  </div>
  <script>
    let pool = "";
    let modelDecision = "";
    let index = 0;
    let state = null;
    let currentId = null;
    let lastActionId = null;
    let selectedIds = new Set();

    async function request(path, options) {
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    async function load() {
      const params = new URLSearchParams({ pool, decision: modelDecision, index: String(index) });
      state = await request("/api/state?" + params.toString());
      render();
    }

    async function reloadFiles() {
      await request("/api/reload", { method: "POST" });
      selectedIds.clear();
      index = 0;
      await load();
    }

    function render() {
      document.getElementById("root").textContent = state.staging_root;
      document.getElementById("done").textContent = state.stats.done;
      document.getElementById("left").textContent = state.stats.remaining;
      document.getElementById("reviewedCount").textContent = state.stats.library.reviewed_total;
      document.getElementById("extraCount").textContent = state.stats.library.extra_total;
      document.getElementById("rejectedCount").textContent = state.stats.library.rejected_total;
      renderPools();
      renderDecisionFilters();
      renderCurrent();
      renderClasses();
      renderExtraLabels();
      renderThumbs();
      updateSelectedCount();
    }

    function renderPools() {
      const box = document.getElementById("pools");
      box.innerHTML = "";
      const all = document.createElement("button");
      all.className = "pool" + (pool === "" ? " active" : "");
      all.innerHTML = `<span>All pools</span><span class="badge">${state.stats.remaining}/${state.stats.total}</span>`;
      all.onclick = () => { pool = ""; index = 0; load(); };
      box.appendChild(all);
      state.pools.forEach(p => {
        const btn = document.createElement("button");
        btn.className = "pool" + (pool === p.name ? " active" : "");
        btn.innerHTML = `<span>${p.name}</span><span class="badge">${p.remaining}/${p.total}</span>`;
        btn.onclick = () => { pool = p.name; index = 0; load(); };
        box.appendChild(btn);
      });
    }

    function renderDecisionFilters() {
      const box = document.getElementById("decisionFilters");
      box.innerHTML = "";
      const all = document.createElement("button");
      all.className = "pool" + (modelDecision === "" ? " active" : "");
      all.innerHTML = `<span>All decisions</span><span class="badge">${state.stats.remaining}/${state.stats.total}</span>`;
      all.onclick = () => { modelDecision = ""; index = 0; load(); };
      box.appendChild(all);
      state.model_decisions.forEach(d => {
        const btn = document.createElement("button");
        btn.className = "pool" + (modelDecision === d.name ? " active" : "");
        btn.innerHTML = `<span>${d.name}</span><span class="badge">${d.remaining}/${d.total}</span>`;
        btn.onclick = () => { modelDecision = d.name; index = 0; load(); };
        box.appendChild(btn);
      });
    }

    function renderCurrent() {
      const wrap = document.getElementById("imageWrap");
      const item = state.current;
      currentId = item ? item.id : null;
      if (!item) {
        wrap.innerHTML = `<div class="empty">No image</div>`;
        document.getElementById("title").textContent = "Complete";
        document.getElementById("subtitle").textContent = "";
        document.getElementById("counter").textContent = "0 / 0";
        document.getElementById("poolName").textContent = "";
        document.getElementById("suggested").textContent = "";
        document.getElementById("model").textContent = "";
        document.getElementById("modelDecision").textContent = "";
        document.getElementById("duplicate").textContent = "";
        document.getElementById("source").textContent = "";
        document.getElementById("file").textContent = "";
        return;
      }
      wrap.innerHTML = `<img src="${item.image_url}" alt="">`;
      document.getElementById("title").textContent = item.pool;
      document.getElementById("subtitle").textContent = item.method || item.label_name || item.source_dataset || "";
      document.getElementById("counter").textContent = `${state.index + 1} / ${state.count}`;
      document.getElementById("poolName").textContent = item.pool;
      document.getElementById("suggested").textContent = item.suggested_class || "-";
      const confidence = item.model_confidence ? Number(item.model_confidence) : null;
      document.getElementById("model").textContent = item.model_class
        ? `${item.model_class} (${confidence === null ? "?" : (confidence * 100).toFixed(1) + "%"})`
        : "-";
      document.getElementById("modelDecision").textContent = [item.model_decision, item.model_reason].filter(Boolean).join(" | ") || "-";
      document.getElementById("duplicate").textContent = item.duplicate_of
        ? `${item.duplicate_of} (${item.duplicate_distance})`
        : "-";
      document.getElementById("source").textContent = item.source_dataset || "-";
      document.getElementById("file").textContent = item.filename;
    }

    function renderClasses() {
      const box = document.getElementById("classes");
      box.innerHTML = "";
      const suggested = state.current ? state.current.suggested_class : "";
      state.classes.forEach(cls => {
        const btn = document.createElement("button");
        btn.className = "class-btn" + (cls.name === suggested ? " suggested" : "");
        btn.textContent = cls.name;
        btn.onclick = () => act("label", cls.name);
        box.appendChild(btn);
      });
    }

    function renderExtraLabels() {
      const box = document.getElementById("extraLabels");
      box.innerHTML = "";
      state.extra_labels.forEach(label => {
        const btn = document.createElement("button");
        btn.className = "class-btn";
        btn.textContent = label;
        btn.onclick = () => act("label_extra", label);
        box.appendChild(btn);
      });
    }

    function renderThumbs() {
      const box = document.getElementById("thumbs");
      box.innerHTML = "";
      state.nearby.forEach((item, offset) => {
        const wrap = document.createElement("div");
        wrap.className = "thumb-item" + (selectedIds.has(item.id) ? " selected" : "");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "thumb-select";
        checkbox.checked = selectedIds.has(item.id);
        checkbox.onclick = event => {
          event.stopPropagation();
          toggleSelected(item.id);
        };
        const img = document.createElement("img");
        img.className = "thumb" + (item.id === currentId ? " active" : "");
        img.src = item.image_url;
        img.onclick = () => {
          const currentNearbyIndex = state.nearby.findIndex(x => x.id === currentId);
          index += offset - currentNearbyIndex;
          if (index < 0) index = 0;
          load();
        };
        wrap.appendChild(img);
        wrap.appendChild(checkbox);
        box.appendChild(wrap);
      });
    }

    function toggleSelected(id) {
      if (selectedIds.has(id)) selectedIds.delete(id);
      else selectedIds.add(id);
      updateSelectedCount();
      renderThumbs();
    }

    function updateSelectedCount() {
      document.getElementById("selectedCount").textContent = `Selected: ${selectedIds.size}`;
    }

    async function bulkActForIds(ids, action, className = "") {
      const uniqueIds = Array.from(new Set(ids)).filter(Boolean);
      if (!uniqueIds.length) return;
      lastActionId = uniqueIds[uniqueIds.length - 1];
      await request("/api/bulk-action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: uniqueIds, action, class_name: className })
      });
      uniqueIds.forEach(id => selectedIds.delete(id));
      await load();
    }

    async function bulkAct(action, className = "") {
      await bulkActForIds(Array.from(selectedIds), action, className);
    }

    async function act(action, className = "") {
      if (selectedIds.size > 0) {
        await bulkAct(action, className);
        return;
      }
      if (!currentId) return;
      lastActionId = currentId;
      await request("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: currentId, action, class_name: className })
      });
      await load();
    }

    async function undo() {
      const targetId = lastActionId || "";
      await request("/api/undo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: targetId })
      });
      lastActionId = null;
      await load();
    }

    async function addExtraLabel() {
      const input = document.getElementById("extraInput");
      const label = input.value.trim();
      if (!label) return;
      await request("/api/extra-label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label })
      });
      input.value = "";
      await load();
    }

    function move(delta) {
      index += delta;
      if (index < 0) index = 0;
      if (state && index >= state.count) index = state.count - 1;
      load();
    }

    document.getElementById("reject").onclick = () => act("reject");
    document.getElementById("skip").onclick = () => act("skip");
    document.getElementById("undo").onclick = undo;
    document.getElementById("reloadFiles").onclick = reloadFiles;
    document.getElementById("selectVisible").onclick = () => {
      state.nearby.forEach(item => selectedIds.add(item.id));
      renderThumbs();
      updateSelectedCount();
    };
    document.getElementById("clearSelection").onclick = () => {
      selectedIds.clear();
      renderThumbs();
      updateSelectedCount();
    };
    document.getElementById("bulkReject").onclick = () => bulkAct("reject");
    document.getElementById("bulkSkip").onclick = () => bulkAct("skip");
    document.getElementById("skipVisible").onclick = () => bulkActForIds(state.nearby.map(item => item.id), "skip");
    document.getElementById("addExtra").onclick = addExtraLabel;
    document.getElementById("extraInput").addEventListener("keydown", event => {
      if (event.key === "Enter") {
        event.stopPropagation();
        addExtraLabel();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.target && event.target.tagName === "INPUT") return;
      if (!state || !state.current) return;
      const n = Number(event.key);
      if (n >= 1 && n <= 9) {
        const cls = state.classes[n - 1];
        if (cls) act("label", cls.name);
      } else if (event.key === "0") {
        const cls = state.classes[9];
        if (cls) act("label", cls.name);
      } else if (event.key === "-") {
        const cls = state.classes[10];
        if (cls) act("label", cls.name);
      } else if (event.key === "Enter" && state.current.suggested_class) {
        act("label", state.current.suggested_class);
      } else if (event.key === "Backspace") {
        undo();
      } else if (event.key.toLowerCase() === "x") {
        act("reject");
      } else if (event.key.toLowerCase() === "s") {
        act("skip");
      } else if (event.key.toLowerCase() === "r") {
        reloadFiles();
      } else if (event.key === "ArrowRight") {
        move(1);
      } else if (event.key === "ArrowLeft") {
        move(-1);
      }
    });

    load().catch(err => {
      document.body.innerHTML = `<pre>${String(err)}</pre>`;
    });
  </script>
</body>
</html>
"""


def json_response(handler: BaseHTTPRequestHandler, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain; charset=utf-8") -> None:
    payload = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                text_response(self, INDEX_HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                pool = query.get("pool", [""])[0]
                model_decision = query.get("decision", [""])[0]
                index = int(query.get("index", ["0"])[0])
                include_done = query.get("include_done", ["false"])[0].lower() == "true"
                json_response(
                    self,
                    self.store.api_state(
                        pool=pool,
                        index=index,
                        include_done=include_done,
                        model_decision=model_decision,
                    ),
                )
                return
            if parsed.path.startswith("/media/"):
                item_id = parsed.path.rsplit("/", 1)[-1]
                item = self.store.items_by_id.get(item_id)
                if item is None or not item.path.exists():
                    text_response(self, "Not found", HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
                data = item.path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            text_response(self, "Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body)
            if parsed.path == "/api/action":
                result = self.store.action(
                    item_id=data.get("id", ""),
                    action=data.get("action", ""),
                    class_name=data.get("class_name", ""),
                )
                json_response(self, result)
                return
            if parsed.path == "/api/bulk-action":
                result = self.store.bulk_action(
                    item_ids=data.get("ids", []),
                    action=data.get("action", ""),
                    class_name=data.get("class_name", ""),
                )
                json_response(self, result)
                return
            if parsed.path == "/api/undo":
                json_response(self, self.store.undo(data.get("id", "")))
                return
            if parsed.path == "/api/extra-label":
                json_response(self, self.store.add_extra_label(data.get("label", "")))
                return
            if parsed.path == "/api/reload":
                json_response(self, self.store.reload())
                return
            text_response(self, "Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local web app for reviewing external dataset pools.")
    parser.add_argument("--staging", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    staging = args.staging or latest_external_staging()
    store = ReviewStore(staging)
    ReviewHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Review root: {store.staging_root}")
    print(f"Items: {len(store.items)}")
    print(f"Open: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping review server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
