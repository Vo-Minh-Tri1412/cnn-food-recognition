from __future__ import annotations

import argparse
import csv
import json
import mimetypes
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

import torch
from PIL import Image

from canteen_checkout.config import CLASSIFICATION_DIR, DEFAULT_MODEL_PATH, DISH_CLASSES, DOWNLOADS_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT
from canteen_checkout.data_quality import assess_image, hamming_distance_hex
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device


ACTION_FIELDS = [
    "timestamp",
    "item_id",
    "source_path",
    "action",
    "from_class",
    "to_class",
    "output_path",
    "note",
]

MODEL_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class DataItem:
    item_id: str
    path: Path
    rel_path: str
    split: str
    class_name: str
    source: str
    filename: str
    sha256: str
    phash: str
    blur_score: float
    brightness: float


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def source_from_name(path: Path) -> str:
    if path.name.startswith("old_"):
        return "old"
    if path.name.startswith("reviewed_"):
        return "reviewed"
    return "unmanaged"


def stable_item_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def safe_label(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in value.strip().lower())
    value = "_".join(value.split())
    return value or "future_use"


def unique_destination(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if not target.exists():
        return target
    idx = 1
    while True:
        candidate = folder / f"{target.stem}_{idx:03d}{target.suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def roots() -> list[dict[str, str]]:
    external_roots = sorted((DOWNLOADS_DIR / "external_staging").glob("external_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_external = external_roots[0] if external_roots else DOWNLOADS_DIR / "external_staging"
    candidates = [
        ("classification", CLASSIFICATION_DIR),
        ("external_review", latest_external / "review"),
        ("external_reviewed", latest_external / "reviewed"),
        ("quarantine", PROJECT_ROOT / "data" / "quarantine"),
    ]
    return [{"name": name, "path": relative_or_absolute(path)} for name, path in candidates if path.exists()]


def infer_split_class(root: Path, path: Path) -> tuple[str, str] | None:
    rel = path.resolve().relative_to(root.resolve())
    parts = rel.parts
    if len(parts) >= 3 and parts[0] in {"train", "val", "test"} and parts[1] in DISH_CLASSES:
        return parts[0], parts[1]
    if len(parts) >= 2:
        return "", parts[0]
    return None


def list_labeled_paths(root: Path, split: str = "", class_name: str = "") -> list[tuple[Path, str, str]]:
    paths: list[tuple[Path, str, str]] = []
    if not root.exists():
        return paths
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        inferred = infer_split_class(root, path)
        if inferred is None:
            continue
        item_split, item_class = inferred
        if split and item_split != split:
            continue
        if class_name and item_class != class_name:
            continue
        paths.append((path, item_split, item_class))
    return paths


def make_item(path: Path, item_split: str, item_class: str) -> DataItem:
    _, metrics, _ = assess_image(path)
    if metrics is None:
        sha256 = ""
        phash = ""
        blur_score = 0.0
        brightness = 0.0
    else:
        sha256 = metrics.sha256
        phash = metrics.phash
        blur_score = metrics.blur_score
        brightness = metrics.brightness
    return DataItem(
        item_id=stable_item_id(path),
        path=path,
        rel_path=relative_or_absolute(path),
        split=item_split,
        class_name=item_class,
        source=source_from_name(path),
        filename=path.name,
        sha256=sha256,
        phash=phash,
        blur_score=blur_score,
        brightness=brightness,
    )


def list_items(root: Path, split: str = "", class_name: str = "") -> list[DataItem]:
    items: list[DataItem] = []
    for path, item_split, item_class in list_labeled_paths(root, split, class_name):
        items.append(make_item(path, item_split, item_class))
    return items


def count_tree(root: Path) -> dict[str, object]:
    result: dict[str, object] = {"total": 0, "splits": {}, "classes": {}, "sources": {}}
    for path, item_split, item_class in list_labeled_paths(root):
        result["total"] = int(result["total"]) + 1
        split_key = item_split or "root"
        result["splits"].setdefault(split_key, 0)
        result["splits"][split_key] += 1
        result["classes"].setdefault(item_class, 0)
        result["classes"][item_class] += 1
        source = source_from_name(path)
        result["sources"].setdefault(source, 0)
        result["sources"][source] += 1
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
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


def action_log() -> Path:
    return PROJECT_ROOT / "outputs" / "reports" / "data_ide_actions.csv"


def load_model_once(model_path: Path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    device = resolve_device()
    model, class_names, image_size, checkpoint = load_checkpoint(model_path, device)
    cached = (model, class_names, image_size, checkpoint, device)
    MODEL_CACHE[key] = cached
    return cached


@torch.no_grad()
def predict_item(item: DataItem, threshold: float) -> dict[str, object]:
    model, class_names, image_size, checkpoint, device = load_model_once(DEFAULT_MODEL_PATH)
    transform = eval_transforms(image_size)
    image = Image.open(item.path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    values, indices = probs.topk(k=2)
    top1 = class_names[int(indices[0])]
    top2 = class_names[int(indices[1])]
    top1_conf = float(values[0])
    top2_conf = float(values[1])
    margin = top1_conf - top2_conf
    if item.class_name not in DISH_CLASSES:
        decision = "outside_target_classes"
    elif top1 != item.class_name and top1_conf >= threshold:
        decision = "high_confidence_disagreement"
    elif top1_conf < threshold:
        decision = "low_confidence"
    elif margin < 0.15:
        decision = "small_margin"
    else:
        decision = "ok"
    return {
        "top1": top1,
        "top1_confidence": round(top1_conf, 4),
        "top2": top2,
        "top2_confidence": round(top2_conf, 4),
        "margin": round(margin, 4),
        "decision": decision,
        "model_arch": checkpoint.get("arch", ""),
    }


class DataIDE:
    def __init__(self):
        self.last_items: dict[str, DataItem] = {}

    def state(self) -> dict[str, object]:
        root_list = roots()
        return {
            "roots": root_list,
            "classes": DISH_CLASSES,
            "log": relative_or_absolute(action_log()),
        }

    def browse(self, root_value: str, split: str = "", class_name: str = "", page: int = 0, page_size: int = 80) -> dict[str, object]:
        root = resolve_project_path(root_value)
        paths = list_labeled_paths(root, split, class_name)
        start = max(0, page * page_size)
        end = min(len(paths), start + page_size)
        items = [make_item(path, item_split, item_class) for path, item_split, item_class in paths[start:end]]
        self.last_items.update({item.item_id: item for item in items})
        return {
            "root": relative_or_absolute(root),
            "counts": count_tree(root),
            "page": page,
            "page_size": page_size,
            "total": len(paths),
            "items": [self.serialize_item(item) for item in items],
        }

    def serialize_item(self, item: DataItem) -> dict[str, object]:
        return {
            "id": item.item_id,
            "path": item.rel_path,
            "image_url": f"/file?path={item.rel_path}",
            "split": item.split,
            "class_name": item.class_name,
            "source": item.source,
            "filename": item.filename,
            "sha256": item.sha256,
            "phash": item.phash,
            "blur_score": round(item.blur_score, 2),
            "brightness": round(item.brightness, 2),
        }

    def item_from_id(self, item_id: str) -> DataItem:
        item = self.last_items.get(item_id)
        if item and item.path.exists():
            return item
        path = resolve_project_path(item_id)
        for root_info in roots():
            root = resolve_project_path(root_info["path"])
            if is_inside(path, root):
                inferred = infer_split_class(root, path)
                if inferred is None:
                    break
                split, class_name = inferred
                _, metrics, _ = assess_image(path)
                return DataItem(
                    item_id=stable_item_id(path),
                    path=path,
                    rel_path=relative_or_absolute(path),
                    split=split,
                    class_name=class_name,
                    source=source_from_name(path),
                    filename=path.name,
                    sha256=metrics.sha256 if metrics else "",
                    phash=metrics.phash if metrics else "",
                    blur_score=metrics.blur_score if metrics else 0.0,
                    brightness=metrics.brightness if metrics else 0.0,
                )
        raise ValueError("Unknown item")

    def move_to_class(self, item_id: str, class_name: str) -> dict[str, object]:
        if class_name not in DISH_CLASSES:
            raise ValueError("Invalid target class")
        item = self.item_from_id(item_id)
        if item.class_name == class_name:
            return {"ok": True, "output_path": item.rel_path, "noop": True}
        if item.split:
            target_dir = item.path.parents[1] / class_name
        else:
            target_dir = item.path.parent.parent / class_name
        target = unique_destination(target_dir, item.filename)
        shutil.move(str(item.path), str(target))
        append_action(
            action_log(),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "source_path": item.rel_path,
                "action": "move_class",
                "from_class": item.class_name,
                "to_class": class_name,
                "output_path": relative_or_absolute(target),
                "note": "",
            },
        )
        return {"ok": True, "output_path": relative_or_absolute(target)}

    def quarantine(self, item_id: str, label: str = "manual_rejected") -> dict[str, object]:
        item = self.item_from_id(item_id)
        label = safe_label(label)
        target_dir = PROJECT_ROOT / "data" / "quarantine" / label / item.class_name
        target = unique_destination(target_dir, item.filename)
        shutil.move(str(item.path), str(target))
        append_action(
            action_log(),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "source_path": item.rel_path,
                "action": "quarantine",
                "from_class": item.class_name,
                "to_class": label,
                "output_path": relative_or_absolute(target),
                "note": label,
            },
        )
        return {"ok": True, "output_path": relative_or_absolute(target)}

    def undo(self) -> dict[str, object]:
        rows = read_rows(action_log())
        for row in reversed(rows):
            if row.get("action") == "undo":
                continue
            source = resolve_project_path(row.get("source_path", ""))
            output = resolve_project_path(row.get("output_path", ""))
            if output.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(source))
                append_action(
                    action_log(),
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "item_id": row.get("item_id", ""),
                        "source_path": row.get("output_path", ""),
                        "action": "undo",
                        "from_class": row.get("to_class", ""),
                        "to_class": row.get("from_class", ""),
                        "output_path": row.get("source_path", ""),
                        "note": "undo_last_move",
                    },
                )
                return {"ok": True, "undone": True, "restored": relative_or_absolute(source)}
        return {"ok": True, "undone": False}

    def model_predict(self, item_ids: list[str], threshold: float) -> dict[str, object]:
        predictions = []
        for item_id in item_ids:
            item = self.item_from_id(item_id)
            predictions.append({"id": item_id, **predict_item(item, threshold)})
        feedback_path = PROJECT_ROOT / "outputs" / "reports" / "model_review_feedback.csv"
        return {"ok": True, "predictions": predictions, "feedback_path": relative_or_absolute(feedback_path)}

    def save_feedback(self, payload: dict[str, object]) -> dict[str, object]:
        path = PROJECT_ROOT / "outputs" / "reports" / "model_review_feedback.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = ["timestamp", "item_id", "current_class", "model_top1", "is_correct", "correct_class", "note"]
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: str(payload.get(field, "")) for field in fields})
        return {"ok": True, "feedback_path": relative_or_absolute(path)}


STORE = DataIDE()


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canteen Data IDE</title>
  <style>
    :root{--bg:#f5f6f8;--panel:#fff;--line:#d9dee5;--ink:#1b222a;--muted:#687382;--accent:#126a5a;--danger:#b42318;--warn:#9a5b00}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;background:#fff;border-bottom:1px solid var(--line)}
    h1{font-size:18px;margin:0}.app{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 56px)}
    aside{background:#fff;border-right:1px solid var(--line);padding:14px;overflow:auto}.main{padding:14px;overflow:auto}
    .group{border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-bottom:12px}
    .group h2{margin:0 0 10px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
    label{display:block;color:var(--muted);font-size:12px;margin:8px 0 4px} select,input,button{width:100%;height:34px;border:1px solid var(--line);border-radius:6px;background:#fff;padding:0 8px;font:inherit}
    button{cursor:pointer;font-weight:650}.primary{background:var(--accent);border-color:var(--accent);color:#fff}.danger{color:var(--danger)}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat{background:#f3f5f7;border-radius:6px;padding:8px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}.card{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}
    .card.selected{outline:3px solid var(--accent)}.card img{width:100%;aspect-ratio:1/1;object-fit:cover;background:#eef1f4}.card .body{padding:8px}.small{font-size:12px;color:var(--muted);word-break:break-word}
    .pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eef1f4;font-size:12px;margin:2px}.bad{background:#fee4e2;color:#912018}.warn{background:#fff2cc;color:#7a4a00}.ok{background:#dcfae6;color:#05603a}
    table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left}
  </style>
</head>
<body>
<header><h1>Canteen Data IDE</h1><div id="status" class="small">Ready</div></header>
<div class="app">
  <aside>
    <div class="group"><h2>Nguồn dữ liệu</h2>
      <label>Root</label><select id="rootSelect"></select>
      <label>Split</label><select id="splitSelect"><option value="">all/root</option><option>train</option><option>val</option><option>test</option></select>
      <label>Class</label><select id="classFilter"></select>
      <div class="row"><button id="loadBtn" class="primary">Load</button><button id="undoBtn">Undo</button></div>
    </div>
    <div class="group"><h2>Action</h2>
      <label>Move to class</label><select id="targetClass"></select>
      <div class="row"><button id="moveBtn">Move selected</button><button id="bulkMoveBtn">Move all selected</button></div>
      <label>Quarantine label</label><input id="quarantineLabel" value="manual_rejected">
      <div class="row"><button id="quarantineBtn" class="danger">Quarantine</button><button id="futureUseBtn">Future use</button></div>
    </div>
    <div class="group"><h2>Model assistant</h2>
      <label>Threshold</label><input id="threshold" type="number" step="0.01" min="0" max="1" value="0.70">
      <button id="predictBtn" class="primary">Predict visible</button>
      <div id="modelStats" class="small"></div>
    </div>
    <div class="group"><h2>Counts</h2><div id="counts" class="small"></div></div>
  </aside>
  <main class="main">
    <div class="group"><h2>Ảnh</h2><div class="row"><button id="selectAllBtn">Select visible</button><button id="clearBtn">Clear</button></div></div>
    <div id="grid" class="grid"></div>
  </main>
</div>
<script>
const $=id=>document.getElementById(id); const state={items:[], selected:new Set(), preds:{}};
function status(t){$('status').textContent=t}
async function api(url,body=null){const opt=body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{};const r=await fetch(url,opt);const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||r.statusText);return d}
function clsOptions(){return [''].concat(state.classes||[]).map(c=>`<option value="${c}">${c||'all'}</option>`).join('')}
async function init(){const s=await api('/api/state');state.classes=s.classes;$('rootSelect').innerHTML=s.roots.map(r=>`<option value="${r.path}">${r.name}</option>`).join('');$('classFilter').innerHTML=clsOptions();$('targetClass').innerHTML=(state.classes||[]).map(c=>`<option>${c}</option>`).join('');await load()}
async function load(){state.selected.clear();const q=new URLSearchParams({root:$('rootSelect').value,split:$('splitSelect').value,class_name:$('classFilter').value,page_size:'120'});const d=await api('/api/browse?'+q);state.items=d.items;renderCounts(d.counts);renderGrid();status(`${d.total} files`)}
function renderCounts(c){$('counts').innerHTML=`<div class=stats><div class=stat>Total<br><b>${c.total}</b></div><div class=stat>Classes<br><b>${Object.keys(c.classes).length}</b></div><div class=stat>Sources<br><b>${Object.keys(c.sources).join(', ')}</b></div></div><pre>${JSON.stringify(c.classes,null,2)}</pre>`}
function renderGrid(){$('grid').innerHTML=state.items.map(it=>{const p=state.preds[it.id];const dec=p?`<span class="pill ${p.decision==='ok'?'ok':p.decision.includes('disagreement')?'bad':'warn'}">${p.decision}</span>`:'';return `<div class="card ${state.selected.has(it.id)?'selected':''}" data-id="${it.id}"><img src="${it.image_url}"><div class=body><b>${it.class_name}</b> <span class=pill>${it.split||'root'}</span> ${dec}<div class=small>${it.filename}</div><div class=small>${it.source} · blur ${it.blur_score}</div>${p?`<div class=small>top1 ${p.top1} ${p.top1_confidence}<br>top2 ${p.top2} ${p.top2_confidence}</div><div class=row><button data-fb="yes">Model đúng</button><button data-fb="no">Model sai</button></div>`:''}</div></div>`}).join('');document.querySelectorAll('.card').forEach(card=>{card.onclick=e=>{if(e.target.dataset.fb){feedback(card.dataset.id,e.target.dataset.fb);return}state.selected.has(card.dataset.id)?state.selected.delete(card.dataset.id):state.selected.add(card.dataset.id);renderGrid()}})}
async function act(action, extra={}){const ids=[...state.selected];if(!ids.length){alert('Chưa chọn ảnh');return}for(const id of ids){await api('/api/action',{action,item_id:id,...extra})}await load()}
async function predict(){const ids=state.items.map(x=>x.id);const d=await api('/api/predict',{item_ids:ids,threshold:Number($('threshold').value)});state.preds={};for(const p of d.predictions)state.preds[p.id]=p;const counts={};for(const p of d.predictions)counts[p.decision]=(counts[p.decision]||0)+1;$('modelStats').textContent=JSON.stringify(counts);renderGrid()}
async function feedback(id,val){const it=state.items.find(x=>x.id===id),p=state.preds[id]||{};await api('/api/feedback',{timestamp:new Date().toISOString(),item_id:id,current_class:it.class_name,model_top1:p.top1||'',is_correct:val,correct_class:val==='yes'?p.top1:'',note:''});status('Saved feedback')}
$('loadBtn').onclick=load;$('undoBtn').onclick=async()=>{await api('/api/undo',{});await load()};$('moveBtn').onclick=()=>act('move_class',{class_name:$('targetClass').value});$('bulkMoveBtn').onclick=$('moveBtn').onclick;$('quarantineBtn').onclick=()=>act('quarantine',{label:$('quarantineLabel').value});$('futureUseBtn').onclick=()=>act('quarantine',{label:'future_use_'+$('quarantineLabel').value});$('predictBtn').onclick=predict;$('selectAllBtn').onclick=()=>{state.items.forEach(x=>state.selected.add(x.id));renderGrid()};$('clearBtn').onclick=()=>{state.selected.clear();renderGrid()};init().catch(e=>{status(e.message);alert(e.message)});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                self.send_json(STORE.state())
                return
            if parsed.path == "/api/browse":
                q = parse_qs(parsed.query)
                self.send_json(
                    STORE.browse(
                        q.get("root", [""])[0],
                        q.get("split", [""])[0],
                        q.get("class_name", [""])[0],
                        int(q.get("page", ["0"])[0]),
                        int(q.get("page_size", ["80"])[0]),
                    )
                )
                return
            if parsed.path == "/file":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                if not path.exists() or not is_inside(path, PROJECT_ROOT):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/action":
                action = payload.get("action")
                if action == "move_class":
                    self.send_json(STORE.move_to_class(str(payload["item_id"]), str(payload["class_name"])))
                    return
                if action == "quarantine":
                    self.send_json(STORE.quarantine(str(payload["item_id"]), str(payload.get("label") or "manual_rejected")))
                    return
                raise ValueError("Unsupported action")
            if parsed.path == "/api/undo":
                self.send_json(STORE.undo())
                return
            if parsed.path == "/api/predict":
                self.send_json(STORE.model_predict([str(x) for x in payload["item_ids"]], float(payload.get("threshold", 0.7))))
                return
            if parsed.path == "/api/feedback":
                self.send_json(STORE.save_feedback(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Data IDE for classification/review/quarantine folders.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Data IDE: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
