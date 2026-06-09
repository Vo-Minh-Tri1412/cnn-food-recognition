from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shutil
import sys
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch
from PIL import Image

from canteen_checkout.config import (
    BILLS_DIR,
    CROPPED_DISHES_DIR,
    DEFAULT_MODEL_PATH,
    DEMO_TRAYS_DIR,
    DISH_CLASSES,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
)
from canteen_checkout.cropping import CropRegion, crop_regions, five_compartment_template, load_regions
from canteen_checkout.io_utils import load_prices
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device
from canteen_checkout.pricing import THIT_KHO_TRUNG_CLASS, dish_price


UPLOAD_DIR = DEMO_TRAYS_DIR / "uploads"
IGNORE_LABELS = {"ignore", "ignored", "unknown", "other", "extra"}

MODEL_CACHE: dict[str, object] = {}


@torch.no_grad()
def predict_crop(model, class_names: list[str], image_size: int, crop_path: Path, device: torch.device) -> tuple[str, float]:
    transform = eval_transforms(image_size)
    image = Image.open(crop_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, idx = torch.max(probs, dim=0)
    return class_names[int(idx)], float(confidence.cpu().item())


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_safe_project_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def list_demo_images() -> list[dict[str, str]]:
    roots = [DEMO_TRAYS_DIR, PROJECT_ROOT / "Khay_com", PROJECT_ROOT / "data" / "raw_teacher_trays"]
    seen: set[Path] = set()
    images: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append({"path": relative_or_absolute(path), "name": path.name})
    return images


def list_region_templates() -> list[dict[str, str]]:
    templates = [{"path": "", "name": "Auto 5 ô"}]
    config_dir = PROJECT_ROOT / "configs"
    if config_dir.exists():
        for path in sorted(config_dir.glob("*regions*.json")):
            templates.append({"path": relative_or_absolute(path), "name": path.name})
    return templates


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def regions_from_payload(payload: dict, image_path: Path) -> list[CropRegion]:
    raw_regions = payload.get("regions")
    if raw_regions:
        return [
            CropRegion(
                name=str(item.get("name") or f"crop_{idx:02d}"),
                x=int(item["x"]),
                y=int(item["y"]),
                w=int(item["w"]),
                h=int(item["h"]),
                label=str(item.get("label") or "").strip() or None,
            )
            for idx, item in enumerate(raw_regions)
        ]

    template = str(payload.get("template") or "")
    if template:
        return load_regions(resolve_project_path(template))

    width, height = image_size(image_path)
    return five_compartment_template(width, height)


def load_model_once(model_path: Path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    device = resolve_device()
    model, class_names, model_image_size, checkpoint = load_checkpoint(model_path, device)
    cached = (model, class_names, model_image_size, checkpoint, device)
    MODEL_CACHE[key] = cached
    return cached


def run_checkout(payload: dict) -> dict:
    image_path = resolve_project_path(str(payload["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_safe_project_path(image_path):
        raise ValueError("Only project files can be used in the demo app")

    model_path = resolve_project_path(str(payload.get("model_path") or DEFAULT_MODEL_PATH))
    threshold = float(payload.get("threshold", 0.55))
    egg_count = int(payload.get("egg_count", 1))
    regions = regions_from_payload(payload, image_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = CROPPED_DISHES_DIR / f"{image_path.stem}_{run_id}"
    crop_paths = crop_regions(image_path, regions, out_dir)
    prices = load_prices()

    model = None
    class_names: list[str] = []
    model_image_size = 224
    device = torch.device("cpu")
    model_loaded = False
    if model_path.exists():
        model, class_names, model_image_size, _, device = load_model_once(model_path)
        model_loaded = True

    items = []
    total = 0
    for crop_path, region in zip(crop_paths, regions):
        forced_label = region.label or ""
        ignored = forced_label in IGNORE_LABELS
        if ignored:
            class_name = forced_label
            confidence = 1.0
            uncertain = True
        elif forced_label:
            class_name = forced_label
            confidence = 1.0
            uncertain = False
        elif model is not None:
            class_name, confidence = predict_crop(model, class_names, model_image_size, crop_path, device)
            uncertain = confidence < threshold
        else:
            class_name = "unknown"
            confidence = 0.0
            uncertain = True

        price_row = prices.get(class_name)
        price_info = dish_price(
            class_name,
            prices,
            uncertain=uncertain,
            egg_count=egg_count if class_name == THIT_KHO_TRUNG_CLASS else None,
        )
        total += price_info.total_price_vnd
        display_name = class_name if price_row is None else price_row.display_name
        items.append(
            {
                "crop_path": relative_or_absolute(crop_path),
                "crop_url": f"/file?path={relative_or_absolute(crop_path)}",
                "region_name": region.name,
                "class_name": class_name,
                "display_name": display_name,
                "confidence": round(confidence, 4),
                "uncertain": uncertain,
                "ignored": ignored,
                "price_vnd": price_info.total_price_vnd,
                "base_price_vnd": price_info.base_price_vnd,
                "extra_price_vnd": price_info.extra_price_vnd,
                "egg_count": price_info.egg_count,
            }
        )

    bill = {
        "image_path": relative_or_absolute(image_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": relative_or_absolute(model_path) if model_path.exists() else None,
        "threshold": threshold,
        "items": items,
        "total_vnd": total,
    }
    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    bill_path = BILLS_DIR / f"{image_path.stem}_{run_id}_bill.json"
    bill_path.write_text(json.dumps(bill, indent=2, ensure_ascii=False), encoding="utf-8")
    bill["bill_path"] = relative_or_absolute(bill_path)
    bill["model_loaded"] = model_loaded
    return bill


def save_upload(payload: dict) -> dict:
    name = Path(str(payload.get("name") or "upload.jpg")).name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    data_url = str(payload["data_url"])
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(base64.b64decode(encoded))
    return {"path": relative_or_absolute(path), "name": path.name}


def app_state() -> dict:
    prices = load_prices()
    return {
        "project_root": str(PROJECT_ROOT),
        "images": list_demo_images(),
        "templates": list_region_templates(),
        "classes": DISH_CLASSES,
        "labels": ["", "ignore", *DISH_CLASSES],
        "prices": {
            key: {"display_name": value.display_name, "price_vnd": value.price_vnd}
            for key, value in prices.items()
        },
        "default_model_path": relative_or_absolute(DEFAULT_MODEL_PATH),
    }


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Checkout Demo</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #1d232a;
      --muted: #65717f;
      --line: #d9dee5;
      --accent: #126a5a;
      --accent-2: #b8472f;
      --warn: #a76500;
      --ok: #087443;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: 330px minmax(420px, 1fr) 390px;
      min-height: calc(100vh - 56px);
    }
    aside, section {
      padding: 14px;
      border-right: 1px solid var(--line);
      overflow: auto;
    }
    section:last-child { border-right: 0; }
    .group {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .group h2 {
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 10px;
      letter-spacing: 0.06em;
    }
    label { display: block; color: var(--muted); font-size: 12px; margin: 8px 0 4px; }
    select, input, button {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 9px;
      font: inherit;
    }
    input[type="file"] { padding: 5px; }
    button {
      cursor: pointer;
      background: #ffffff;
      font-weight: 650;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.danger { color: var(--accent-2); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .toolbar { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .stage {
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr);
      gap: 12px;
    }
    .image-wrap {
      position: relative;
      width: 100%;
      max-height: calc(100vh - 168px);
      overflow: auto;
      background: #e8ecef;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .image-box {
      position: relative;
      display: inline-block;
      min-width: 100%;
      text-align: center;
      padding: 10px;
    }
    #trayImage { max-width: 100%; height: auto; display: block; margin: 0 auto; }
    .region {
      position: absolute;
      border: 2px solid var(--accent);
      background: rgba(18, 106, 90, 0.14);
      color: #fff;
      text-shadow: 0 1px 2px rgba(0,0,0,.45);
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: flex-start;
      padding: 4px;
      cursor: move;
      min-width: 24px;
      min-height: 24px;
    }
    .region.selected { border-color: var(--accent-2); background: rgba(184, 71, 47, 0.16); }
    .region.ignored { border-style: dashed; border-color: #5f6872; background: rgba(95,104,114,.16); }
    .region::after {
      content: "";
      position: absolute;
      width: 12px;
      height: 12px;
      right: -7px;
      bottom: -7px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid currentColor;
      cursor: nwse-resize;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 5px; text-align: left; vertical-align: middle; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    td input, td select { height: 30px; min-width: 0; }
    .small { color: var(--muted); font-size: 12px; }
    .bill-item {
      display: grid;
      grid-template-columns: 72px 1fr;
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }
    .bill-item img {
      width: 72px;
      height: 72px;
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid var(--line);
    }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef1f4;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-left: 5px;
    }
    .pill.warn { color: var(--warn); background: #fff2d6; }
    .pill.ok { color: var(--ok); background: #dff5e9; }
    .total {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      font-size: 20px;
      font-weight: 800;
      margin-top: 12px;
    }
    .status { color: var(--muted); }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      aside, section { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Checkout the Canteen</h1>
    <div class="status" id="status">Ready</div>
  </header>
  <main>
    <aside>
      <div class="group">
        <h2>Ảnh</h2>
        <label for="imageSelect">Ảnh demo</label>
        <select id="imageSelect"></select>
        <label for="uploadInput">Upload</label>
        <input id="uploadInput" type="file" accept="image/*">
        <div class="row">
          <button id="reloadBtn">Reload</button>
          <button id="loadImageBtn">Load</button>
        </div>
      </div>
      <div class="group">
        <h2>Crop</h2>
        <label for="templateSelect">Template</label>
        <select id="templateSelect"></select>
        <div class="toolbar">
          <button id="applyTemplateBtn">Apply</button>
          <button id="addRegionBtn">Add</button>
          <button id="deleteRegionBtn" class="danger">Delete</button>
        </div>
      </div>
      <div class="group">
        <h2>Model</h2>
        <label for="thresholdInput">Threshold</label>
        <input id="thresholdInput" type="number" min="0" max="1" step="0.01" value="0.55">
        <label for="eggCountInput">Egg count</label>
        <input id="eggCountInput" type="number" min="1" step="1" value="1">
        <button id="runBtn" class="primary">Run checkout</button>
      </div>
    </aside>
    <section class="stage">
      <div class="group">
        <h2>Vùng crop</h2>
        <table>
          <thead><tr><th>Name</th><th>Label</th><th>X</th><th>Y</th><th>W</th><th>H</th></tr></thead>
          <tbody id="regionRows"></tbody>
        </table>
      </div>
      <div class="image-wrap">
        <div class="image-box" id="imageBox">
          <img id="trayImage" alt="">
        </div>
      </div>
    </section>
    <section>
      <div class="group">
        <h2>Hóa đơn</h2>
        <div id="billList" class="small">Chưa chạy demo.</div>
        <div class="total"><span>Total</span><span id="totalValue">0 VND</span></div>
      </div>
      <div class="group">
        <h2>JSON</h2>
        <pre id="billJson" class="small"></pre>
      </div>
    </section>
  </main>
  <script>
    const state = { app: null, imagePath: "", imageWidth: 0, imageHeight: 0, regions: [], selected: -1, dragging: null };
    const $ = (id) => document.getElementById(id);
    const fmt = (n) => `${Number(n || 0).toLocaleString("en-US")} VND`;
    const setStatus = (text) => $("status").textContent = text;

    async function api(path, body=null) {
      const options = body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {};
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
      return data;
    }

    async function loadState() {
      const data = await api("/api/state");
      state.app = data;
      $("imageSelect").innerHTML = data.images.map(x => `<option value="${x.path}">${x.name}</option>`).join("");
      $("templateSelect").innerHTML = data.templates.map(x => `<option value="${x.path}">${x.name}</option>`).join("");
      if (data.images.length && !state.imagePath) {
        state.imagePath = data.images[0].path;
        $("imageSelect").value = state.imagePath;
        await loadImage(state.imagePath);
      }
    }

    async function loadImage(path) {
      state.imagePath = path;
      const meta = await api(`/api/image-info?path=${encodeURIComponent(path)}`);
      state.imageWidth = meta.width;
      state.imageHeight = meta.height;
      $("trayImage").src = `/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
      await applyTemplate();
      setStatus(path);
    }

    function scaleInfo() {
      const img = $("trayImage");
      const box = $("imageBox");
      const rendered = img.getBoundingClientRect();
      const outer = box.getBoundingClientRect();
      const sx = rendered.width / state.imageWidth;
      const sy = rendered.height / state.imageHeight;
      return { sx, sy, ox: rendered.left - outer.left, oy: rendered.top - outer.top };
    }

    function renderRegions() {
      const labels = state.app.labels.map(x => `<option value="${x}">${x || "model"}</option>`).join("");
      $("regionRows").innerHTML = state.regions.map((r, i) => `
        <tr data-i="${i}">
          <td><input data-k="name" value="${r.name || ""}"></td>
          <td><select data-k="label">${labels}</select></td>
          <td><input data-k="x" type="number" value="${r.x}"></td>
          <td><input data-k="y" type="number" value="${r.y}"></td>
          <td><input data-k="w" type="number" value="${r.w}"></td>
          <td><input data-k="h" type="number" value="${r.h}"></td>
        </tr>`).join("");
      [...$("regionRows").querySelectorAll("tr")].forEach((tr) => {
        const i = Number(tr.dataset.i);
        tr.querySelector("select").value = state.regions[i].label || "";
        tr.onclick = () => { state.selected = i; drawOverlay(); };
        tr.querySelectorAll("input,select").forEach((el) => {
          el.oninput = () => {
            const k = el.dataset.k;
            state.regions[i][k] = ["x","y","w","h"].includes(k) ? Number(el.value || 0) : el.value;
            drawOverlay(false);
          };
        });
      });
      drawOverlay(false);
    }

    function drawOverlay(syncRows=true) {
      const box = $("imageBox");
      [...box.querySelectorAll(".region")].forEach(x => x.remove());
      const {sx, sy, ox, oy} = scaleInfo();
      state.regions.forEach((r, i) => {
        const div = document.createElement("div");
        div.className = `region ${i === state.selected ? "selected" : ""} ${r.label === "ignore" ? "ignored" : ""}`;
        div.style.left = `${ox + r.x * sx}px`;
        div.style.top = `${oy + r.y * sy}px`;
        div.style.width = `${r.w * sx}px`;
        div.style.height = `${r.h * sy}px`;
        div.textContent = r.label || r.name || `crop_${i}`;
        div.onpointerdown = (e) => {
          state.selected = i;
          const rect = div.getBoundingClientRect();
          const resize = e.clientX > rect.right - 16 && e.clientY > rect.bottom - 16;
          state.dragging = { i, resize, x:e.clientX, y:e.clientY, r:{...r} };
          div.setPointerCapture(e.pointerId);
          drawOverlay(false);
        };
        div.onpointermove = (e) => {
          const d = state.dragging;
          if (!d || d.i !== i) return;
          const dx = (e.clientX - d.x) / sx;
          const dy = (e.clientY - d.y) / sy;
          if (d.resize) {
            r.w = Math.max(8, Math.round(d.r.w + dx));
            r.h = Math.max(8, Math.round(d.r.h + dy));
          } else {
            r.x = Math.max(0, Math.min(state.imageWidth - 1, Math.round(d.r.x + dx)));
            r.y = Math.max(0, Math.min(state.imageHeight - 1, Math.round(d.r.y + dy)));
          }
          drawOverlay();
        };
        div.onpointerup = () => { state.dragging = null; renderRegions(); };
        box.appendChild(div);
      });
      if (syncRows) renderRegions();
    }

    async function applyTemplate() {
      const template = $("templateSelect").value;
      const payload = { image_path: state.imagePath, template };
      const data = await api("/api/regions", payload);
      state.regions = data.regions;
      state.selected = state.regions.length ? 0 : -1;
      renderRegions();
    }

    async function runCheckout() {
      setStatus("Running...");
      const payload = {
        image_path: state.imagePath,
        regions: state.regions,
        threshold: Number($("thresholdInput").value || 0.55),
        egg_count: Number($("eggCountInput").value || 1),
      };
      const bill = await api("/api/run", payload);
      $("totalValue").textContent = fmt(bill.total_vnd);
      $("billList").innerHTML = bill.items.map((item, idx) => `
        <div class="bill-item">
          <img src="${item.crop_url}">
          <div>
            <strong>${idx + 1}. ${item.display_name}</strong>
            ${item.ignored ? '<span class="pill">ignored</span>' : item.uncertain ? '<span class="pill warn">uncertain</span>' : '<span class="pill ok">ok</span>'}
            <div>${fmt(item.price_vnd)} · conf=${Number(item.confidence).toFixed(2)}</div>
            <div class="small">${item.region_name} · ${item.class_name}</div>
          </div>
        </div>`).join("");
      $("billJson").textContent = JSON.stringify(bill, null, 2);
      setStatus(`Bill: ${bill.bill_path}`);
    }

    $("reloadBtn").onclick = loadState;
    $("loadImageBtn").onclick = () => loadImage($("imageSelect").value);
    $("imageSelect").onchange = () => loadImage($("imageSelect").value);
    $("applyTemplateBtn").onclick = applyTemplate;
    $("addRegionBtn").onclick = () => {
      state.regions.push({name:`crop_${state.regions.length}`, x:20, y:20, w:160, h:120, label:""});
      state.selected = state.regions.length - 1;
      renderRegions();
    };
    $("deleteRegionBtn").onclick = () => {
      if (state.selected >= 0) state.regions.splice(state.selected, 1);
      state.selected = Math.min(state.selected, state.regions.length - 1);
      renderRegions();
    };
    $("runBtn").onclick = () => runCheckout().catch(err => { setStatus(err.message); alert(err.message); });
    $("uploadInput").onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        const saved = await api("/api/upload", { name: file.name, data_url: reader.result });
        await loadState();
        $("imageSelect").value = saved.path;
        await loadImage(saved.path);
      };
      reader.readAsDataURL(file);
    };
    $("trayImage").onload = () => drawOverlay(false);
    window.addEventListener("resize", () => drawOverlay(false));
    loadState().catch(err => { setStatus(err.message); alert(err.message); });
  </script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CanteenDemo/1.0"

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
                self.send_json(app_state())
                return
            if parsed.path == "/api/image-info":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                width, height = image_size(path)
                self.send_json({"width": width, "height": height})
                return
            if parsed.path == "/file":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                if not path.exists() or not is_safe_project_path(path):
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
            if parsed.path == "/api/upload":
                self.send_json({"ok": True, **save_upload(payload)})
                return
            if parsed.path == "/api/regions":
                image_path = resolve_project_path(str(payload["image_path"]))
                regions = regions_from_payload(payload, image_path)
                self.send_json(
                    {
                        "ok": True,
                        "regions": [
                            {"name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "label": r.label or ""}
                            for r in regions
                        ],
                    }
                )
                return
            if parsed.path == "/api/run":
                self.send_json({"ok": True, **run_checkout(payload)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local web UI for checkout demos.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo checkout app: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
