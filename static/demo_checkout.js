const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";

const state = {
  app: null,
  imagePath: "",
  imageWidth: 0,
  imageHeight: 0,
  regions: [],
  selected: -1,
  drag: null,
  zoom: 1,
  total: 0,
  regionMeta: {},
  customer: null,
  currentBill: null,
  selectedVoucherId: "",
};

function setStatus(text) {
  const el = $("status");
  if (el) el.textContent = text;
}

function setLoading(active) {
  $("loading").classList.toggle("hidden", !active);
}

function setInlineMessage(id, text, kind = "") {
  const el = $(id);
  if (!el) return;
  el.textContent = text || "";
  el.className = `inline-message${kind ? ` ${kind}` : ""}`;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function fmtVnd(value) {
  return `${Number(value || 0).toLocaleString("en-US")} VND`;
}

async function api(url, body = null) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function optionHtml(items, valueKey = "path", labelKey = "name") {
  return items.map((item) => `<option value="${esc(item[valueKey])}">${esc(item[labelKey])}</option>`).join("");
}

async function init() {
  state.app = await api("/api/state");
  $("imageSelect").innerHTML = optionHtml(state.app.images);
  $("templateSelect").innerHTML = optionHtml(state.app.templates);
  $("rewardSelect").innerHTML = (state.app.reward_catalog || []).map((reward) => (
    `<option value="${esc(reward.class_name)}">${esc(reward.display_name)} · ${reward.points_cost} điểm</option>`
  )).join("");
  renderCustomer();
  const badge = $("modelBadge");
  if (badge) {
    const classifierReady = state.app.default_model_path ? "Classifier ready" : "Classifier missing";
    const detectorReady = state.app.default_detector_path ? "Detector ready" : "Detector missing";
    const regionReady = state.app.default_region_detector_path ? "Region AI ready" : "Region AI missing";
    badge.textContent = `${classifierReady} · ${regionReady} · ${detectorReady}`;
  }

  if (state.app.images.length) {
    await loadImage(state.app.images[0].path);
  } else {
    setStatus("No demo images found");
  }
}

function renderCustomer(message = "", kind = "") {
  const summary = $("memberSummary");
  const createButton = $("createCustomerBtn");
  if (!state.customer) {
    summary.classList.add("hidden");
    if (!message) createButton.classList.add("hidden");
    setInlineMessage("customerMessage", message || "Chưa chọn thành viên", kind);
    state.selectedVoucherId = "";
    return;
  }

  createButton.classList.add("hidden");
  summary.classList.remove("hidden");
  $("memberPhone").textContent = state.customer.phone_masked;
  $("memberPoints").textContent = `${state.customer.points_balance} điểm`;
  const activeVouchers = state.customer.active_vouchers || [];
  if (!activeVouchers.some((voucher) => voucher.id === state.selectedVoucherId)) {
    state.selectedVoucherId = "";
  }
  $("voucherSelect").innerHTML = [
    '<option value="">Không dùng voucher</option>',
    ...activeVouchers.map((voucher) => (
      `<option value="${esc(voucher.id)}">${esc(voucher.display_name)} · −${fmtVnd(voucher.discount_vnd)}</option>`
    )),
  ].join("");
  $("voucherSelect").value = state.selectedVoucherId;
  setInlineMessage("customerMessage", message || `${activeVouchers.length} voucher khả dụng`, kind || "success");
}

async function lookupCustomer() {
  const phone = $("phoneInput").value.trim();
  const data = await api("/api/customers/lookup", { phone });
  if (data.found) {
    state.customer = data.customer;
    $("phoneInput").value = "";
    renderCustomer("Đã tải tài khoản thành viên", "success");
  } else {
    state.customer = null;
    state.selectedVoucherId = "";
    $("createCustomerBtn").classList.remove("hidden");
    renderCustomer("Chưa có thành viên. Chọn Tạo mới để đăng ký.");
    $("createCustomerBtn").classList.remove("hidden");
  }
}

async function createCustomer() {
  const phone = $("phoneInput").value.trim();
  const data = await api("/api/customers", { phone });
  state.customer = data.customer;
  $("phoneInput").value = "";
  renderCustomer(data.created ? "Đã tạo thành viên" : "Thành viên đã tồn tại", "success");
}

function clearCustomer() {
  state.customer = null;
  state.selectedVoucherId = "";
  clearBill();
  renderCustomer("Đã chuyển sang khách vãng lai");
}

async function loadImage(path) {
  state.imagePath = path;
  setStatus("Loading image...");
  const meta = await api(`/api/image-info?path=${encodeURIComponent(path)}`);
  state.imageWidth = meta.width;
  state.imageHeight = meta.height;

  await new Promise((resolve, reject) => {
    const image = $("trayImage");
    image.onload = resolve;
    image.onerror = () => reject(new Error("Could not load tray image"));
    image.src = `/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  });

  const titleEl = $("trayTitle");
  if (titleEl) titleEl.textContent = path.split(/[\\/]/).pop();
  state.zoom = 1;
  $("zoomInput").value = "1";
  await applyAutoRegions();
}

function fitWidth() {
  // Image is CSS width:100% by default — get rendered width
  const img = $("trayImage");
  if (!img.naturalWidth) return Math.max(320, $("viewport").clientWidth - 30);
  return img.clientWidth || img.naturalWidth;
}

function applyZoom() {
  const image = $("trayImage");
  if (state.zoom === 1) {
    // Default: let CSS handle it (width: 100%)
    image.style.width = "";
    image.style.maxWidth = "100%";
  } else {
    // Zoomed: override CSS to fixed pixel width
    const baseWidth = $("viewport").clientWidth - 30;
    image.style.width = `${Math.round(baseWidth * state.zoom)}px`;
    image.style.maxWidth = "none";
  }
  positionOverlay();
}

function positionOverlay() {
  if (!state.imageWidth || !$("trayImage").complete) return;
  const image = $("trayImage");
  const box = $("imageBox");
  const overlay = $("overlay");
  const imageRect = image.getBoundingClientRect();
  const boxRect = box.getBoundingClientRect();
  overlay.style.left = `${imageRect.left - boxRect.left + box.scrollLeft}px`;
  overlay.style.top = `${imageRect.top - boxRect.top + box.scrollTop}px`;
  overlay.style.width = `${imageRect.width}px`;
  overlay.style.height = `${imageRect.height}px`;
  overlay.setAttribute("viewBox", `0 0 ${state.imageWidth} ${state.imageHeight}`);
}

function snap(value) {
  return Math.round(value / 4) * 4;
}

function clampRegion(region) {
  const minSize = 24;
  region.x = Math.max(0, Math.min(state.imageWidth - minSize, Math.round(region.x)));
  region.y = Math.max(0, Math.min(state.imageHeight - minSize, Math.round(region.y)));
  region.w = Math.max(minSize, Math.min(state.imageWidth - region.x, Math.round(region.w)));
  region.h = Math.max(minSize, Math.min(state.imageHeight - region.y, Math.round(region.h)));
}

function pointInImage(event) {
  const point = $("overlay").createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const transformed = point.matrixTransform($("overlay").getScreenCTM().inverse());
  return { x: Math.round(transformed.x), y: Math.round(transformed.y) };
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function handleSize() {
  const overlay = $("overlay");
  if (!overlay || !state.imageWidth) return 16;
  const displayWidth = overlay.clientWidth || state.imageWidth;
  const scale = state.imageWidth / displayWidth;
  return Math.round(18 * scale); // 18px visual size on screen
}

function addHandle(group, region, index, mode, x, y, size) {
  const handle = svgEl("rect", {
    class: "handle",
    "data-i": index,
    "data-mode": mode,
    x,
    y,
    width: size,
    height: size,
    rx: 9,
  });
  group.appendChild(handle);
}

function draw(syncRows = true) {
  applyZoom();
  const overlay = $("overlay");
  overlay.replaceChildren();

  state.regions.forEach((region, index) => {
    clampRegion(region);
    const group = svgEl("g", { "data-i": index });
    const selected = index === state.selected;
    const ignored = region.label === "ignore";
    const size = selected ? handleSize() : 0;
    const rect = svgEl("rect", {
      class: `region-rect${selected ? " selected" : ""}${ignored ? " ignored" : ""}`,
      "data-i": index,
      "data-mode": "move",
      x: region.x,
      y: region.y,
      width: region.w,
      height: region.h,
      rx: 12,
    });
    group.appendChild(rect);

    // Removing region label rendering (top_left, bottom_right, etc.)
    // as requested by the user.

    if (selected) {
      addHandle(group, region, index, "nw", region.x - size / 2, region.y - size / 2, size);
      addHandle(group, region, index, "ne", region.x + region.w - size / 2, region.y - size / 2, size);
      addHandle(group, region, index, "sw", region.x - size / 2, region.y + region.h - size / 2, size);
      addHandle(group, region, index, "se", region.x + region.w - size / 2, region.y + region.h - size / 2, size);
    }
    overlay.appendChild(group);
  });

  $("regionCount").textContent = `${state.regions.length} crops`;
  if (syncRows) renderRows();
}

function labelOptions(selected) {
  return state.app.labels.map((label) => {
    const display = label || "model";
    return `<option value="${esc(label)}" ${label === selected ? "selected" : ""}>${esc(display)}</option>`;
  }).join("");
}

function renderRows() {
  $("regionRows").innerHTML = state.regions.map((region, index) => `
    <tr data-index="${index}" class="${index === state.selected ? "active" : ""}">
      <td><input data-key="name" value="${esc(region.name || "")}"></td>
      <td><select data-key="label">${labelOptions(region.label || "")}</select></td>
      <td><input data-key="x" type="number" value="${region.x}"></td>
      <td><input data-key="y" type="number" value="${region.y}"></td>
      <td><input data-key="w" type="number" value="${region.w}"></td>
      <td><input data-key="h" type="number" value="${region.h}"></td>
    </tr>
  `).join("");

  $("regionRows").querySelectorAll("tr").forEach((row) => {
    const index = Number(row.dataset.index);
    row.addEventListener("click", () => {
      state.selected = index;
      draw();
    });
    row.querySelectorAll("input,select").forEach((input) => {
      input.addEventListener("input", () => {
        const key = input.dataset.key;
        state.regions[index][key] = ["x", "y", "w", "h"].includes(key) ? Number(input.value || 0) : input.value;
        clampRegion(state.regions[index]);
        draw(false);
      });
    });
  });
}

async function applyTemplate() {
  if (!state.imagePath) return;
  const data = await api("/api/regions", {
    image_path: state.imagePath,
    mode: "template",
    template: $("templateSelect").value,
  });
  state.regions = data.regions;
  state.regionMeta = data;
  state.selected = state.regions.length ? 0 : -1;
  draw();
  setStatus("Grid regions ready");
}

async function applyAutoRegions() {
  if (!state.imagePath) return;
  setStatus("Detecting food regions...");
  const data = await api("/api/regions", {
    image_path: state.imagePath,
    mode: "auto",
    region_threshold: 0.35,
    template: $("templateSelect").value,
  });
  state.regions = data.regions;
  state.regionMeta = data;
  state.selected = state.regions.length ? 0 : -1;
  draw();
  if (data.region_source === "auto_detector") {
    setStatus(`AI found ${state.regions.length} food regions`);
  } else {
    setStatus(`Using grid fallback: ${data.fallback_reason || "detector unavailable"}`);
  }
}

function addRegion() {
  if (!state.imageWidth) return;
  const width = Math.round(state.imageWidth * 0.23);
  const height = Math.round(state.imageHeight * 0.18);
  state.regions.push({
    name: `crop_${state.regions.length + 1}`,
    x: Math.round((state.imageWidth - width) / 2),
    y: Math.round((state.imageHeight - height) / 2),
    w: width,
    h: height,
    label: "",
  });
  state.selected = state.regions.length - 1;
  draw();
}

function duplicateRegion() {
  if (state.selected < 0) return;
  const copy = { ...state.regions[state.selected] };
  copy.name = `${copy.name || "crop"}_copy`;
  copy.x += 28;
  copy.y += 28;
  clampRegion(copy);
  state.regions.push(copy);
  state.selected = state.regions.length - 1;
  draw();
}

function deleteRegion() {
  if (state.selected < 0) return;
  state.regions.splice(state.selected, 1);
  state.selected = Math.min(state.selected, state.regions.length - 1);
  draw();
}

function toggleIgnore() {
  if (state.selected < 0) return;
  const region = state.regions[state.selected];
  region.label = region.label === "ignore" ? "" : "ignore";
  draw();
}

function resizeRegion(region, start, mode, dx, dy) {
  if (mode === "move") {
    region.x = snap(start.x + dx);
    region.y = snap(start.y + dy);
    return;
  }
  let x = start.x;
  let y = start.y;
  let w = start.w;
  let h = start.h;
  if (mode.includes("e")) w = start.w + dx;
  if (mode.includes("s")) h = start.h + dy;
  if (mode.includes("w")) {
    x = start.x + dx;
    w = start.w - dx;
  }
  if (mode.includes("n")) {
    y = start.y + dy;
    h = start.h - dy;
  }
  region.x = snap(x);
  region.y = snap(y);
  region.w = snap(w);
  region.h = snap(h);
}

$("overlay").addEventListener("pointerdown", (event) => {
  const target = event.target.closest("[data-i]");
  if (!target) return;
  const index = Number(target.dataset.i);
  state.selected = index;
  state.drag = {
    index,
    mode: target.dataset.mode || "move",
    point: pointInImage(event),
    region: { ...state.regions[index] },
  };
  $("overlay").setPointerCapture(event.pointerId);
  draw();
  event.preventDefault();
});

$("overlay").addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const point = pointInImage(event);
  const drag = state.drag;
  const region = state.regions[drag.index];
  resizeRegion(region, drag.region, drag.mode, point.x - drag.point.x, point.y - drag.point.y);
  clampRegion(region);
  draw(false);
});

$("overlay").addEventListener("pointerup", () => {
  state.drag = null;
  renderRows();
});

function animateTotal(toValue) {
  const fromValue = state.total || 0;
  const duration = 520;
  const start = performance.now();
  state.total = toValue;
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = Math.round(fromValue + (toValue - fromValue) * eased);
    $("totalValue").textContent = fmtVnd(current);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Nutrition advice engine ── */
// Map keywords found in class_name / display_name to nutrient groups
const NUTRIENT_MAP = {
  // ── Protein sources (Vietnamese food names) ──
  thit:         { protein: true },           // thịt kho, thịt kho trứng
  suon:         { protein: true },           // sườn nướng
  ca_hu:        { protein: true },           // cá hú kho
  ca:           { protein: true },           // cá (generic)
  trung:        { protein: true },           // trứng chiên
  dau_hu:       { protein: true },           // đậu hũ sốt cà
  // ── Vietnamese veggie / fibre ──
  rau:          { veggie: true },            // rau xào, canh rau
  canh:         { veggie: true },            // canh chua, canh rau
  // ── Carbs ──
  com:          { carb: true },              // cơm trắng
  bun:          { carb: true },
  mi:           { carb: true },
  // ── English fallback ──
  egg:          { protein: true },
  chicken:      { protein: true },
  pork:         { protein: true },
  fish:         { protein: true },
  rice:         { carb: true },
  vegetable:    { veggie: true },
};

function analyseNutrition(items) {
  let hasProtein = false, hasVeg = false, hasCarb = false;
  items.forEach(item => {
    if (item.ignored) return;
    const name = ((item.class_name || "") + " " + (item.display_name || "")).toLowerCase();
    for (const [keyword, flags] of Object.entries(NUTRIENT_MAP)) {
      if (name.includes(keyword)) {
        if (flags.protein) hasProtein = true;
        if (flags.veggie)  hasVeg     = true;
        if (flags.carb)    hasCarb    = true;
      }
    }
  });
  const tips = [];
  if (hasProtein && hasVeg && hasCarb) {
    tips.push({ icon: "✅", color: "ok", text: "Bữa ăn cân bằng! Đủ đạm, rau và tinh bột — tuyệt vời! 🎉" });
  } else {
    if (!hasProtein) tips.push({ icon: "🥩", color: "warn", text: "Bữa ăn thiếu đạm — hãy thêm thịt, cá, trứng hoặc đậu hũ!" });
    if (!hasVeg)     tips.push({ icon: "🥦", color: "warn", text: "Thiếu rau xanh — thêm rau củ để bổ sung chất xơ và vitamin nhé!" });
    if (!hasCarb)    tips.push({ icon: "🍚", color: "warn", text: "Chưa có tinh bột — cơm/bún/mì giúp cung cấp năng lượng cho buổi chiều!" });
    if (hasProtein && hasVeg)
      tips.push({ icon: "💪", color: "ok", text: "Đủ đạm và rau — khá tốt! Thêm chút cơm/bún để có đủ năng lượng." });
    else if (hasProtein && hasCarb)
      tips.push({ icon: "💪", color: "ok", text: "Đủ đạm và tinh bột — thêm rau để bữa ăn hoàn hảo hơn!" });
    else if (hasVeg && hasCarb)
      tips.push({ icon: "💪", color: "ok", text: "Có rau và cơm — bổ sung thêm đạm (thịt/cá/trứng) nha!" });
  }
  return tips;
}

function renderNutrition(items) {
  const el = $("nutritionAdvice");
  if (!el) return;
  const tips = analyseNutrition(items);
  el.innerHTML = tips.map(t => `
    <div class="nutrition-tip tip-${t.color}">
      <span class="tip-icon">${t.icon}</span>
      <span>${t.text}</span>
    </div>
  `).join("");
  el.style.display = "block";
}

/* ── QR Code (via qrcode.js CDN) ── */
function renderQR(bill) {
  const el = $("qrSection");
  if (!el) return;
  const qrContainer = $("qrCanvas");
  if (!qrContainer) return;
  qrContainer.innerHTML = "";

  const text = `CANTEEN BILL\nTotal: ${fmtVnd(bill.net_total_vnd ?? bill.total_vnd)}\nItems: ${bill.items.filter(i=>!i.ignored).length}\nRef: ${bill.bill_id || "checkout"}\nTime: ${new Date().toLocaleString("vi-VN")}`;

  // Use QRious or fallback to a QR API
  if (window.QRious) {
    const canvas = document.createElement("canvas");
    qrContainer.appendChild(canvas);
    new QRious({ element: canvas, value: text, size: 180, backgroundAlpha: 0, foreground: "#3a5299", level: "H" });
  } else {
    const img = document.createElement("img");
    img.src = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(text)}&size=180x180&color=3a5299&bgcolor=ffffff&ecc=H`;
    img.alt = "Payment QR code";
    img.width = 180;
    qrContainer.appendChild(img);
  }
  el.style.display = "block";
}

function ratingSummaryText(summary) {
  const count = Number(summary?.count || 0);
  return count ? `★ ${Number(summary.average).toFixed(1)} · ${count} lượt` : "Chưa có đánh giá";
}

function ratingMarkup(item, paid) {
  const rateable = paid && !item.ignored && !item.uncertain && Number(item.price_vnd || 0) > 0;
  if (!rateable) return "";
  const selected = Number(item.rating?.stars || 0);
  const stars = [1, 2, 3, 4, 5].map((value) => (
    `<button class="star-btn${value <= selected ? " active" : ""}" type="button" data-rating-star="${value}" aria-label="${value} sao">★</button>`
  )).join("");
  return `
    <div class="rating-block" data-item-id="${esc(item.bill_item_id)}" data-stars="${selected}">
      <div class="rating-controls">
        ${stars}
        <button class="btn btn-secondary rating-save" type="button" data-save-rating>Lưu</button>
      </div>
      <textarea class="rating-comment" maxlength="500" placeholder="Nhận xét tùy chọn">${esc(item.rating?.comment || "")}</textarea>
      <div class="rating-status">${selected ? "Đã lưu đánh giá" : ""}</div>
    </div>
  `;
}

function wireRatingControls() {
  document.querySelectorAll(".rating-block").forEach((block) => {
    block.querySelectorAll("[data-rating-star]").forEach((button) => {
      button.addEventListener("click", () => {
        const stars = Number(button.dataset.ratingStar);
        block.dataset.stars = String(stars);
        block.querySelectorAll("[data-rating-star]").forEach((candidate) => {
          candidate.classList.toggle("active", Number(candidate.dataset.ratingStar) <= stars);
        });
      });
    });
    block.querySelector("[data-save-rating]").addEventListener("click", () => submitRating(block).catch(showError));
  });
}

async function submitRating(block) {
  const stars = Number(block.dataset.stars || 0);
  const status = block.querySelector(".rating-status");
  if (!stars) {
    status.textContent = "Hãy chọn số sao";
    return;
  }
  const data = await api("/api/ratings", {
    bill_item_id: block.dataset.itemId,
    customer_id: state.customer?.id || null,
    stars,
    comment: block.querySelector(".rating-comment").value,
  });
  const item = state.currentBill?.items.find((candidate) => candidate.bill_item_id === block.dataset.itemId);
  if (item) {
    item.rating = data.rating;
    item.rating_summary = data.summary;
  }
  const summary = block.closest(".bill-item").querySelector(".rating-summary");
  if (summary) summary.textContent = ratingSummaryText(data.summary);
  status.textContent = "Đã lưu đánh giá";
}

function renderRewardPanel() {
  const panel = $("rewardPanel");
  const eligible = state.currentBill?.status === "paid" && state.customer && state.currentBill.customer_id === state.customer.id;
  panel.classList.toggle("hidden", !eligible);
  if (!eligible) return;
  const alreadyIssued = Boolean(state.currentBill.voucher_issued);
  $("issueVoucherBtn").disabled = alreadyIssued;
  setInlineMessage(
    "rewardMessage",
    alreadyIssued ? "Bill này đã đổi một voucher" : `Số dư hiện tại: ${state.customer.points_balance} điểm`,
    alreadyIssued ? "success" : "",
  );
}

function renderBill(bill) {
  state.currentBill = bill;
  if (bill.customer) {
    state.customer = bill.customer;
    renderCustomer("Đã cập nhật điểm và voucher", "success");
  }
  const netTotal = Number(bill.net_total_vnd ?? bill.total_vnd ?? 0);
  animateTotal(netTotal);
  const billMetaEl = $("billMeta");
  if (billMetaEl) {
    const status = bill.status === "paid" ? "Đã thanh toán" : "Bill nháp";
    billMetaEl.textContent = `${bill.items.filter(i=>!i.ignored).length} món · ${status}`;
  }
  const jsonEl = $("billJson");
  if (jsonEl) jsonEl.textContent = JSON.stringify(bill, null, 2);
  $("billBreakdown").classList.remove("hidden");
  $("grossValue").textContent = fmtVnd(bill.gross_total_vnd ?? netTotal);
  $("discountValue").textContent = `−${fmtVnd(bill.discount_vnd || 0)}`;
  $("confirmPaymentBtn").classList.toggle("hidden", bill.status !== "draft");
  if (bill.status === "paid") {
    const message = bill.customer
      ? `Đã cộng ${bill.earned_points} điểm · Số dư ${bill.customer.points_balance} điểm`
      : "Đã xác nhận thanh toán ẩn danh";
    setInlineMessage("paymentMessage", message, "success");
  } else {
    setInlineMessage("paymentMessage", "Điểm và voucher chỉ được ghi khi xác nhận");
  }

  $("billList").classList.remove("empty-state");
  $("billList").innerHTML = bill.items.map((item, index) => {
    const confidence = Math.max(0, Math.min(1, Number(item.confidence || 0)));
    const confPct = Math.round(confidence * 100);
    const raw = item.raw_class_name && item.raw_class_name !== item.class_name
      ? `${item.raw_class_name} → ${item.class_name}`
      : item.class_name;
    const evidence = [];
    const detectorEgg = Number(item.detector_evidence?.egg_count || 0);
    if (detectorEgg > 0) evidence.push(`egg=${detectorEgg}`);
    if (Number(item.fish_count || 0) > 0) evidence.push(`fish=${item.fish_count}`);
    if (item.fusion_reason && item.fusion_reason !== "classifier_only") evidence.push(item.fusion_reason);
    const tag = item.ignored
      ? '<span class="tag muted">bỏ qua</span>'
      : item.uncertain
        ? '<span class="tag warn">chưa chắc</span>'
        : '<span class="tag ok">✓</span>';
    const finalPrice = Number(item.final_price_vnd ?? item.price_vnd ?? 0);
    const price = Number(item.discount_vnd || 0) > 0
      ? `<s>${fmtVnd(item.price_vnd)}</s>${fmtVnd(finalPrice)}`
      : fmtVnd(finalPrice);
    const summary = item.rating_summary || state.app.rating_summaries?.[item.class_name] || { average: 0, count: 0 };
    return `
      <article class="bill-item">
        <img src="${esc(item.crop_url)}" alt="${esc(item.display_name)} crop">
        <div>
          <strong>${index + 1}. ${esc(item.display_name)} ${tag}</strong>
          <div class="bill-price">${price}</div>
          <div class="confidence"><span style="width:${confPct}%"></span></div>
          <div class="subline">${esc(raw)} · ${confPct}%${evidence.length ? " · " + esc(evidence.join(" · ")) : ""}</div>
          <div class="rating-summary">${ratingSummaryText(summary)}</div>
          ${ratingMarkup(item, bill.status === "paid")}
        </div>
      </article>
    `;
  }).join("");
  wireRatingControls();
  $("cropStrip").innerHTML = bill.items.map((item) => (
    `<img src="${esc(item.crop_url)}" title="${esc(item.class_name)}" alt="${esc(item.class_name)} crop">`
  )).join("");

  renderNutrition(bill.items);
  renderQR(bill);
  renderRewardPanel();
}

async function runCheckout() {
  if (!state.imagePath) return;
  setLoading(true);
  setStatus("Running checkout...");
  try {
    const thresholdInput = $("thresholdInput");
    const bill = await api("/api/run", {
      image_path: state.imagePath,
      regions: state.regions,
      region_mode: state.regionMeta.requested_mode || "manual",
      region_metadata: state.regionMeta,
      region_threshold: 0.35,
      threshold: thresholdInput ? Number(thresholdInput.value || 0.55) : 0.55,
      use_detector: true,
      detector_threshold: 0.25,
      customer_id: state.customer?.id || null,
      voucher_id: state.selectedVoucherId || null,
    });
    renderBill(bill);
    setStatus(`✅ Đã nhận diện xong!`);
  } finally {
    setLoading(false);
  }
}

async function confirmPayment() {
  if (!state.currentBill?.bill_id || state.currentBill.status !== "draft") return;
  setLoading(true);
  setStatus("Đang xác nhận thanh toán...");
  try {
    const bill = await api("/api/checkout/confirm", { bill_id: state.currentBill.bill_id });
    renderBill(bill);
    setStatus("Đã xác nhận thanh toán");
  } finally {
    setLoading(false);
  }
}

async function issueVoucher() {
  if (!state.customer || state.currentBill?.status !== "paid") return;
  const data = await api("/api/vouchers", {
    customer_id: state.customer.id,
    source_bill_id: state.currentBill.bill_id,
    class_name: $("rewardSelect").value,
  });
  state.customer = data.customer;
  state.currentBill.customer = data.customer;
  state.currentBill.voucher_issued = true;
  renderBill(state.currentBill);
  renderCustomer(`Đã đổi voucher ${data.voucher.display_name}`, "success");
  setStatus("Voucher đã sẵn sàng cho bill sau");
}

function clearBill() {
  state.total = 0;
  state.currentBill = null;
  $("totalValue").textContent = "0 VND";
  const billMetaEl = $("billMeta");
  if (billMetaEl) billMetaEl.textContent = "Waiting for checkout";
  $("billBreakdown").classList.add("hidden");
  $("confirmPaymentBtn").classList.add("hidden");
  $("rewardPanel").classList.add("hidden");
  setInlineMessage("paymentMessage", "");
  $("billList").className = "bill-list empty-state";
  $("billList").textContent = "No items yet";
  const jsonEl = $("billJson");
  if (jsonEl) jsonEl.textContent = "{}";
  $("cropStrip").replaceChildren();
  const nutr = $("nutritionAdvice");
  if (nutr) { nutr.innerHTML = ""; nutr.style.display = "none"; }
  const qrSec = $("qrSection");
  if (qrSec) qrSec.style.display = "none";
}

function nudgeSelected(event) {
  if (state.selected < 0 || ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  const region = state.regions[state.selected];
  const step = event.shiftKey ? 16 : 4;
  if (event.key === "ArrowLeft") region.x -= step;
  else if (event.key === "ArrowRight") region.x += step;
  else if (event.key === "ArrowUp") region.y -= step;
  else if (event.key === "ArrowDown") region.y += step;
  else if (event.key === "Delete") deleteRegion();
  else return;
  event.preventDefault();
  clampRegion(region);
  draw();
}

$("loadImageBtn").addEventListener("click", () => loadImage($("imageSelect").value).catch(showError));
$("imageSelect").addEventListener("change", () => loadImage($("imageSelect").value).catch(showError));
$("applyTemplateBtn").addEventListener("click", () => applyTemplate().catch(showError));
$("autoRegionBtn").addEventListener("click", () => applyAutoRegions().catch(showError));
$("addRegionBtn").addEventListener("click", addRegion);
$("duplicateRegionBtn").addEventListener("click", duplicateRegion);
$("deleteRegionBtn").addEventListener("click", deleteRegion);
$("ignoreRegionBtn").addEventListener("click", toggleIgnore);
$("fitBtn").addEventListener("click", () => {
  state.zoom = 1;
  $("zoomInput").value = "1";
  draw(false);
});
$("clearBillBtn").addEventListener("click", clearBill);
$("runBtn").addEventListener("click", () => runCheckout().catch(showError));
$("lookupCustomerBtn").addEventListener("click", () => lookupCustomer().catch(showError));
$("createCustomerBtn").addEventListener("click", () => createCustomer().catch(showError));
$("clearCustomerBtn").addEventListener("click", clearCustomer);
$("confirmPaymentBtn").addEventListener("click", () => confirmPayment().catch(showError));
$("issueVoucherBtn").addEventListener("click", () => issueVoucher().catch(showError));
$("voucherSelect").addEventListener("change", (event) => {
  state.selectedVoucherId = event.target.value;
  clearBill();
  setStatus(state.selectedVoucherId ? "Voucher sẽ áp dụng ở lần Predict tiếp theo" : "Đã bỏ chọn voucher");
});
$("phoneInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") lookupCustomer().catch(showError);
});
$("zoomInput").addEventListener("input", (event) => {
  state.zoom = Number(event.target.value);
  draw(false);
});
$("viewport").addEventListener("scroll", positionOverlay);
window.addEventListener("resize", () => draw(false));
document.addEventListener("keydown", nudgeSelected);

$("uploadInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const saved = await api("/api/upload", { name: file.name, data_url: reader.result });
      state.app.images.unshift(saved);
      $("imageSelect").innerHTML = optionHtml(state.app.images);
      $("imageSelect").value = saved.path;
      await loadImage(saved.path);
    } catch (error) {
      showError(error);
    }
  };
  reader.readAsDataURL(file);
});

function showError(error) {
  setLoading(false);
  setStatus(error.message);
  alert(error.message);
}

init().catch(showError);
