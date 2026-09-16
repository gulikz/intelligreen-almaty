"use strict";
/* IntelliGreen Almaty — геопортал.
   Модель XGBoost выгружена в JSON и выполняется в браузере; сравнение признака с
   порогом идёт в float32, иначе маршрут по дереву расходится с обучающим кодом. */

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const f1 = v => v.toFixed(1).replace(".", ","), f2 = v => v.toFixed(2).replace(".", ",");
const f3 = v => v.toFixed(3).replace(".", ","), pc = v => (100 * v).toFixed(1).replace(".", ",") + "%";
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ---------- палитры ---------- */
const PAL = {
  risk:  ["#1d3b2c", "#4d4526", "#8f3a24", "#c05a34", "#e5924a", "#f7cf82"],
  gi:    ["#2f6d92", "#67a3c2", "#a9cbdb", "#2a3a33", "#e0a06e", "#cf6644", "#a02a18"],
  green: ["#3a2418", "#7a5a24", "#7f9a3e", "#3f9d70", "#a6e6c3"],
  heat:  ["#1b2620", "#3f4a26", "#8a6a25", "#c4732f", "#e8994a", "#f7cf82"],
  syn:   ["#d9a94a", "#57c088", "#d95f4e", "#6aa6d9", "#b47ad0", "#d0b05a"],
};
const rgb = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16));
function ramp(st, t) {
  t = clamp(t, 0, 1); const n = st.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = rgb(st[i]), b = rgb(st[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}

/* ---------- состояние ---------- */
let H, META, MODEL, LAY, FEATS, FI, base;
let pred, prob, gi, giSig, riskBreaks = [], synLab;
let map, hexLayer, hexes = [], boundLayer, prioLayer, repLayer, measureLayer;
let theme = "risk", catIdx = 0, opacity = .85, tool = "ident";
let selected = -1, filtered = null, sortKey = "risk", sortDir = -1;
let myReports = [], measurePts = [], ecoPoints = 0;
const STORE = "intelligreen.portal.v1";
const F = n => FI[FEATS.indexOf(n)];

/* ---------- инференс ---------- */
const fr = Math.fround;
function runTrees(m, row) {
  let s = m.base_score;
  for (const nodes of m.trees) { let n = nodes[0];
    while (n[0] !== -1) n = nodes[fr(row[n[0]]) < fr(n[1]) ? n[2] : n[3]];
    s += n[1]; }
  return s;
}
function rowOf(i) { const r = new Array(FEATS.length); for (let k = 0; k < FEATS.length; k++) r[k] = FI[k][i]; return r; }
function predictOne(i) { const r = rowOf(i);
  return { risk: clamp(runTrees(MODEL.regressor, r), 0, 10), prob: 1 / (1 + Math.exp(-runTrees(MODEL.classifier, r))) }; }
function predictAll() {
  const n = H.lat.length; pred = new Float64Array(n); prob = new Float64Array(n);
  for (let i = 0; i < n; i++) { const p = predictOne(i); pred[i] = p.risk; prob[i] = p.prob; }
  const s = Array.from(pred).sort((a, b) => a - b);
  riskBreaks = [.02, .2, .4, .6, .8, .98].map(q => s[Math.floor(q * (n - 1))]);
}
function stretch(v, br) {
  if (v <= br[0]) return 0; const last = br.length - 1; if (v >= br[last]) return 1;
  for (let k = 0; k < last; k++) if (v <= br[k + 1]) return (k + (v - br[k]) / Math.max(1e-9, br[k + 1] - br[k])) / last;
  return 1;
}

/* ---------- пространственная статистика ---------- */
function getisOrd(x) {
  const n = x.length, nb = META.neighbors; let s = 0, sq = 0;
  for (let i = 0; i < n; i++) { s += x[i]; sq += x[i] * x[i]; }
  const mean = s / n, S = Math.sqrt(Math.max(1e-12, sq / n - mean * mean)), z = new Float64Array(n);
  for (let i = 0; i < n; i++) { let lo = x[i], w = 1;
    for (const j of nb[i]) { lo += x[j]; w++; }
    z[i] = (lo - mean * w) / (S * Math.sqrt(Math.max(1e-12, (n * w - w * w) / (n - 1)))); }
  return z;
}
function normCdf(v) { const t = 1 / (1 + .2316419 * v), d = .3989422804014327 * Math.exp(-v * v / 2);
  return 1 - d * t * (.31938153 + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))); }
/* Поправка Бенджамини–Хохберга: при 2365 проверках без неё около сотни кварталов
   объявляются горячими просто из-за множественности сравнений. */
function fdr(z, q) {
  const n = z.length, p = new Float64Array(n);
  for (let i = 0; i < n; i++) p[i] = 2 * (1 - normCdf(Math.abs(z[i])));
  const ord = [...Array(n).keys()].sort((a, b) => p[a] - p[b]), sig = new Uint8Array(n);
  let cut = -1;
  for (let k = 0; k < n; k++) if (p[ord[k]] <= q * (k + 1) / n) cut = k;
  for (let k = 0; k <= cut; k++) sig[ord[k]] = 1;
  return sig;
}
function moransI(x) {
  const n = x.length, nb = META.neighbors; let m = 0;
  for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let num = 0, den = 0, W = 0;
  for (let i = 0; i < n; i++) { const zi = x[i] - m; den += zi * zi;
    for (const j of nb[i]) { num += zi * (x[j] - m); W++; } }
  return (n / W) * (num / den);
}
const recomputeSpatial = () => { gi = getisOrd(pred); giSig = fdr(gi, .05); };

/* ---------- геометрия гексагона ---------- */
function hexRing(lat, lng) {
  const R = META.grid_km, dLat = R / 111.0, dLng = R / (111.0 * Math.cos(lat * Math.PI / 180));
  const p = [];
  for (let k = 0; k < 6; k++) { const a = Math.PI / 3 * k;
    p.push([lat + dLat * Math.cos(a), lng + dLng * Math.sin(a)]); }
  return p;
}

/* ---------- раскраска ---------- */
function value(i) {
  switch (theme) {
    case "risk": return pred[i];
    case "gi":   return gi[i];
    case "syn":  return synLab[i];
    case "ndvi": return F("ndvi")[i];
    case "lst":  return F("lst_c")[i];
    case "pop":  return F("pop_density")[i];
    case "built":return F("built_density")[i];
    case "elev": return F("elevation_m")[i];
    case "rep":  return F("rep_count")[i];
    case "cat":  { const c = LAY.counts[i]; return c ? c[catIdx] : 0; }
  }
  return 0;
}
function colorOf(i) {
  const v = value(i);
  switch (theme) {
    case "risk": return ramp(PAL.risk, stretch(v, riskBreaks));
    case "gi":   return ramp(PAL.gi, clamp((v + 4) / 8, 0, 1));
    case "syn":  return v < 0 ? "#2a3a33" : PAL.syn[v % PAL.syn.length];
    case "ndvi": return ramp(PAL.green, clamp(v / .75, 0, 1));
    case "lst":  return ramp(PAL.risk, clamp((v - 15) / 22, 0, 1));
    case "pop":  return ramp(PAL.heat, clamp(Math.log1p(v) / Math.log1p(18000), 0, 1));
    case "built":return ramp(PAL.heat, clamp(v / .6, 0, 1));
    case "elev": return ramp(PAL.green, clamp((v - 600) / 1200, 0, 1));
    case "rep":  return ramp(PAL.heat, clamp(Math.log1p(v) / Math.log1p(45), 0, 1));
    case "cat":  return ramp(PAL.heat, clamp(Math.log1p(v) / Math.log1p(Math.max(3, LAY.totals[catIdx] / 260)), 0, 1));
  }
  return "#888";
}
const passes = i => !filtered || filtered[i];

/* ---------- фильтр ---------- */
function applyFilter() {
  const mn = parseFloat($("#fMin").value), mx = parseFloat($("#fMax").value);
  const d = $("#fDist").value, l = $("#fLand").value;
  const hot = $("#fHot").checked, high = $("#fHigh").checked;
  const any = !isNaN(mn) || !isNaN(mx) || d !== "" || l !== "" || hot || high;
  if (!any) { filtered = null; }
  else {
    filtered = new Uint8Array(H.lat.length);
    for (let i = 0; i < H.lat.length; i++) {
      if (!isNaN(mn) && pred[i] < mn) continue;
      if (!isNaN(mx) && pred[i] > mx) continue;
      if (d !== "" && H.district[i] != d) continue;
      if (l !== "" && H.landuse[i] != l) continue;
      if (hot && !(giSig[i] && gi[i] > 0)) continue;
      if (high && prob[i] < META.metrics.thr_cost) continue;
      filtered[i] = 1;
    }
  }
  const n = filtered ? filtered.reduce((a, b) => a + b, 0) : H.lat.length;
  $("#fCount").textContent = `Под условие подходит ${n} из ${H.lat.length} кварталов.`;
  $("#stCount").textContent = `выбрано ${n}`;
  restyle(); renderTable();
}

/* ---------- отрисовка карты ---------- */
function styleOf(i) {
  const on = passes(i);
  return { color: "#0d1310", weight: .35, opacity: on ? .5 : .12,
           fillColor: colorOf(i), fillOpacity: on ? opacity : opacity * .12 };
}
function restyle() { for (let i = 0; i < hexes.length; i++) hexes[i].setStyle(styleOf(i)); drawLegend(); }

function buildMap() {
  const b = META.bbox;
  map = L.map("map", { preferCanvas: true, zoomControl: false, attributionControl: true,
                       renderer: L.canvas({ padding: .4 }) });
  map.fitBounds([[b.lat_min, b.lng_min], [b.lat_max, b.lng_max]]);
  map.setMaxZoom(19);
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

  // CARTO с 2024 года требует ключ, поэтому подложки берём у провайдеров без ключа
  const ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/",
        ATTR_ESRI = 'Esri, HERE, Garmin, &copy; OpenStreetMap contributors';
  const BASES = {
    "Тёмная": L.tileLayer(ESRI + "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16, maxNativeZoom: 16, attribution: ATTR_ESRI }),
    "Светлая": L.tileLayer(ESRI + "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16, maxNativeZoom: 16, attribution: ATTR_ESRI }),
    "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }),
    "Спутник": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: 'Esri, Maxar, Earthstar Geographics' }),
    "Рельеф": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
      { maxZoom: 17, attribution: '&copy; OpenTopoMap, &copy; OpenStreetMap contributors' }),
  };
  let current = BASES["Тёмная"].addTo(map);
  $("#baseList").innerHTML = Object.keys(BASES).map((k, i) =>
    `<label class="lyr ${i ? "" : "sel"}"><input type="radio" name="bm" value="${k}" ${i ? "" : "checked"}><span>${k}</span></label>`).join("");
  $("#baseList").onchange = e => {
    map.removeLayer(current); current = BASES[e.target.value].addTo(map); current.bringToBack();
    $$("#baseList .lyr").forEach(l => l.classList.toggle("sel", l.querySelector("input").checked));
  };

  hexLayer = L.layerGroup().addTo(map);
  for (let i = 0; i < H.lat.length; i++) {
    const p = L.polygon(hexRing(H.lat[i], H.lng[i]), styleOf(i));
    p._i = i; p.addTo(hexLayer); hexes.push(p);
    p.on("mouseover", () => { $("#stHex").textContent =
        `${H.id[i]} · ${META.districts[H.district[i]]} · EcoRisk ${f2(pred[i])} · обращений ${Math.round(F("rep_count")[i])}`; });
    p.on("click", ev => { L.DomEvent.stop(ev); onMapClick(ev.latlng, i); });
  }
  boundLayer = L.layerGroup(); prioLayer = L.layerGroup();
  repLayer = L.layerGroup().addTo(map); measureLayer = L.layerGroup().addTo(map);
  buildBoundaries(); buildPriority(); drawReports();

  map.on("mousemove", e => $("#stCoord").textContent =
    `${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`);
  map.on("click", e => onMapClick(e.latlng, nearest(e.latlng)));
  $("#zIn").onclick = () => map.zoomIn(); $("#zOut").onclick = () => map.zoomOut();
  $("#zHome").onclick = () => map.fitBounds([[b.lat_min, b.lng_min], [b.lat_max, b.lng_max]]);
  $("#zSel").onclick = () => { if (selected >= 0) map.setView([H.lat[selected], H.lng[selected]], 15); };
}
function nearest(ll) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < H.lat.length; i++) {
    const dx = (H.lng[i] - ll.lng) * .73, dy = H.lat[i] - ll.lat, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return Math.sqrt(bd) < .01 ? best : -1;
}
/* Границы районов получаем растворением сетки: рисуем только те рёбра гексагона,
   по другую сторону которых другой район или край города. */
function buildBoundaries() {
  const key = (a, b) => `${a.toFixed(5)}|${b.toFixed(5)}`;
  const own = new Map();
  for (let i = 0; i < H.lat.length; i++) {
    const r = hexRing(H.lat[i], H.lng[i]);
    for (let k = 0; k < 6; k++) {
      const a = r[k], b = r[(k + 1) % 6];
      const e = [key(a[0], a[1]), key(b[0], b[1])].sort().join("::");
      if (!own.has(e)) own.set(e, { d: H.district[i], seg: [a, b], n: 1 });
      else { const o = own.get(e); o.n++; if (o.d === H.district[i]) o.same = true; }
    }
  }
  const segs = [];
  for (const v of own.values()) if (v.n === 1 || !v.same) segs.push(v.seg);
  L.polyline(segs, { color: "#f2ece0", weight: 1.1, opacity: .55, interactive: false }).addTo(boundLayer);
  for (let d = 0; d < META.districts.length; d++) {
    let la = 0, ln = 0, n = 0;
    for (let i = 0; i < H.lat.length; i++) if (H.district[i] === d) { la += H.lat[i]; ln += H.lng[i]; n++; }
    L.marker([la / n, ln / n], { interactive: false, icon: L.divIcon({ className: "",
      html: `<div style="color:#f2ece0;font:600 11px var(--sans);text-shadow:0 0 6px #000,0 0 3px #000;white-space:nowrap">${META.districts[d]}</div>`,
      iconSize: [90, 14], iconAnchor: [45, 7] }) }).addTo(boundLayer);
  }
}
function priorityScore(i) {
  const nd = F("ndvi")[i], pop = F("pop_density")[i], rep = F("rep_count")[i];
  return .45 * (pred[i] / 10) + .20 * Math.max(0, 1 - nd / .6)
       + .20 * Math.min(1, pop / 6000) + .15 * Math.min(1, Math.log1p(rep) / Math.log1p(40));
}
function topList(n) { return [...Array(H.lat.length).keys()].sort((a, b) => priorityScore(b) - priorityScore(a)).slice(0, n); }
function actionFor(i) {
  const nd = F("ndvi")[i], lu = META.landuse[H.landuse[i]];
  if (nd < .25) return "Озеленение и посадка";
  if (F("rep_count")[i] > 12) return "Санитарная очистка";
  if (lu === "Пустырь") return "Благоустройство";
  if (F("lst_c")[i] > 30) return "Озеленение и посадка";
  return "Экологический надзор";
}
function buildPriority() {
  prioLayer.clearLayers();
  topList(100).forEach((i, k) => {
    L.marker([H.lat[i], H.lng[i]], { icon: L.divIcon({ className: "", html: `<div class="rank">${k + 1}</div>`,
      iconSize: [20, 20], iconAnchor: [10, 10] }) })
      .bindTooltip(`<b>№${k + 1} · ${H.id[i]}</b><br>${META.districts[H.district[i]]} · EcoRisk ${f2(pred[i])}<br>${actionFor(i)}`)
      .on("click", () => selectHex(i)).addTo(prioLayer);
  });
}
function drawReports() {
  repLayer.clearLayers();
  myReports.forEach(r => {
    L.marker([r.lat, r.lng], { icon: L.divIcon({ className: "",
      html: `<div class="rep" style="background:${PAL.syn[r.cat % PAL.syn.length]}"></div>`,
      iconSize: [13, 13], iconAnchor: [6, 6] }) })
      .bindTooltip(`<b>${LAY.categories[r.cat]}</b><br>серьёзность ${r.sev} · ${new Date(r.t).toLocaleString("ru-RU")}`)
      .addTo(repLayer);
  });
}

/* ---------- инструменты ---------- */
function onMapClick(latlng, i) {
  if (tool === "measure") return measureClick(latlng);
  if (i < 0) return;
  selected = i;
  if (tool === "report") { showReportForm(i, latlng); }
  else showObject(i);
  restyle(); hi(i);
}
let hiLayer = null;
function hi(i) {
  if (hiLayer) map.removeLayer(hiLayer);
  hiLayer = L.polygon(hexRing(H.lat[i], H.lng[i]),
    { color: "#f2ece0", weight: 2, fill: false, interactive: false }).addTo(map);
}
function selectHex(i) { selected = i; showObject(i); hi(i); map.setView([H.lat[i], H.lng[i]], Math.max(map.getZoom(), 14)); renderTable(); }
function measureClick(ll) {
  measurePts.push(ll); measureLayer.clearLayers();
  L.polyline(measurePts, { color: "#d9a94a", weight: 2, dashArray: "5 4" }).addTo(measureLayer);
  measurePts.forEach(p => L.circleMarker(p, { radius: 3, color: "#d9a94a", fillOpacity: 1 }).addTo(measureLayer));
  let d = 0;
  for (let k = 1; k < measurePts.length; k++) d += measurePts[k - 1].distanceTo(measurePts[k]);
  $("#stMeasure").textContent = measurePts.length > 1
    ? `длина ${d < 1000 ? Math.round(d) + " м" : f2(d / 1000) + " км"} · двойной клик — сброс` : "кликайте по карте";
}

/* ---------- карточка объекта ---------- */
function showObject(i) {
  switchTab("obj");
  const hiRisk = prob[i] >= META.metrics.thr_cost, d = pred[i] - H.truth[i];
  const cat = LAY.counts[i] || [];
  const top = cat.map((v, k) => [k, v]).filter(x => x[1]).sort((a, b) => b[1] - a[1]).slice(0, 4);
  $("#paneObj").innerHTML = `
    <div class="big">${f2(pred[i])}<span style="font-size:12px;color:var(--ink-muted)"> EcoRisk 0–10</span></div>
    <div style="margin:.5rem 0 .7rem;display:flex;gap:.35rem;flex-wrap:wrap">
      <span class="chip ${hiRisk ? "hi" : "lo"}"><span class="dot"></span>${hiRisk ? "высокий риск" : "риск в норме"}</span>
      <span class="chip">p = ${f2(prob[i])}</span>
      ${giSig[i] ? `<span class="chip ${gi[i] > 0 ? "hi" : "lo"}">${gi[i] > 0 ? "горячая" : "холодная"} зона</span>` : ""}
    </div>
    <div class="kv"><span class="k">Квартал</span><span class="v">${H.id[i]}</span></div>
    <div class="kv"><span class="k">Район</span><span class="v">${META.districts[H.district[i]]}</span></div>
    <div class="kv"><span class="k">Землепользование</span><span class="v">${META.landuse[H.landuse[i]]}</span></div>
    <div class="kv"><span class="k">Координаты</span><span class="v">${H.lat[i].toFixed(5)}, ${H.lng[i].toFixed(5)}</span></div>
    <h4>Слой B · дистанционное зондирование</h4>
    <div class="kv"><span class="k">NDVI</span><span class="v">${f3(F("ndvi")[i])}</span></div>
    <div class="kv"><span class="k">Тренд NDVI за 3 года</span><span class="v">${f3(F("ndvi_trend_3y")[i])}</span></div>
    <div class="kv"><span class="k">Температура поверхности</span><span class="v">${f1(F("lst_c")[i])} °C</span></div>
    <h4>Слои C–F · город и рельеф</h4>
    <div class="kv"><span class="k">Плотность застройки</span><span class="v">${f2(F("built_density")[i])}</span></div>
    <div class="kv"><span class="k">Плотность дорог</span><span class="v">${f2(F("road_density")[i])}</span></div>
    <div class="kv"><span class="k">Население на км²</span><span class="v">${Math.round(F("pop_density")[i])}</span></div>
    <div class="kv"><span class="k">Высота</span><span class="v">${Math.round(F("elevation_m")[i])} м</span></div>
    <h4>Слой A · гражданские отчёты</h4>
    <div class="kv"><span class="k">Обращений</span><span class="v">${Math.round(F("rep_count")[i])}</span></div>
    <div class="kv"><span class="k">Средняя серьёзность</span><span class="v">${f2(F("rep_sev_mean")[i])}</span></div>
    <div class="kv"><span class="k">Уникальных участников</span><span class="v">${Math.round(F("rep_uniq_users")[i])}</span></div>
    <div class="kv"><span class="k">Доля верифицированных</span><span class="v">${pc(F("rep_verified_share")[i])}</span></div>
    ${top.length ? "<h4>Структура обращений</h4>" + top.map(([k, v]) =>
      `<div class="kv"><span class="k">${LAY.categories[k]}</span><span class="v">${v}</span></div>`).join("") : ""}
    <h4>Проверка модели</h4>
    <div class="kv"><span class="k">Getis-Ord Gi* (z)</span><span class="v">${f2(gi[i])}${giSig[i] ? " · значим" : ""}</span></div>
    <div class="kv"><span class="k">Эталонный EcoRisk</span><span class="v">${f2(H.truth[i])}</span></div>
    <div class="kv"><span class="k">Ошибка прогноза</span><span class="v" style="color:${Math.abs(d) <= 1 ? "var(--emerald-bright)" : "var(--oxblood)"}">${d >= 0 ? "+" : "−"}${f2(Math.abs(d))}</span></div>
    <h4>Рекомендация</h4>
    <div class="kv"><span class="k">Тип работы</span><span class="v">${actionFor(i)}</span></div>
    <button class="btn" id="objReport">Сообщить о проблеме здесь</button>`;
  $("#objReport").onclick = () => showReportForm(i, { lat: H.lat[i], lng: H.lng[i] });
}

/* ---------- гражданский отчёт ---------- */
function showReportForm(i, ll) {
  switchTab("obj");
  $("#paneObj").innerHTML = `
    <h4>Гражданский отчёт</h4>
    <div class="kv"><span class="k">Квартал</span><span class="v">${H.id[i]} · ${META.districts[H.district[i]]}</span></div>
    <div class="kv"><span class="k">Точка</span><span class="v">${ll.lat.toFixed(5)}, ${ll.lng.toFixed(5)}</span></div>
    <span class="lab">Категория проблемы</span>
    <select id="rCat">${LAY.categories.map((c, k) => `<option value="${k}">${c}</option>`).join("")}</select>
    <span class="lab">Серьёзность: <b id="rSevL">3 — заметно</b></span>
    <input type="range" class="op" id="rSev" min="1" max="5" value="3">
    <span class="lab">Комментарий (необязательно)</span>
    <input class="fin" id="rNote" placeholder="что именно не так">
    <button class="btn" id="rSend">Отправить отчёт</button>
    <button class="btn ghost" id="rCancel">Отмена</button>
    <p class="note">Отчёт меняет VGI-признаки квартала — число обращений, среднюю и максимальную
      серьёзность, накопленное доверие, долю поведенческих категорий — и модель немедленно
      пересчитывает EcoRisk. В прототипе отчёты хранятся в вашем браузере: статический хостинг
      не может быть общим хранилищем. В рабочей системе это PostGIS, FastAPI и очередь верификации.</p>`;
  const SEV = ["1 — незначительно", "2 — слабо", "3 — заметно", "4 — серьёзно", "5 — критично"];
  $("#rSev").oninput = e => $("#rSevL").textContent = SEV[e.target.value - 1];
  $("#rCancel").onclick = () => showObject(i);
  $("#rSend").onclick = () => {
    const cat = +$("#rCat").value, sev = +$("#rSev").value, before = pred[i];
    applyReport(i, cat, sev);
    myReports.push({ i, cat, sev, lat: ll.lat, lng: ll.lng, note: $("#rNote").value.slice(0, 200), t: Date.now() });
    ecoPoints += 10 + sev * 2; save();
    const p = predictOne(i); pred[i] = p.risk; prob[i] = p.prob;
    recomputeSpatial(); buildPriority(); drawReports(); restyle(); renderTable(); renderAnalytics(); renderMe();
    showObject(i);
    $("#paneObj").insertAdjacentHTML("afterbegin",
      `<div class="chip ${pred[i] > before ? "hi" : "lo"}" style="margin-bottom:.6rem">Отчёт учтён:
        EcoRisk ${f2(before)} → ${f2(pred[i])}</div>`);
  };
}
function applyReport(i, cat, sev) {
  const cnt = F("rep_count"), mean = F("rep_sev_mean"), mx = F("rep_sev_max"),
        tr = F("rep_trust_sum"), ver = F("rep_verified_share"), uq = F("rep_uniq_users"), bh = F("rep_behav_share");
  const n0 = cnt[i], n1 = n0 + 1, beh = META.behavioural.includes(cat) ? 1 : 0;
  mean[i] = (mean[i] * n0 + sev) / n1; mx[i] = Math.max(mx[i], sev);
  bh[i] = (bh[i] * n0 + beh) / n1; ver[i] = ver[i] * n0 / n1;
  tr[i] += 0.5; uq[i] += 1; cnt[i] = n1;          // доверие нового участника по методике — 0,5
  const c = LAY.counts[i] || (LAY.counts[i] = new Array(LAY.categories.length).fill(0));
  c[cat]++;
}
function save() { try { localStorage.setItem(STORE, JSON.stringify({ r: myReports, p: ecoPoints })); } catch (e) {} }
function restore() {
  let s = {}; try { s = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
  myReports = Array.isArray(s.r) ? s.r.filter(x => x && Number.isInteger(x.i) && x.i >= 0 && x.i < H.lat.length) : [];
  ecoPoints = +s.p || 0;
  for (const r of myReports) applyReport(r.i, r.cat, r.sev);
}
function resetMine() {
  myReports = []; ecoPoints = 0; try { localStorage.removeItem(STORE); } catch (e) {}
  FI = FEATS.map(f => Float64Array.from(base[f]));
  LAY.counts = JSON.parse(JSON.stringify(LAY._counts0));
  predictAll(); recomputeSpatial(); buildPriority(); drawReports(); restyle();
  renderTable(); renderAnalytics(); renderMe();
}

/* ---------- панели: аналитика и вклад ---------- */
function renderAnalytics() {
  const m = META.metrics, thr = m.thr_cost;
  const agg = META.districts.map(() => ({ n: 0, s: 0, hi: 0 }));
  for (let i = 0; i < H.lat.length; i++) { const a = agg[H.district[i]];
    a.n++; a.s += pred[i]; if (prob[i] >= thr) a.hi++; }
  const rows = META.districts.map((d, k) => ({ d, n: agg[k].n, mean: agg[k].s / agg[k].n, hi: agg[k].hi }))
    .sort((a, b) => b.mean - a.mean);
  const mx = rows[0].mean;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < H.lat.length; i++) { const p = H.prob_oof[i] >= thr ? 1 : 0, y = H.high_true[i];
    if (p && y) tp++; else if (p) fp++; else if (y) fn++; else tn++; }
  const hot = giSig.reduce((a, b, i) => a + (b && gi[i] > 0 ? 1 : 0), 0);
  $("#paneAna").innerHTML = `
    <h4>Рейтинг районов</h4>
    ${rows.map(r => `<div style="margin-bottom:.45rem;cursor:pointer" data-d="${META.districts.indexOf(r.d)}" class="drow">
      <div class="kv" style="border:0;padding:.1rem 0"><span class="k">${r.d}</span>
        <span class="v">${f2(r.mean)} · высокого риска ${r.hi}</span></div>
      <div class="bar"><i style="width:${(100 * r.mean / mx).toFixed(1)}%"></i></div></div>`).join("")}
    <h4>Пространственная статистика</h4>
    <div class="kv"><span class="k">Глобальный индекс Морана</span><span class="v">${f3(moransI(pred))}</span></div>
    <div class="kv"><span class="k">Значимых горячих зон</span><span class="v">${hot}</span></div>
    <div class="kv"><span class="k">Значимых зон всего (Gi*)</span><span class="v">${giSig.reduce((a, b) => a + b, 0)}</span></div>
    <p class="note">Поправка Бенджамини–Хохберга, уровень 0,05. Индекс Морана около 0,9 означает
      сильную пространственную связность: риск идёт массивами, а не случайными точками.</p>
    <h4>Качество модели</h4>
    <div class="kv"><span class="k">R², пространственная CV</span><span class="v">${f3(m.r2_spatial)}</span></div>
    <div class="kv"><span class="k">MAE</span><span class="v">${f3(m.mae_spatial)}</span></div>
    <div class="kv"><span class="k">AUC</span><span class="v">${f3(m.auc_spatial)}</span></div>
    <div class="kv"><span class="k">Порог по цене ошибки 3:1</span><span class="v">${f2(thr)}</span></div>
    <div class="kv"><span class="k">Пропуск деградации (FNR)</span><span class="v" style="color:var(--oxblood)">${pc(fn / (fn + tp))}</span></div>
    <div class="kv"><span class="k">Ложная тревога (FPR)</span><span class="v">${pc(fp / (fp + tn))}</span></div>
    <p class="note">${m.cv}. ${MODEL.regressor.trees.length} деревьев регрессии и
      ${MODEL.classifier.trees.length} деревьев классификации выполняются в браузере.</p>
    <h4>Синдромы проблем</h4>
    ${LAY.syndromes.map((s, k) => `<div class="kv"><span class="k">
      <span class="sw" style="display:inline-block;background:${PAL.syn[s.id % PAL.syn.length]};margin-right:.35rem"></span>
      ${s.top[0].category} + ${s.top[1].category}</span><span class="v">${s.hexes}</span></div>`).join("")}
    <p class="note">Кластеризация кварталов по доле категорий в их обращениях: типичные сочетания
      проблем, которые лечатся одним типом работ.</p>`;
  $$("#paneAna .drow").forEach(el => el.onclick = () => {
    $("#fDist").value = el.dataset.d; applyFilter();
    const d = +el.dataset.d, pts = [];
    for (let i = 0; i < H.lat.length; i++) if (H.district[i] === d) pts.push([H.lat[i], H.lng[i]]);
    map.fitBounds(L.latLngBounds(pts).pad(.05));
  });
}
function renderMe() {
  const lv = ecoPoints >= 500 ? "Эко-хранитель квартала" : ecoPoints >= 200 ? "Регулярный активист"
    : ecoPoints >= 50 ? "Эпизодический отчётчик" : "Новичок";
  const byCat = {};
  myReports.forEach(r => byCat[r.cat] = (byCat[r.cat] || 0) + 1);
  $("#paneMe").innerHTML = `
    <div class="big">${ecoPoints}<span style="font-size:12px;color:var(--ink-muted)"> эко-баллов</span></div>
    <div class="chip lo" style="margin-top:.5rem"><span class="dot"></span>${lv}</div>
    <div class="kv" style="margin-top:.7rem"><span class="k">Мои отчёты</span><span class="v">${myReports.length}</span></div>
    <div class="kv"><span class="k">Кварталов затронуто</span><span class="v">${new Set(myReports.map(r => r.i)).size}</span></div>
    <div class="kv"><span class="k">Доверие участника</span><span class="v">0,5 → растёт после верификации</span></div>
    ${Object.keys(byCat).length ? "<h4>По категориям</h4>" + Object.entries(byCat).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<div class="kv"><span class="k">${LAY.categories[k]}</span><span class="v">${v}</span></div>`).join("") : ""}
    ${myReports.length ? `<h4>Последние</h4>` + myReports.slice(-6).reverse().map(r =>
      `<div class="kv"><span class="k">${LAY.categories[r.cat]}</span>
        <span class="v">${H.id[r.i]} · ${r.sev}</span></div>`).join("") : '<p class="empty">Отчётов пока нет. Включите инструмент «Добавить отчёт» и кликните по карте — модель пересчитает риск квартала сразу.</p>'}
    <button class="btn ghost" id="meExport">Выгрузить мои отчёты (GeoJSON)</button>
    <button class="btn ghost" id="meReset">Очистить мои отчёты</button>
    <p class="note">Геймификация из методики: эко-баллы за подтверждённый вклад, уровни участия,
      рейтинг районов. Баллы начисляются за отчёт и его серьёзность.</p>`;
  $("#meReset").onclick = resetMine;
  $("#meExport").onclick = () => download("intelligreen-my-reports.geojson", JSON.stringify({
    type: "FeatureCollection", features: myReports.map(r => ({ type: "Feature",
      geometry: { type: "Point", coordinates: [r.lng, r.lat] },
      properties: { hex_id: H.id[r.i], category: LAY.categories[r.cat], severity: r.sev,
                    note: r.note || "", created: new Date(r.t).toISOString() } })) }, null, 1));
}
function switchTab(t) {
  $$(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $("#paneObj").hidden = t !== "obj"; $("#paneAna").hidden = t !== "ana"; $("#paneMe").hidden = t !== "me";
  if (t === "ana") renderAnalytics(); if (t === "me") renderMe();
}

/* ---------- таблица ---------- */
const COLS = [
  ["id", "Квартал", i => H.id[i], 0], ["dist", "Район", i => META.districts[H.district[i]], 0],
  ["risk", "EcoRisk", i => f2(pred[i]), 1], ["prob", "p(высокий)", i => f2(prob[i]), 1],
  ["gi", "Gi* z", i => f2(gi[i]), 1], ["ndvi", "NDVI", i => f3(F("ndvi")[i]), 1],
  ["lst", "LST °C", i => f1(F("lst_c")[i]), 1], ["pop", "Население", i => Math.round(F("pop_density")[i]), 1],
  ["rep", "Обращений", i => Math.round(F("rep_count")[i]), 1],
  ["land", "Землепользование", i => META.landuse[H.landuse[i]], 0],
];
const SORTV = { id: i => i, dist: i => H.district[i], risk: i => pred[i], prob: i => prob[i], gi: i => gi[i],
  ndvi: i => F("ndvi")[i], lst: i => F("lst_c")[i], pop: i => F("pop_density")[i],
  rep: i => F("rep_count")[i], land: i => H.landuse[i] };
function tableRows() {
  const idx = [];
  for (let i = 0; i < H.lat.length; i++) if (passes(i)) idx.push(i);
  const g = SORTV[sortKey];
  idx.sort((a, b) => (g(a) > g(b) ? 1 : g(a) < g(b) ? -1 : 0) * sortDir);
  return idx;
}
function renderTable() {
  const idx = tableRows(), show = idx.slice(0, 400);
  $("#tCount").textContent = `${idx.length} объектов${idx.length > 400 ? ", показаны первые 400" : ""}`;
  $("#tbl").innerHTML =
    `<thead><tr>${COLS.map(c => `<th data-k="${c[0]}" class="${c[3] ? "num" : ""}">${c[1]}${sortKey === c[0] ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead>` +
    `<tbody>${show.map(i => `<tr data-i="${i}" class="${i === selected ? "sel" : ""}">${
      COLS.map(c => `<td class="${c[3] ? "num" : ""}">${c[2](i)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}
function download(name, text, mime) {
  const b = new Blob([text], { type: mime || "application/json;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
function exportCsv() {
  const idx = tableRows();
  const head = ["hex_id", "lat", "lng", "district", "landuse", "ecorisk", "prob_high", "gi_z", "gi_sig",
                "ndvi", "lst_c", "pop_density", "built_density", "reports", "truth"];
  const lines = [head.join(";")].concat(idx.map(i => [H.id[i], H.lat[i], H.lng[i],
    META.districts[H.district[i]], META.landuse[H.landuse[i]], pred[i].toFixed(3), prob[i].toFixed(4),
    gi[i].toFixed(3), giSig[i], F("ndvi")[i], F("lst_c")[i], Math.round(F("pop_density")[i]),
    F("built_density")[i], Math.round(F("rep_count")[i]), H.truth[i]].join(";")));
  download("intelligreen-selection.csv", "﻿" + lines.join("\n"), "text/csv;charset=utf-8");
}
function exportGeo() {
  const idx = tableRows();
  download("intelligreen-selection.geojson", JSON.stringify({
    type: "FeatureCollection", crs: "EPSG:4326", license: "CC-BY-4.0",
    features: idx.map(i => ({ type: "Feature",
      geometry: { type: "Polygon", coordinates: [[...hexRing(H.lat[i], H.lng[i]), hexRing(H.lat[i], H.lng[i])[0]].map(p => [+p[1].toFixed(6), +p[0].toFixed(6)])] },
      properties: { hex_id: H.id[i], district: META.districts[H.district[i]],
        landuse: META.landuse[H.landuse[i]], ecorisk: +pred[i].toFixed(3), prob_high: +prob[i].toFixed(4),
        gi_z: +gi[i].toFixed(3), gi_significant: !!giSig[i], ndvi: F("ndvi")[i], lst_c: F("lst_c")[i],
        pop_density: Math.round(F("pop_density")[i]), reports: Math.round(F("rep_count")[i]),
        action: actionFor(i) } })) }));
}

/* ---------- легенда ---------- */
function drawLegend() {
  const L_ = {
    risk: ["EcoRisk, прогноз модели", PAL.risk, f1(riskBreaks[0]), f1(riskBreaks[5])],
    gi:   ["Getis-Ord Gi*, z-оценка", PAL.gi, "−4 холодная", "+4 горячая"],
    ndvi: ["NDVI, вегетационный индекс", PAL.green, "0", "0,75"],
    lst:  ["Температура поверхности", PAL.risk, "15 °C", "37 °C"],
    pop:  ["Плотность населения, чел/км²", PAL.heat, "0", "18 000"],
    built:["Плотность застройки", PAL.heat, "0", "0,6"],
    elev: ["Высота над уровнем моря", PAL.green, "600 м", "1 800 м"],
    rep:  ["Обращений жителей", PAL.heat, "0", "45"],
    cat:  [LAY.categories[catIdx], PAL.heat, "0", "много"],
  }[theme];
  if (theme === "syn") {
    $("#legend").innerHTML = `<div class="t">Синдромы проблем</div>` +
      LAY.syndromes.map(s => `<div class="lgi"><span class="sw" style="background:${PAL.syn[s.id % PAL.syn.length]}"></span>
        ${s.top[0].category}</div>`).join("") +
      `<div class="lgi"><span class="sw" style="background:#2a3a33"></span>мало обращений</div>`;
    return;
  }
  $("#legend").innerHTML = `<div class="t">${L_[0]}</div>
    <div class="ramp">${L_[1].map(c => `<i style="background:${c}"></i>`).join("")}</div>
    <div class="rlab"><span>${L_[2]}</span><span>${L_[3]}</span></div>`;
}

/* ---------- запуск ---------- */
const THEMES = [
  ["risk", "EcoRisk — прогноз модели", "итог слияния всех слоёв"],
  ["gi", "Горячие зоны Getis-Ord Gi*", "локальная статистика с поправкой FDR"],
  ["syn", "Синдромы проблем", "кластеры по сочетанию категорий"],
  ["rep", "Обращения жителей", "слой A · плотность VGI"],
  ["cat", "Обращения по категории", "слой A · 12 тепловых карт"],
  ["ndvi", "NDVI", "слой B · Sentinel-2"],
  ["lst", "Температура поверхности", "слой B · Landsat"],
  ["built", "Плотность застройки", "слои C и D"],
  ["pop", "Плотность населения", "слой E · WorldPop"],
  ["elev", "Рельеф", "слой F · SRTM"],
];
async function init() {
  const [h, m, mo, la] = await Promise.all(
    ["data/hexes.json", "data/meta.json", "data/model.json", "data/layers.json"]
      .map(u => fetch(u).then(r => { if (!r.ok) throw new Error(u + " — HTTP " + r.status); return r.json(); })));
  H = h; META = m; MODEL = mo; LAY = la; FEATS = MODEL.features;
  base = H.feat; FI = FEATS.map(f => Float64Array.from(H.feat[f]));
  const cn = {}; for (const k in LAY.counts) cn[+k] = LAY.counts[k];
  LAY.counts = cn; LAY._counts0 = JSON.parse(JSON.stringify(cn));
  synLab = LAY.syndrome;

  restore(); predictAll(); recomputeSpatial(); buildMap();

  $("#themeList").innerHTML = THEMES.map(([k, t, s], i) =>
    `<label class="lyr ${i ? "" : "sel"}"><input type="radio" name="th" value="${k}" ${i ? "" : "checked"}>
      <span>${t}<small>${s}</small></span></label>`).join("");
  $("#themeList").onchange = e => {
    theme = e.target.value; $("#catWrap").hidden = theme !== "cat";
    $$("#themeList .lyr").forEach(l => l.classList.toggle("sel", l.querySelector("input").checked));
    restyle();
  };
  $("#catSel").innerHTML = LAY.categories.map((c, k) =>
    `<option value="${k}">${c} — ${LAY.totals[k]} обращений</option>`).join("");
  $("#catSel").onchange = e => { catIdx = +e.target.value; restyle(); };
  $("#overList").innerHTML = `
    <label class="lyr"><input type="checkbox" id="ovBound"><span>Границы и названия районов</span></label>
    <label class="lyr"><input type="checkbox" id="ovPrio"><span>ТОП-100 приоритетных адресов</span></label>
    <label class="lyr"><input type="checkbox" id="ovRep" checked><span>Мои гражданские отчёты</span></label>`;
  $("#ovBound").onchange = e => e.target.checked ? boundLayer.addTo(map) : map.removeLayer(boundLayer);
  $("#ovPrio").onchange = e => e.target.checked ? prioLayer.addTo(map) : map.removeLayer(prioLayer);
  $("#ovRep").onchange = e => e.target.checked ? repLayer.addTo(map) : map.removeLayer(repLayer);
  $("#opacity").oninput = e => { opacity = e.target.value / 100; $("#opv").textContent = e.target.value + "%"; restyle(); };

  $("#fDist").innerHTML = `<option value="">все районы</option>` +
    META.districts.map((d, k) => `<option value="${k}">${d}</option>`).join("");
  $("#fLand").innerHTML = `<option value="">любое</option>` +
    META.landuse.map((d, k) => `<option value="${k}">${d}</option>`).join("");
  ["#fMin", "#fMax", "#fDist", "#fLand", "#fHot", "#fHigh"].forEach(s => $(s).onchange = applyFilter);
  $("#fReset").onclick = () => { $("#fMin").value = ""; $("#fMax").value = "";
    $("#fDist").value = ""; $("#fLand").value = ""; $("#fHot").checked = false; $("#fHigh").checked = false; applyFilter(); };

  $$(".sec>h3").forEach(h3 => h3.onclick = () => h3.parentElement.classList.toggle("closed"));
  $$(".tabs button").forEach(b => b.onclick = () => switchTab(b.dataset.tab));
  $$("[data-tool]").forEach(b => b.onclick = () => {
    tool = b.dataset.tool; $$("[data-tool]").forEach(x => x.classList.toggle("on", x === b));
    measurePts = []; measureLayer.clearLayers(); $("#stMeasure").textContent = "";
    $("#hint").textContent = { ident: "Идентификация: клик по кварталу открывает его карточку со всеми слоями и проверкой прогноза.",
      report: "Отчёт: кликните по месту проблемы на карте — откроется форма из 12 категорий, модель пересчитает риск квартала.",
      measure: "Измерение: кликайте по карте, чтобы построить ломаную. Двойной клик сбрасывает." }[tool];
  });
  $("#hint").textContent = "Идентификация: клик по кварталу открывает его карточку со всеми слоями и проверкой прогноза.";
  map.on("dblclick", () => { if (tool === "measure") { measurePts = []; measureLayer.clearLayers(); $("#stMeasure").textContent = ""; } });

  $("#tToggle").onclick = () => { const p = $("#tablepane"); p.classList.toggle("min");
    $("#tToggle").textContent = p.classList.contains("min") ? "Развернуть" : "Свернуть";
    setTimeout(() => map.invalidateSize(), 220); };
  $("#tTable").onclick = () => { const p = $("#tablepane");
    if (innerWidth <= 1080) p.classList.toggle("open");
    else { p.classList.remove("min"); $("#tToggle").textContent = "Свернуть"; }
    setTimeout(() => map.invalidateSize(), 220); };
  $("#tCsv").onclick = exportCsv; $("#tGeo").onclick = exportGeo;
  $("#tExport").onclick = exportGeo;
  document.addEventListener("click", e => {
    const th = e.target.closest("#tbl th[data-k]");
    if (th) { const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : -1; sortKey = k; renderTable(); return; }
    const tr = e.target.closest("#tbl tr[data-i]");
    if (tr) selectHex(+tr.dataset.i);
  });
  $("#mLeft").onclick = () => $("#left").classList.toggle("open");
  $("#mRight").onclick = () => $("#right").classList.toggle("open");

  const q = $("#q"), sg = $("#sugg");
  q.oninput = () => {
    const v = q.value.trim().toLowerCase();
    if (v.length < 2) { sg.hidden = true; return; }
    const coord = v.match(/^(-?\d+[.,]\d+)[,\s]+(-?\d+[.,]\d+)$/);
    let out = [];
    if (coord) out.push({ t: "Перейти к координатам", m: v, go: () =>
      map.setView([parseFloat(coord[1].replace(",", ".")), parseFloat(coord[2].replace(",", "."))], 15) });
    META.districts.forEach((d, k) => { if (d.toLowerCase().includes(v))
      out.push({ t: d, m: "район", go: () => { $("#fDist").value = k; applyFilter();
        const pts = []; for (let i = 0; i < H.lat.length; i++) if (H.district[i] === k) pts.push([H.lat[i], H.lng[i]]);
        map.fitBounds(L.latLngBounds(pts).pad(.05)); } }); });
    for (let i = 0; i < H.lat.length && out.length < 12; i++)
      if (H.id[i].toLowerCase().includes(v))
        out.push({ t: H.id[i], m: `${META.districts[H.district[i]]} · EcoRisk ${f2(pred[i])}`, go: () => selectHex(i) });
    sg.innerHTML = out.map((o, k) => `<div data-k="${k}"><b>${o.t}</b> <span class="m">${o.m}</span></div>`).join("")
      || `<div class="m" style="padding:.5em .6em">ничего не найдено</div>`;
    sg.hidden = false; sg._out = out;
  };
  sg.onclick = e => { const d = e.target.closest("div[data-k]"); if (!d) return;
    sg._out[+d.dataset.k].go(); sg.hidden = true; q.value = ""; };
  document.addEventListener("click", e => { if (!e.target.closest(".search")) sg.hidden = true; });

  applyFilter(); renderTable(); drawLegend(); switchTab("obj");
  $("#paneObj").innerHTML = `<p class="empty">Кликните по кварталу на карте — откроется карточка
    со всеми слоями, прогнозом модели, локальной статистикой Gi* и рекомендацией по типу работ.</p>
    <p class="note">Прототип геопортала к заявке на грантовый проект акимата города Алматы, 2026.
    Модель выполняется в браузере: ${MODEL.regressor.trees.length} деревьев регрессии,
    ${META.metrics.n_features} признаков, ${H.lat.length} кварталов.</p>`;
  addEventListener("resize", () => map.invalidateSize());
}
init().catch(e => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="position:fixed;inset:0;display:grid;place-items:center;background:#0d1310;z-index:9999;padding:2rem;text-align:center">
      <div><h3 style="color:#d95f4e">Не удалось загрузить данные портала</h3>
      <p style="color:#b4ae9f;margin-top:.6rem">${e.message}</p></div></div>`);
});
