"use strict";
/* IntelliGreen Almaty — демонстрационный сервис.
   Модель XGBoost выгружена в JSON и выполняется в браузере; сравнение порогов
   идёт в float32, как в самом XGBoost, иначе прогноз расходится с обучающим кодом. */

const $ = s => document.querySelector(s);
const fmt1 = v => v.toFixed(1).replace(".", ",");
const fmt2 = v => v.toFixed(2).replace(".", ",");
const fmt3 = v => v.toFixed(3).replace(".", ",");
const pct = v => (100 * v).toFixed(1).replace(".", ",") + "%";
const RAMP = ["#22332a", "#4a3a22", "#8f3a24", "#c05a34", "#e5924a", "#f4c672"];
const GREEN = ["#3a2418", "#7a5a24", "#7f9a3e", "#3f9d70", "#9fe0bd"];
const HOT = ["#2f6d92", "#5b9dbe", "#9dc6d8", "#f0f0e6", "#e8a87c", "#cf6644", "#9c2d1e"];

let H, META, MODEL, FEATS, FI;          // данные, метаданные, модель, список признаков, индексы
let base = null;                        // исходные VGI-признаки (для сброса)
let pred, prob, gi, giSig;              // прогнозы и пространственная статистика
let riskBreaks = [];                    // квантили прогноза для растяжки палитры
let layer = "risk", selected = -1, hover = -1;
let thrProb = 0.25;
const STORE = "intelligreen.reports.v1";
let myReports = [];

/* ---------------- инференс модели ---------------- */
const fr = Math.fround;
function runTrees(m, row) {
  let s = m.base_score;
  for (const nodes of m.trees) {
    let k = 0, n = nodes[0];
    while (n[0] !== -1) { k = fr(row[n[0]]) < fr(n[1]) ? n[2] : n[3]; n = nodes[k]; }
    s += n[1];
  }
  return s;
}
function featRow(i) { const r = new Array(FEATS.length); for (let f = 0; f < FEATS.length; f++) r[f] = FI[f][i]; return r; }
function predictOne(i) {
  const row = featRow(i);
  return { risk: Math.max(0, Math.min(10, runTrees(MODEL.regressor, row))),
           prob: 1 / (1 + Math.exp(-runTrees(MODEL.classifier, row))) };
}
function predictAll() {
  const n = H.lat.length;
  pred = new Float64Array(n); prob = new Float64Array(n);
  for (let i = 0; i < n; i++) { const p = predictOne(i); pred[i] = p.risk; prob[i] = p.prob; }
  const srt = Array.from(pred).sort((a, b) => a - b);
  riskBreaks = [0.02, 0.2, 0.4, 0.6, 0.8, 0.98].map(q => srt[Math.floor(q * (n - 1))]);
}
function stretch(v, br) {                 // линейная интерполяция по квантильным границам
  if (v <= br[0]) return 0;
  const last = br.length - 1;
  if (v >= br[last]) return 1;
  for (let k = 0; k < last; k++)
    if (v <= br[k + 1]) return (k + (v - br[k]) / Math.max(1e-9, br[k + 1] - br[k])) / last;
  return 1;
}

/* ---------------- пространственная статистика ---------------- */
/* Getis-Ord Gi*: кольцо из шести соседей плюс сам квартал, веса единичные. */
function getisOrd(x) {
  const n = x.length, nb = META.neighbors;
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) { sum += x[i]; sq += x[i] * x[i]; }
  const mean = sum / n, S = Math.sqrt(Math.max(1e-12, sq / n - mean * mean));
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let lo = x[i], w = 1;
    for (const j of nb[i]) { lo += x[j]; w++; }
    const denom = S * Math.sqrt(Math.max(1e-12, (n * w - w * w) / (n - 1)));
    z[i] = (lo - mean * w) / denom;
  }
  return z;
}
/* Двусторонние p-значения и поправка Бенджамини–Хохберга: без неё при 2365 проверках
   около сотни кварталов объявляются «горячими» просто из-за множественности. */
function fdr(z, q) {
  const n = z.length, p = new Float64Array(n);
  for (let i = 0; i < n; i++) p[i] = 2 * (1 - normCdf(Math.abs(z[i])));
  const ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  const sig = new Uint8Array(n);
  let cut = -1;
  for (let k = 0; k < n; k++) if (p[ord[k]] <= q * (k + 1) / n) cut = k;
  for (let k = 0; k <= cut; k++) sig[ord[k]] = 1;
  return sig;
}
function normCdf(v) {                    // приближение Абрамовица–Стиган, 7.1.26
  const t = 1 / (1 + 0.2316419 * v), d = 0.3989422804014327 * Math.exp(-v * v / 2);
  return 1 - d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
}
function moransI(x) {
  const n = x.length, nb = META.neighbors;
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
  let num = 0, den = 0, W = 0;
  for (let i = 0; i < n; i++) {
    const zi = x[i] - mean; den += zi * zi;
    for (const j of nb[i]) { num += zi * (x[j] - mean); W++; }
  }
  return (n / W) * (num / den);
}
function recomputeSpatial() { gi = getisOrd(pred); giSig = fdr(gi, 0.05); }

/* ---------------- проекция и отрисовка ---------------- */
const cv = $("#cv"), ctx = cv.getContext("2d");
let PX = 1, ox = 0, oy = 0, hexR = 6, kmx = [], kmy = [];
function project() {
  const lat0 = (META.bbox.lat_min + META.bbox.lat_max) / 2;
  const k = Math.cos(lat0 * Math.PI / 180);
  kmx = H.lng.map(v => (v - META.bbox.lng_min) * 111.0 * k);
  kmy = H.lat.map(v => (META.bbox.lat_max - v) * 111.0);
}
function layoutCanvas() {
  const box = cv.parentElement.getBoundingClientRect();
  const W = Math.max(300, Math.round(box.width));
  const pad = 10;
  const spanX = Math.max(...kmx) + 2 * META.grid_km, spanY = Math.max(...kmy) + 2 * META.grid_km;
  const maxH = Math.max(340, Math.min(700, Math.round(innerHeight * 0.74)));
  const Hh = Math.min(maxH, Math.round(W * spanY / spanX));
  const s = Math.min((W - 2 * pad) / spanX, (Hh - 2 * pad) / spanY);
  PX = window.devicePixelRatio || 1;
  cv.width = Math.round(W * PX); cv.height = Math.round(Hh * PX); cv.style.height = Hh + "px";
  ox = (W - spanX * s) / 2 + META.grid_km * s;
  oy = (Hh - spanY * s) / 2 + META.grid_km * s;
  hexR = META.grid_km * s;
  cv._s = s; cv._W = W; cv._H = Hh;
}
const sx = i => ox + kmx[i] * cv._s, sy = i => oy + kmy[i] * cv._s;

function colorFor(i) {
  if (layer === "hot") { const t = Math.max(0, Math.min(1, (gi[i] + 4) / 8)); return ramp(HOT, t); }
  if (layer === "ndvi") return ramp(GREEN, Math.max(0, Math.min(1, FI[FEATS.indexOf("ndvi")][i] / 0.75)));
  if (layer === "lst") return ramp(RAMP, Math.max(0, Math.min(1, (FI[FEATS.indexOf("lst_c")][i] - 15) / 22)));
  if (layer === "rep") { const c = FI[FEATS.indexOf("rep_count")][i]; return ramp(RAMP, Math.max(0, Math.min(1, Math.log1p(c) / Math.log1p(40)))); }
  return ramp(RAMP, stretch(pred[i], riskBreaks));
}
function ramp(stops, t) {
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
function hex2rgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(k => parseInt(h.slice(k, k + 2), 16)); }

function hexPath(x, y, r) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 180) * (60 * k - 90);
    const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
    k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }
  ctx.closePath();
}
function draw() {
  ctx.setTransform(PX, 0, 0, PX, 0, 0);
  ctx.clearRect(0, 0, cv._W, cv._H);
  ctx.fillStyle = "#121a15"; ctx.fillRect(0, 0, cv._W, cv._H);
  const showSig = $("#hotspots").checked;
  for (let i = 0; i < H.lat.length; i++) {
    hexPath(sx(i), sy(i), hexR * 0.98);
    ctx.fillStyle = colorFor(i);
    ctx.globalAlpha = showSig && !giSig[i] ? 0.22 : 1;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  for (const [idx, col, w] of [[hover, "#f2ece0", 1.6], [selected, "#d9a94a", 2.2]]) {
    if (idx < 0) continue;
    hexPath(sx(idx), sy(idx), hexR * 1.05);
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.stroke();
  }
  for (const r of myReports) {
    const i = r.i;
    ctx.beginPath(); ctx.arc(sx(i), sy(i), Math.max(2, hexR * .32), 0, 6.2832);
    ctx.fillStyle = "#57c088"; ctx.strokeStyle = "#0d1310"; ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
  }
}
function pick(ev) {
  const r = cv.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (cv._W / r.width), y = (ev.clientY - r.top) * (cv._H / r.height);
  let best = -1, bd = hexR * hexR * 1.2;
  for (let i = 0; i < H.lat.length; i++) {
    const dx = sx(i) - x, dy = sy(i) - y, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

/* ---------------- панели ---------------- */
function featVal(name, i) { return FI[FEATS.indexOf(name)][i]; }
function hexCard(i) {
  const d = META.districts[H.district[i]], lu = META.landuse[H.landuse[i]];
  const hi = prob[i] >= thrProb;
  const dr = pred[i] - H.truth[i];
  $("#hexempty").hidden = true;
  const body = $("#hexbody"); body.hidden = false;
  body.innerHTML = `
    <div class="big">${fmt2(pred[i])} <span class="muted" style="font-size:.5em">EcoRisk 0–10</span></div>
    <div style="margin:.5rem 0 .7rem">
      <span class="badge ${hi ? "hi" : "lo"}"><span class="dot"></span>${hi ? "высокий риск" : "риск в норме"}</span>
      <span class="badge">p = ${fmt2(prob[i])}</span>
    </div>
    <div class="kv"><span class="k">Квартал</span><span class="val">${H.id[i]}</span></div>
    <div class="kv"><span class="k">Район</span><span class="val">${d}</span></div>
    <div class="kv"><span class="k">Землепользование</span><span class="val">${lu}</span></div>
    <div class="kv"><span class="k">Координаты</span><span class="val">${H.lat[i].toFixed(4)}, ${H.lng[i].toFixed(4)}</span></div>
    <div class="kv"><span class="k">NDVI</span><span class="val">${fmt3(featVal("ndvi", i))}</span></div>
    <div class="kv"><span class="k">Температура поверхности</span><span class="val">${fmt1(featVal("lst_c", i))} °C</span></div>
    <div class="kv"><span class="k">Плотность застройки</span><span class="val">${fmt2(featVal("built_density", i))}</span></div>
    <div class="kv"><span class="k">Население на км²</span><span class="val">${Math.round(featVal("pop_density", i))}</span></div>
    <div class="kv"><span class="k">Обращений жителей</span><span class="val">${Math.round(featVal("rep_count", i))}</span></div>
    <div class="kv"><span class="k">Getis-Ord Gi* (z)</span><span class="val">${fmt2(gi[i])}${giSig[i] ? " · значим" : ""}</span></div>
    <div class="kv"><span class="k">Эталон EcoRisk</span><span class="val">${fmt2(H.truth[i])}</span></div>
    <div class="delta ${Math.abs(dr) <= 1 ? "em" : "ox"}">Отклонение прогноза от эталона: ${dr >= 0 ? "+" : "−"}${fmt2(Math.abs(dr))}</div>`;
  $("#reportcard").hidden = false;
}

/* ---------------- гражданский отчёт ---------------- */
function applyReport(i, cat, sev) {
  const g = n => FI[FEATS.indexOf(n)];
  const cnt = g("rep_count"), mean = g("rep_sev_mean"), mx = g("rep_sev_max"),
        trust = g("rep_trust_sum"), ver = g("rep_verified_share"), uniq = g("rep_uniq_users"),
        beh = g("rep_behav_share");
  const n0 = cnt[i], n1 = n0 + 1;
  const isBeh = META.behavioural.includes(cat) ? 1 : 0;
  mean[i] = (mean[i] * n0 + sev) / n1;
  mx[i] = Math.max(mx[i], sev);
  beh[i] = (beh[i] * n0 + isBeh) / n1;
  ver[i] = (ver[i] * n0) / n1;            // новый отчёт ещё не верифицирован
  trust[i] = trust[i] + 0.5;              // доверие нового участника по методике — 0,5
  uniq[i] = uniq[i] + 1;
  cnt[i] = n1;
}
function submitReport() {
  if (selected < 0) return;
  const cat = +$("#cat").value, sev = +$("#sev").value;
  const before = pred[selected];
  applyReport(selected, cat, sev);
  myReports.push({ i: selected, cat, sev, t: Date.now() });
  save();
  const p = predictOne(selected); pred[selected] = p.risk; prob[selected] = p.prob;
  recomputeSpatial();
  hexCard(selected); draw(); renderPriority(); renderDistricts(); renderThreshold();
  const d = pred[selected] - before;
  const el = document.createElement("div");
  el.className = "delta " + (d > 0 ? "ox" : "em");
  el.textContent = `Отчёт учтён. EcoRisk квартала: ${fmt2(before)} → ${fmt2(pred[selected])} (${d >= 0 ? "+" : "−"}${fmt2(Math.abs(d))})`;
  $("#hexbody").appendChild(el);
}
function save() { try { localStorage.setItem(STORE, JSON.stringify(myReports)); } catch (e) {} }
function restore() {
  let r = [];
  try { r = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch (e) { r = []; }
  if (!Array.isArray(r)) r = [];
  myReports = r.filter(x => x && Number.isInteger(x.i) && x.i >= 0 && x.i < H.lat.length);
  for (const x of myReports) applyReport(x.i, x.cat, x.sev);
}
function resetReports() {
  myReports = []; try { localStorage.removeItem(STORE); } catch (e) {}
  FI = FEATS.map(f => Float64Array.from(base[f]));
  predictAll(); recomputeSpatial();
  if (selected >= 0) hexCard(selected);
  draw(); renderPriority(); renderDistricts(); renderThreshold();
}

/* ---------------- приоритизация и таблицы ---------------- */
/* Многокритериальная свёртка: риск, дефицит зелени, население, плотность обращений. */
function priorityScore(i) {
  const ndvi = featVal("ndvi", i), pop = featVal("pop_density", i), rep = featVal("rep_count", i);
  const deficit = Math.max(0, 1 - ndvi / 0.6);
  const popN = Math.min(1, pop / 6000), repN = Math.min(1, Math.log1p(rep) / Math.log1p(40));
  return 0.45 * (pred[i] / 10) + 0.20 * deficit + 0.20 * popN + 0.15 * repN;
}
function renderPriority() {
  const n = H.lat.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => priorityScore(b) - priorityScore(a)).slice(0, 100);
  const act = Object.values(META.actions);
  $("#prio").innerHTML =
    `<thead><tr><th class="num">#</th><th>Квартал</th><th>Район</th><th class="num">EcoRisk</th>
      <th class="num">NDVI</th><th class="num">Население</th><th class="num">Обращений</th><th>Рекомендуемое действие</th></tr></thead><tbody>` +
    idx.map((i, k) => {
      const ndvi = featVal("ndvi", i), lu = META.landuse[H.landuse[i]];
      const a = ndvi < 0.25 ? "Озеленение и посадка" : featVal("rep_count", i) > 12 ? "Санитарная очистка"
        : lu === "Пустырь" ? "Благоустройство" : featVal("lst_c", i) > 30 ? "Озеленение и посадка" : "Экологический надзор";
      return `<tr data-i="${i}"><td class="num">${k + 1}</td><td>${H.id[i]}</td><td>${META.districts[H.district[i]]}</td>
        <td class="num">${fmt2(pred[i])}</td><td class="num">${fmt3(ndvi)}</td>
        <td class="num">${Math.round(featVal("pop_density", i))}</td>
        <td class="num">${Math.round(featVal("rep_count", i))}</td><td>${a}</td></tr>`;
    }).join("") + "</tbody>";
}
function renderDistricts() {
  const agg = META.districts.map(() => ({ n: 0, risk: 0, hi: 0, ndvi: 0, rep: 0 }));
  for (let i = 0; i < H.lat.length; i++) {
    const a = agg[H.district[i]];
    a.n++; a.risk += pred[i]; a.ndvi += featVal("ndvi", i); a.rep += featVal("rep_count", i);
    if (prob[i] >= thrProb) a.hi++;
  }
  const rows = META.districts.map((d, k) => ({ d, ...agg[k], mean: agg[k].risk / agg[k].n }))
    .sort((a, b) => b.mean - a.mean);
  const max = rows[0].mean;
  $("#dist").innerHTML =
    `<thead><tr><th>Район</th><th class="num">Сводный EcoRisk</th><th></th><th class="num">Кварталов</th>
      <th class="num">Высокий риск</th><th class="num">Доля</th><th class="num">Средний NDVI</th><th class="num">Обращений</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.d}</td><td class="num">${fmt2(r.mean)}</td>
      <td style="width:22%"><div class="bar"><i style="width:${(100 * r.mean / max).toFixed(1)}%"></i></div></td>
      <td class="num">${r.n}</td><td class="num">${r.hi}</td><td class="num">${pct(r.hi / r.n)}</td>
      <td class="num">${fmt3(r.ndvi / r.n)}</td><td class="num">${Math.round(r.rep)}</td></tr>`).join("") + "</tbody>";
}

/* ---------------- модель, порог, матрица ошибок ---------------- */
function confusionAt(t) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < H.lat.length; i++) {
    const p = H.prob_oof[i] >= t ? 1 : 0, y = H.high_true[i];
    if (p && y) tp++; else if (p && !y) fp++; else if (!p && y) fn++; else tn++;
  }
  return { tp, fp, tn, fn, fnr: fn / (fn + tp), fpr: fp / (fp + tn),
           precision: tp / Math.max(1, tp + fp), recall: tp / Math.max(1, tp + fn),
           cost: 3 * fn + fp };
}
function renderThreshold() {
  const c = confusionAt(thrProb);
  $("#throut").innerHTML =
    `<div class="kv"><span class="k">Порог вероятности</span><span class="val">${fmt2(thrProb)}</span></div>
     <div class="kv"><span class="k">Пропуск деградации (FNR)</span><span class="val ox">${pct(c.fnr)}</span></div>
     <div class="kv"><span class="k">Ложная тревога (FPR)</span><span class="val">${pct(c.fpr)}</span></div>
     <div class="kv"><span class="k">Точность / полнота</span><span class="val">${fmt3(c.precision)} / ${fmt3(c.recall)}</span></div>
     <div class="kv"><span class="k">Цена ошибок 3·FN + FP</span><span class="val">${c.cost}</span></div>`;
  $("#cm").innerHTML =
    `<div><b class="em">${c.tp}</b>верно найдено</div><div><b>${c.fp}</b>ложная тревога</div>
     <div><b class="ox">${c.fn}</b>пропущено</div><div><b>${c.tn}</b>верно отклонено</div>`;
}
function renderModel() {
  const m = META.metrics;
  $("#mmetrics").innerHTML =
    `<div class="kv"><span class="k">R², пространственная CV</span><span class="val em">${fmt3(m.r2_spatial)}</span></div>
     <div class="kv"><span class="k">Средняя абсолютная ошибка</span><span class="val">${fmt3(m.mae_spatial)}</span></div>
     <div class="kv"><span class="k">AUC детекции высокого риска</span><span class="val em">${fmt3(m.auc_spatial)}</span></div>
     <div class="kv"><span class="k">Кварталов в обучении</span><span class="val">${m.n_hexes}</span></div>
     <div class="kv"><span class="k">Признаков</span><span class="val">${m.n_features}</span></div>
     <div class="kv"><span class="k">Глобальный Moran's I</span><span class="val">${fmt3(moransI(pred))}</span></div>
     <p class="note">${m.cv}. Модель выгружена из XGBoost в JSON и выполняется в браузере:
       ${MODEL.regressor.trees.length} деревьев регрессии и ${MODEL.classifier.trees.length} деревьев классификации.</p>`;
  const use = new Float64Array(FEATS.length);
  for (const t of MODEL.regressor.trees) for (const nd of t) if (nd[0] !== -1) use[nd[0]]++;
  let vgi = 0, tot = 0;
  FEATS.forEach((f, k) => { tot += use[k]; if (META.vgi_features.includes(f)) vgi += use[k]; });
  const top = FEATS.map((f, k) => [f, use[k]]).sort((a, b) => b[1] - a[1]).slice(0, 6);
  $("#contrib").innerHTML =
    `<div class="kv"><span class="k">Гражданский сигнал (VGI)</span><span class="val">${pct(vgi / tot)}</span></div>
     <div class="kv"><span class="k">Открытые геоданные</span><span class="val">${pct(1 - vgi / tot)}</span></div>` +
    top.map(([f, v]) => `<div class="kv"><span class="k">${f}</span><span class="val">${pct(v / tot)}</span></div>`).join("");
}

/* ---------------- легенда и слои ---------------- */
const LAYERS = [["risk", "EcoRisk"], ["hot", "Gi* (z-оценка)"], ["ndvi", "NDVI"], ["lst", "Температура"], ["rep", "Обращения"]];
function renderLegend() {
  const L = {
    risk: [fmt1(riskBreaks[0]) + " — спокойно", RAMP, fmt1(riskBreaks[riskBreaks.length - 1]) + " — критический"],
    hot: ["−4 холодная зона", HOT, "+4 горячая зона"],
    ndvi: ["0 — нет зелени", GREEN, "0,75 — плотная зелень"],
    lst: ["15 °C", RAMP, "37 °C"],
    rep: ["нет обращений", RAMP, "40 и более"],
  }[layer];
  $("#legend").innerHTML =
    `<span>${L[0]}</span><span class="ramp">${L[1].map(c => `<i style="background:${c}"></i>`).join("")}</span><span>${L[2]}</span>
     <span class="muted">· 2 365 кварталов по 350 м · клик по кварталу открывает карточку</span>`;
}

/* ---------------- запуск ---------------- */
async function init() {
  const [h, m, mo] = await Promise.all([
    fetch("data/hexes.json").then(r => r.json()),
    fetch("data/meta.json").then(r => r.json()),
    fetch("data/model.json").then(r => r.json()),
  ]);
  H = h; META = m; MODEL = mo; FEATS = MODEL.features;
  base = H.feat;
  FI = FEATS.map(f => Float64Array.from(H.feat[f]));
  thrProb = META.metrics.thr_cost;

  restore();
  predictAll(); recomputeSpatial(); project(); layoutCanvas();

  $("#layers").innerHTML = LAYERS.map(([k, t], i) =>
    `<button data-l="${k}" class="${i ? "" : "on"}">${t}</button>`).join("");
  $("#cat").innerHTML = META.categories.map((c, i) => `<option value="${i}">${c}</option>`).join("");
  const SEV = ["1 — незначительно", "2 — слабо", "3 — заметно", "4 — серьёзно", "5 — критично"];
  $("#sev").oninput = e => $("#sevval").textContent = SEV[e.target.value - 1];
  const t = $("#thr"); t.value = Math.round(thrProb * 100);
  t.oninput = e => { thrProb = e.target.value / 100; renderThreshold(); renderDistricts(); draw(); if (selected >= 0) hexCard(selected); };

  const hi = H.high_true.reduce((a, b) => a + b, 0);
  $("#kpis").innerHTML = [
    [H.lat.length.toLocaleString("ru-RU"), "Кварталов под наблюдением, сетка 350 м"],
    [fmt3(META.metrics.r2_spatial), "R² прогноза EcoRisk, пространственная CV"],
    [pct(META.metrics.fnr), "Пропуск деградированных кварталов при пороге по цене ошибки"],
    [hi, "Кварталов высокого риска по эталону"],
    [META.metrics.n_features, "Признаков: гражданские отчёты и открытые геоданные"],
  ].map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  $("#mapnote").textContent =
    "Цвет квартала — прогноз модели, а не эталон. Включите горячие зоны, чтобы оставить только кварталы, " +
    "у которых локальная статистика Getis-Ord Gi* значима после поправки на множественность проверок.";

  renderLegend(); renderModel(); renderThreshold(); renderPriority(); renderDistricts(); draw();

  $("#layers").onclick = e => {
    const b = e.target.closest("button[data-l]"); if (!b) return;
    $("#layers").querySelectorAll("button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); layer = b.dataset.l; renderLegend(); draw();
  };
  $("#hotspots").onchange = draw;
  $("#reset").onclick = resetReports;
  $("#send").onclick = submitReport;
  cv.onmousemove = ev => {
    const i = pick(ev);
    if (i !== hover) { hover = i; draw(); }
    const tip = $("#tip");
    if (i < 0) { tip.style.opacity = 0; return; }
    tip.innerHTML = `<b>${H.id[i]}</b> · ${META.districts[H.district[i]]}<br>
      EcoRisk ${fmt2(pred[i])} · NDVI ${fmt3(featVal("ndvi", i))}<br>
      обращений ${Math.round(featVal("rep_count", i))} · Gi* z ${fmt2(gi[i])}`;
    tip.style.opacity = 1;
    const r = cv.parentElement.getBoundingClientRect();
    let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
    if (x + 250 > r.width) x = ev.clientX - r.left - 250;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  cv.onmouseleave = () => { hover = -1; $("#tip").style.opacity = 0; draw(); };
  cv.onclick = ev => { const i = pick(ev); if (i < 0) return; selected = i; hexCard(i); draw(); };
  document.addEventListener("click", e => {
    const tr = e.target.closest("#prio tr[data-i]"); if (!tr) return;
    selected = +tr.dataset.i; hexCard(selected); draw();
    document.getElementById("map").scrollIntoView({ block: "start" });
  });
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { layoutCanvas(); draw(); }, 120); });
}
init().catch(e => {
  document.querySelector(".hero").insertAdjacentHTML("beforeend",
    `<p class="note ox">Не удалось загрузить данные сервиса: ${e.message}</p>`);
});
