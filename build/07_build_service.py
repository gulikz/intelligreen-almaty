# -*- coding: utf-8 -*-
"""Сборка демонстрационного веб-сервиса IntelliGreen Almaty.

Готовит компактные данные и переносимую в браузер модель:
  service/data/hexes.json   — колоночная выгрузка 2365 гексов (координаты, признаки, эталон)
  service/data/model.json   — деревья XGBoost в компактном виде, инференс идёт в браузере
  service/data/meta.json    — районы, землепользование, геометрия сетки, соседи, метрики
  service/api/v1/*.json     — статический read-only API (в т.ч. GeoJSON)

Метрики компактной модели считаются по пространственной кросс-валидации —
теми же блоками, что и в 02_run_cases.py, чтобы оценка оставалась честной.
"""
import json, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE                       # hexes.csv лежит рядом со скриптом
SVC  = os.path.join(HERE, "..")   # data/ и api/ пишутся в корень репозитория
for d in ("data", "api/v1"):
    os.makedirs(os.path.join(SVC, d), exist_ok=True)

hx = pd.read_csv(os.path.join(DATA, "hexes.csv"))
N = len(hx)

VGI = ["rep_count", "rep_sev_mean", "rep_sev_max", "rep_trust_sum", "rep_verified_share",
       "rep_uniq_users", "rep_behav_share", "rep_late_share"]
GIS = ["ndvi", "ndvi_trend_3y", "lst_c", "pop_density", "built_density", "road_density",
       "dist_road_m", "impervious", "elevation_m"]
LANDUSE = sorted(hx["landuse"].unique())
DISTRICTS = sorted(hx["district"].unique())
lu = pd.get_dummies(hx["landuse"], prefix="lu").astype(float)[[f"lu_{l}" for l in LANDUSE]]
X = pd.concat([hx[VGI], hx[GIS], lu], axis=1)
FEATS = list(X.columns)
y_reg = hx["eco_risk_true"].values
y_clf = hx["high_risk_true"].values
THRESH = 6.5

# ---- те же пространственные блоки, что и в основном расчёте ------------------
coords = StandardScaler().fit_transform(hx[["lat", "lng"]].values)
blocks = KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(coords)
gkf = GroupKFold(n_splits=6)

def mk_reg():
    return XGBRegressor(n_estimators=140, max_depth=4, learning_rate=0.10, subsample=0.85,
                        colsample_bytree=0.85, reg_lambda=1.2, min_child_weight=3,
                        random_state=0, n_jobs=4)
def mk_clf():
    w = (y_clf == 0).sum() / max(1, (y_clf == 1).sum())
    return XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.10, subsample=0.85,
                         colsample_bytree=0.85, reg_lambda=1.2, min_child_weight=3,
                         scale_pos_weight=w, eval_metric="logloss", random_state=0, n_jobs=4)

oof_r = np.zeros(N); oof_p = np.zeros(N)
for tr, te in gkf.split(X, y_reg, groups=blocks):
    oof_r[te] = mk_reg().fit(X.iloc[tr], y_reg[tr]).predict(X.iloc[te])
    oof_p[te] = mk_clf().fit(X.iloc[tr], y_clf[tr]).predict_proba(X.iloc[te])[:, 1]

C_FN, C_FP = 3.0, 1.0
grid = np.round(np.arange(0.05, 0.95, 0.005), 3)
cost = [C_FN * confusion_matrix(y_clf, (oof_p >= t).astype(int))[1, 0] +
        C_FP * confusion_matrix(y_clf, (oof_p >= t).astype(int))[0, 1] for t in grid]
thr_cost = float(grid[int(np.argmin(cost))])
tn, fp, fn, tp = confusion_matrix(y_clf, (oof_p >= thr_cost).astype(int)).ravel()
METRICS = {
    "n_hexes": int(N), "n_features": len(FEATS),
    "r2_spatial": round(float(r2_score(y_reg, oof_r)), 4),
    "mae_spatial": round(float(mean_absolute_error(y_reg, oof_r)), 4),
    "auc_spatial": round(float(roc_auc_score(y_clf, oof_p)), 4),
    "thr_cost": thr_cost, "cost_ratio": "3:1",
    "fnr": round(fn / (fn + tp), 4), "fpr": round(fp / (fp + tn), 4),
    "precision": round(tp / (tp + fp), 4), "recall": round(tp / (tp + fn), 4),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "ecorisk_threshold": THRESH, "cv": "GroupKFold(6) по 12 пространственным блокам KMeans",
}
print("компактная модель (spatial CV): R²=%.3f MAE=%.3f AUC=%.3f | порог %.3f -> FNR=%.1f%% FPR=%.1f%%"
      % (METRICS["r2_spatial"], METRICS["mae_spatial"], METRICS["auc_spatial"],
         thr_cost, 100 * METRICS["fnr"], 100 * METRICS["fpr"]))

# ---- финальные модели на всех данных + компактная выгрузка деревьев ---------
def base_score_of(booster):
    """XGBoost 2.x подбирает base_score по данным; для binary:logistic он хранится
    как вероятность, а суммирование листьев идёт в логит-пространстве."""
    cfg = json.loads(booster.save_config())["learner"]
    b = float(cfg["learner_model_param"]["base_score"])
    if cfg["objective"]["name"] == "binary:logistic":
        b = float(np.log(b / (1.0 - b)))
    return b

def dump_trees(booster):
    base = base_score_of(booster)
    df = booster.trees_to_dataframe()
    out = []
    for t, g in df.groupby("Tree", sort=True):
        idx = {nid: k for k, nid in enumerate(g["ID"].tolist())}
        nodes = []
        for _, r in g.iterrows():
            if r["Feature"] == "Leaf":
                nodes.append([-1, round(float(r["Gain"]), 6), -1, -1])
            else:
                # порог храним девятью значащими цифрами: этого хватает, чтобы Math.fround
                # в браузере восстановил ровно тот же float32, с которым сравнивает XGBoost
                nodes.append([FEATS.index(r["Feature"]), float("%.9g" % np.float32(r["Split"])),
                              idx[r["Yes"]], idx[r["No"]]])
        out.append(nodes)
    return {"base_score": base, "trees": out}

reg = mk_reg().fit(X, y_reg)
clf = mk_clf().fit(X, y_clf)
MODEL = {
    "features": FEATS,
    "regressor": dump_trees(reg.get_booster()),
    "classifier": dump_trees(clf.get_booster()),
    "note": "инференс: сумма листьев по деревьям + base_score; классификатор через сигмоиду",
}
json.dump(MODEL, open(os.path.join(SVC, "data", "model.json"), "w"), separators=(",", ":"))

# сверка: JS будет считать так же, проверяем питоновской реализацией
def walk(trees, row):
    """Та же арифметика, что и в браузере: сравнение в float32."""
    s = trees["base_score"]
    for nodes in trees["trees"]:
        k = 0
        while nodes[k][0] != -1:
            f, thr, l, r = nodes[k]
            k = l if np.float32(row[f]) < np.float32(thr) else r
        s += nodes[k][1]
    return s
def sigmoid(v): return 1.0 / (1.0 + np.exp(-v))
dr = np.abs(np.array([walk(MODEL["regressor"], X.values[i]) for i in range(N)]) - reg.predict(X)).max()
dc = np.abs(sigmoid(np.array([walk(MODEL["classifier"], X.values[i]) for i in range(N)]))
            - clf.predict_proba(X)[:, 1]).max()
print("сверка выгрузки деревьев: регрессор max|Δ| = %.2e, классификатор max|Δ| = %.2e" % (dr, dc))
assert dr < 1e-4 and dc < 1e-4, "выгрузка деревьев расходится с xgboost"

# ---- соседи для Getis-Ord Gi* (кольцо гексагона) ---------------------------
xy = np.c_[hx["lng"].values * np.cos(np.radians(43.27)), hx["lat"].values]
from scipy.spatial import cKDTree
d, nb = cKDTree(xy).query(xy, k=7)
NEIGH = nb[:, 1:].astype(int).tolist()

# ---- колоночная выгрузка гексов -------------------------------------------
pred_full = reg.predict(X)
HEX = {
    "id": hx["hex_id"].tolist(),
    "lat": [round(v, 5) for v in hx["lat"]],
    "lng": [round(v, 5) for v in hx["lng"]],
    "district": [DISTRICTS.index(v) for v in hx["district"]],
    "landuse": [LANDUSE.index(v) for v in hx["landuse"]],
    "truth": [round(float(v), 3) for v in y_reg],
    "high_true": [int(v) for v in y_clf],
    "pred_oof": [round(float(v), 3) for v in oof_r],
    "prob_oof": [round(float(v), 4) for v in oof_p],
    "feat": {c: [round(float(v), 4) for v in X[c]] for c in FEATS},
}
json.dump(HEX, open(os.path.join(SVC, "data", "hexes.json"), "w"), separators=(",", ":"))

META = {
    "city": "Алматы", "grid_km": 0.35,
    "bbox": {"lat_min": float(hx["lat"].min()), "lat_max": float(hx["lat"].max()),
             "lng_min": float(hx["lng"].min()), "lng_max": float(hx["lng"].max())},
    "districts": DISTRICTS, "landuse": LANDUSE,
    "vgi_features": VGI, "gis_features": GIS, "features": FEATS,
    "neighbors": NEIGH, "metrics": METRICS,
    "categories": ["Засохшее дерево", "Пень / вырубка", "Нелегальная вырубка", "Стихийная свалка",
                   "Парковка на газоне", "Отсутствие озеленения двора", "Асфальтированный газон",
                   "Переполненный мусорный бак", "Сломанная скамейка или фонарь", "Подтопление",
                   "Дымящая труба", "Хорошее место"],
    "behavioural": [3, 4, 6, 7, 10],
    "actions": {"Засохшее дерево": "Санитарная обрезка и посадка", "Пень / вырубка": "Озеленение и посадка",
                "Нелегальная вырубка": "Экологический надзор", "Стихийная свалка": "Санитарная очистка",
                "Парковка на газоне": "Контроль и штрафы", "Отсутствие озеленения двора": "Озеленение и посадка",
                "Асфальтированный газон": "Благоустройство", "Переполненный мусорный бак": "Санитарная очистка",
                "Сломанная скамейка или фонарь": "Благоустройство", "Подтопление": "Инженерные сети",
                "Дымящая труба": "Экологический надзор", "Хорошее место": "Сохранить и тиражировать"},
}
json.dump(META, open(os.path.join(SVC, "data", "meta.json"), "w"), ensure_ascii=False, separators=(",", ":"))

# ---- статический read-only API --------------------------------------------
api = os.path.join(SVC, "api", "v1")
feats_geo = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [round(float(hx["lng"][i]), 5), round(float(hx["lat"][i]), 5)]},
    "properties": {"hex_id": hx["hex_id"][i], "district": hx["district"][i],
                   "ecorisk": round(float(pred_full[i]), 2), "ecorisk_oof": round(float(oof_r[i]), 2),
                   "high_risk_prob": round(float(oof_p[i]), 3),
                   "ndvi": round(float(hx["ndvi"][i]), 3), "lst_c": round(float(hx["lst_c"][i]), 1),
                   "reports": int(hx["rep_count"][i]), "landuse": hx["landuse"][i]},
} for i in range(N)]
json.dump({"type": "FeatureCollection", "crs": "EPSG:4326", "license": "CC-BY-4.0",
           "generated": "demo", "features": feats_geo},
          open(os.path.join(api, "hexes.geojson"), "w"), ensure_ascii=False, separators=(",", ":"))

dist = []
for d_i, name in enumerate(DISTRICTS):
    m = hx["district"] == name
    dist.append({"district": name, "hexes": int(m.sum()),
                 "mean_ecorisk": round(float(pred_full[m.values].mean()), 3),
                 "high_risk_hexes": int((oof_p[m.values] >= thr_cost).sum()),
                 "mean_ndvi": round(float(hx.loc[m, "ndvi"].mean()), 3),
                 "mean_lst_c": round(float(hx.loc[m, "lst_c"].mean()), 2),
                 "reports": int(hx.loc[m, "rep_count"].sum())})
dist.sort(key=lambda r: -r["mean_ecorisk"])
json.dump(dist, open(os.path.join(api, "districts.json"), "w"), ensure_ascii=False, indent=1)

json.dump({"model": "XGBoost (компактная сборка для браузера)", "metrics": METRICS,
           "features": FEATS, "license": "CC-BY-4.0"},
          open(os.path.join(api, "metrics.json"), "w"), ensure_ascii=False, indent=1)

order = np.argsort(-pred_full)[:100]
json.dump([{"rank": int(k + 1), "hex_id": hx["hex_id"][i], "district": hx["district"][i],
            "lat": round(float(hx["lat"][i]), 5), "lng": round(float(hx["lng"][i]), 5),
            "ecorisk": round(float(pred_full[i]), 2), "ndvi": round(float(hx["ndvi"][i]), 3),
            "landuse": hx["landuse"][i]} for k, i in enumerate(order)],
          open(os.path.join(api, "priority.json"), "w"), ensure_ascii=False, indent=1)

for f in ("data/hexes.json", "data/model.json", "data/meta.json",
          "api/v1/hexes.geojson", "api/v1/districts.json", "api/v1/priority.json", "api/v1/metrics.json"):
    print(f"  {f:26s} {os.path.getsize(os.path.join(SVC, f))/1024:8.1f} КБ")
