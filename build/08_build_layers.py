# -*- coding: utf-8 -*-
"""Дополнительные слои портала: тепловые карты по 12 категориям обращений
и кластеризация кварталов по «синдромам» — типичным сочетаниям проблем.

Пишет service/data/layers.json.
"""
import json, os, collections
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = HERE if os.path.exists(os.path.join(HERE, "reports.csv")) else os.path.join(HERE, "..", "..", "data")
OUT  = os.path.join(HERE, "..", "data")

hx = pd.read_csv(os.path.join(SRC if os.path.exists(os.path.join(SRC, "hexes.csv")) else HERE, "hexes.csv"))
rp = pd.read_csv(os.path.join(SRC, "reports.csv"))
idx = {h: i for i, h in enumerate(hx["hex_id"])}
CATS = list(rp["category"].value_counts().index)
CATS.sort()
ci = {c: k for k, c in enumerate(CATS)}
N, C = len(hx), len(CATS)

cnt = np.zeros((N, C), dtype=np.int32)
sev = np.zeros((N, C), dtype=np.float64)
ver = np.zeros((N, C), dtype=np.int32)
for h, c, s, v in zip(rp["hex_id"], rp["category"], rp["severity"], rp["verified"]):
    i, k = idx.get(h), ci[c]
    if i is None: continue
    cnt[i, k] += 1; sev[i, k] += s; ver[i, k] += int(v)
mean_sev = np.divide(sev, np.maximum(cnt, 1))

# «синдромы»: кластеризация кварталов по доле категорий в их обращениях.
# Методика называет HDBSCAN; здесь k-средние по нормированному профилю — тот же
# смысл (сочетание проблем), но воспроизводимо и без зависимости сверх sklearn.
tot = cnt.sum(1)
prof = np.divide(cnt, np.maximum(tot, 1)[:, None])
act = tot >= 3
km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(prof[act])
lab = np.full(N, -1, dtype=int); lab[act] = km.labels_
syn = []
for k in range(6):
    m = lab == k
    share = cnt[m].sum(0) / max(1, cnt[m].sum())
    top = np.argsort(-share)[:3]
    syn.append({"id": k, "hexes": int(m.sum()),
                "name": " + ".join(CATS[t] for t in top[:2]),
                "top": [{"category": CATS[t], "share": round(float(share[t]), 3)} for t in top],
                "mean_reports": round(float(tot[m].mean()), 1)})
syn.sort(key=lambda s: -s["hexes"])

sparse = {}
for i in range(N):
    if tot[i]:
        sparse[str(i)] = [int(v) for v in cnt[i]]

json.dump({
    "categories": CATS,
    "counts": sparse,
    "totals": [int(v) for v in cnt.sum(0)],
    "mean_severity": [round(float(v), 2) for v in
                      np.divide(sev.sum(0), np.maximum(cnt.sum(0), 1))],
    "verified_share": [round(float(a / max(1, b)), 3) for a, b in zip(ver.sum(0), cnt.sum(0))],
    "syndrome": [int(v) for v in lab],
    "syndromes": syn,
    "n_reports": int(len(rp)),
    "n_users": int(rp["user_id"].nunique()),
}, open(os.path.join(OUT, "layers.json"), "w"), ensure_ascii=False, separators=(",", ":"))

print(f"категорий: {C}, кварталов с обращениями: {len(sparse)}, отчётов: {len(rp)}")
for s in syn:
    print(f"  синдром «{s['name']}»: {s['hexes']} кварталов, в среднем {s['mean_reports']} обращений")
print("layers.json:", round(os.path.getsize(os.path.join(OUT, 'layers.json')) / 1024, 1), "КБ")
