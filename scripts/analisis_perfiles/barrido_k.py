"""Barrido de K de k-means para películas (V) y usuarios (10.000 u): inercia, silhouette, Davies-Bouldin y Calinski-Harabasz."""
import json, sys, time
import numpy as np, torch, yaml
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from src.datos import cargar
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import PROCESADOS, RAIZ

OUT = sys.argv[1]
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8"))
datos = cargar(PROCESADOS/"completo")
m = DosTorres.desde_config(datos.feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
with torch.no_grad():
    Vt = m.vectores_items()
    Ab, Bb = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    idx = np.sort(np.random.default_rng(0).choice(datos.n_usuarios, 10000, replace=False))   # la misma muestra de siempre
    U = torch.cat([m.vectores_usuarios(Vt, torch.from_numpy(idx[i:i + 2500]), Ab, Bb) for i in range(0, len(idx), 2500)]).numpy()
V = Vt.numpy()
rng = np.random.default_rng(1)
res = {}
for nombre, X, ks in (("peliculas", V, range(2, 31)), ("usuarios", U, range(2, 21))):
    sub = rng.choice(len(X), 5000, replace=False)           # submuestra para silhouette (cuadrática)
    filas = []
    for k in ks:
        t = time.time()
        km = KMeans(k, n_init=3, random_state=0).fit(X)
        filas.append({"k": k, "inercia": float(km.inertia_), "silhouette": float(silhouette_score(X[sub], km.labels_[sub])),
                      "db": float(davies_bouldin_score(X, km.labels_)), "ch": float(calinski_harabasz_score(X, km.labels_)),
                      "min_tam": int(np.bincount(km.labels_).min()), "seg": round(time.time() - t, 1)})
        print(nombre, filas[-1], flush=True)
    res[nombre] = filas
json.dump(res, open(OUT + "_barrido.json", "w"))
print("ok")
