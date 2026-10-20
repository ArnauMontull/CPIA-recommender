"""Por persona: usuarios más afines del dataset, afinidad con los grupos de usuarios (K=5) y de películas (K=15) y top-20 con MMR (λ=0.7 sobre los 200 mejores)."""
import json, os, sys
import numpy as np, pandas as pd, torch, yaml
from sklearn.cluster import KMeans
from src.datos import cargar
from src.demo_letterboxd import mmr, recomendar_usuario
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ

S = sys.argv[1]
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
datos = cargar(PROCESADOS/"completo")
m = DosTorres.desde_config(datos.feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
with torch.no_grad():
    V = m.vectores_items()
    Ab, Bb = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    Uall = torch.cat([m.vectores_usuarios(V, torch.arange(i, min(i + 4000, datos.n_usuarios)), Ab, Bb) for i in range(0, datos.n_usuarios, 4000)]).numpy()
idx = np.sort(np.random.default_rng(0).choice(datos.n_usuarios, 10000, replace=False))
km = KMeans(5, n_init=10, random_state=0).fit(Uall[idx]); cu = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
NU = json.load(open(S + "/nombres_usuarios5.json", encoding="utf-8")); NF = json.load(open(S + "/nombres15.json", encoding="utf-8"))
labF = pd.read_parquet(S + "/k15.parquet", columns=["c"]).c.to_numpy()
mu = pd.read_parquet(PROCESADOS/"completo"/"mapeo_usuarios.parquet").set_index("user_idx").userId
mapeo = pd.read_parquet(PROCESADOS/"dev"/"mapeo_peliculas.parquet").set_index("item_idx")
mv = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId", "titulo", "anio", "n_ratings"]).set_index("movieId")
nom = lambda mid: f"{mv.titulo[mid]} ({'' if pd.isna(mv.anio[mid]) else int(mv.anio[mid])})"
nomi = lambda i: nom(int(mapeo.loc[int(i), "movieId"]))
R = pd.read_parquet(PROCESADOS/"ratings.parquet", columns=["userId", "movieId", "rating"])
ang = lambda c: float(np.degrees(np.arccos(np.clip(c, -1, 1))))
out = {}
for al in os.environ["PERSONAS"].split(","):
    nombre = al.replace("_api", "").capitalize(); ld = lambda n: np.load(base/al/n)
    A, B, val = (ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), (ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")), ld("valorados_indices.npy")
    with torch.no_grad():
        u = m.vectores_usuarios(V, torch.tensor([0]), bolsa(*A), bolsa(*B))[0].numpy()
    top, sc = recomendar_usuario(m, V, A, B, val, 200)
    el = mmr(V, top, sc, 20, 0.7); scm = sc[[int(np.where(top == i)[0][0]) for i in el]]
    recs = [(nomi(i), float(s), NF[labF[int(i)]]) for i, s in zip(el, scm)]
    # usuarios más afines (coseno) sobre los 200.948
    cs = Uall @ u; vec = np.argsort(-cs)[:5]; vecinos = []
    for j in vec:
        r = R[R.userId == mu[int(j)]]; fav = r[r.rating >= 4.5].movieId
        fav = sorted(fav, key=lambda mid: -mv.n_ratings.get(mid, 0))[:5]
        vecinos.append({"userId": int(mu[int(j)]), "angulo": ang(cs[j]), "n": len(r), "media": float(r.rating.mean()), "favoritas": [nom(x) for x in fav]})
    sg = V.numpy() @ u; gf = pd.Series(sg).groupby(labF).mean().sort_values(ascending=False)
    out[nombre] = {"recs": recs, "vecinos": vecinos, "pct_vecino5": float((cs > cs[vec[4]]).mean() * 100),
                   "grupos_usuarios": [(NU[g], ang(float(cu[g] @ u))) for g in np.argsort([-(cu[g] @ u) for g in range(5)])],
                   "grupos_pelis": [(NF[int(g)], float(v)) for g, v in gf.items()], "n_val": int(len(val)), "ang_medio_dataset": ang(float(cs.mean()))}
json.dump(out, open(S + "/analisis_personas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
