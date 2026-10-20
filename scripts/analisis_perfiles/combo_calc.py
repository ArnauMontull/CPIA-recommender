"""Dos PCA (ejes de máxima varianza de las películas / de los usuarios) y todo proyectado en ambos: películas con su grupo, 10.000 usuarios con su grupo y las personas."""
import json, os, sys
import numpy as np, pandas as pd, torch, yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.datos import cargar
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ

OUT = sys.argv[1]; KF, KU = int(os.environ.get("KF", 12)), int(os.environ.get("KU", 6))
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
datos = cargar(PROCESADOS/"completo")
m = DosTorres.desde_config(datos.feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
with torch.no_grad():
    Vt = m.vectores_items()
    Ab, Bb = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    Uall = torch.cat([m.vectores_usuarios(Vt, torch.arange(i, min(i + 4000, datos.n_usuarios)), Ab, Bb) for i in range(0, datos.n_usuarios, 4000)]).numpy()
    gente = {}
    for al in os.environ["PERSONAS"].split(","):
        ld = lambda n: np.load(base/al/n)
        gente[al.replace("_api", "").capitalize()] = m.vectores_usuarios(Vt, torch.tensor([0]), bolsa(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")),
                                                                         bolsa(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy()
V = Vt.numpy()
idx = np.sort(np.random.default_rng(0).choice(datos.n_usuarios, 10000, replace=False))     # misma muestra que antes
U10 = Uall[idx]
pcaV = PCA(3, random_state=0).fit(V); pcaU = PCA(3, random_state=0).fit(Uall)
labF = KMeans(KF, n_init=10, random_state=0).fit(V).labels_
labU = KMeans(KU, n_init=10, random_state=0).fit(U10).labels_
print("varianza PCA películas:", pcaV.explained_variance_ratio_.round(3), "| PCA usuarios:", pcaU.explained_variance_ratio_.round(3))

mapeo = pd.read_parquet(PROCESADOS/"dev"/"mapeo_peliculas.parquet").set_index("item_idx")
mv = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId", "titulo", "anio"]).set_index("movieId")
d = mv.loc[mapeo.loc[np.arange(len(V)), "movieId"].to_numpy()]
titulos = [t + (f" ({int(a)})" if pd.notna(a) else "") for t, a in zip(d.titulo, d.anio)]
r = lambda a: np.round(a, 3)
both = lambda X: np.hstack([pcaV.transform(X), pcaU.transform(X)])
F, U, G = both(V), both(U10), {k: both(u[None])[0] for k, u in gente.items()}
cn = KMeans(KU, n_init=10, random_state=0).fit(U10).cluster_centers_; cn = cn / np.linalg.norm(cn, axis=1, keepdims=True)
json.dump({"films": [list(map(float, r(F[i]))) + [int(labF[i]), titulos[i]] for i in range(len(V))],
           "users": [list(map(float, r(U[i]))) + [int(labU[i])] for i in range(len(U))],
           "people": {k: list(map(float, r(v))) + [int((cn @ gente[k]).argmax())] for k, v in G.items()},
           "varV": pcaV.explained_variance_ratio_.tolist(), "varU": pcaU.explained_variance_ratio_.tolist(),
           "tamU": np.bincount(labU).tolist()}, open(OUT + "_combo.json", "w", encoding="utf-8"), ensure_ascii=False)
print("ok")
