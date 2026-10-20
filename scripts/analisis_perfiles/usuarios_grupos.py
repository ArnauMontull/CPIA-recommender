"""k-means (k=6) sobre los vectores u de los 10.000 usuarios al azar + un perfil por grupo (sus ratings reales) para nombrarlos."""
import json, os, sys
from collections import Counter
import numpy as np, pandas as pd, torch, yaml
from sklearn.cluster import KMeans
from src.datos import cargar
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ

S = sys.argv[1]; K = int(sys.argv[2])
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
datos = cargar(PROCESADOS/"completo")
m = DosTorres.desde_config(datos.feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
with torch.no_grad():
    V = m.vectores_items()
    Ab, Bb = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    idx = np.sort(np.random.default_rng(0).choice(datos.n_usuarios, 10000, replace=False))   # misma muestra que en mapa_calc.py
    U = torch.cat([m.vectores_usuarios(V, torch.from_numpy(idx[i:i+2500]), Ab, Bb) for i in range(0, len(idx), 2500)]).numpy()
    gente = {}
    for al in os.environ["PERSONAS"].split(","):
        ld = lambda n: np.load(base/al/n)
        gente[al] = m.vectores_usuarios(V, torch.tensor([0]), bolsa(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), bolsa(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy()
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(U); lab = km.labels_
cn = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
asig = {al: int((cn @ u).argmax()) for al, u in gente.items()}
print("tamaños:", np.bincount(lab).tolist()); print("grupo más cercano (coseno al centroide):", {a.replace("_api", ""): g for a, g in asig.items()})

mu = pd.read_parquet(PROCESADOS/"completo"/"mapeo_usuarios.parquet").set_index("user_idx").userId
mp = pd.read_parquet(PROCESADOS/"completo"/"mapeo_peliculas.parquet").set_index("movieId") if (PROCESADOS/"completo"/"mapeo_peliculas.parquet").exists() else None
mv = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId", "titulo", "anio", "generos"]).set_index("movieId")
R = pd.read_parquet(PROCESADOS/"ratings.parquet", columns=["userId", "movieId", "rating"])
grupo_de_usuario = pd.Series(lab, index=mu.loc[idx].to_numpy())
R = R[R.userId.isin(grupo_de_usuario.index)].copy()
R["g"] = grupo_de_usuario.loc[R.userId].to_numpy()
R["c"] = R.rating - R.groupby("userId").rating.transform("mean")          # nota centrada en la media del usuario
nombre = lambda mid: f"{mv.titulo.get(mid, mid)} ({'' if pd.isna(mv.anio.get(mid)) else int(mv.anio.get(mid))})"
gen = lambda mid: [x for x in (mv.generos.get(mid) if not isinstance(mv.generos.get(mid), str) else mv.generos.get(mid).split("|")) if x] if mid in mv.index and mv.generos.get(mid) is not None else []
R["gen"] = R.movieId.map(lambda i: gen(i))
tot_pel = R.groupby("movieId").size(); n_tot = len(R)
NL = chr(10)
os.makedirs(S + "_usuarios", exist_ok=True)
for g in range(K):
    Rg = R[R.g == g]; nu = int((lab == g).sum())
    pg = Rg.groupby("movieId").size()
    sobre = (pg / len(Rg)) / (tot_pel.loc[pg.index] / n_tot)                 # sobre-representación de la película en el grupo
    sobre = sobre[pg >= max(25, 0.01 * nu)].sort_values(ascending=False).head(25)
    cen = Rg.groupby("movieId").c.agg(["mean", "size"]); cen = cen[cen["size"] >= max(25, 0.01 * nu)].sort_values("mean", ascending=False)
    pop = pg.sort_values(ascending=False).head(25)
    def lista(it): return NL.join("- " + nombre(i) for i in it)
    gt = Counter(x for l in R.gen for x in l); gg = Counter(x for l in Rg.gen for x in l)
    gd = Counter(x for l in R.gen for x in l)
    rel = {k: (gg[k] / len(Rg)) / (gt[k] / n_tot) for k in gg if gt[k] > 2000}
    gen_over = ", ".join(f"{k} x{v:.2f}" for k, v in sorted(rel.items(), key=lambda a: -a[1])[:6])
    gen_under = ", ".join(f"{k} x{v:.2f}" for k, v in sorted(rel.items(), key=lambda a: a[1])[:4])
    gnota = Rg.explode("gen").groupby("gen").c.agg(["mean", "size"]); gnota = gnota[gnota["size"] >= 2000].sort_values("mean")
    anios = pd.Series([mv.anio.get(i) for i in Rg.movieId]).dropna().astype(int)
    dec = dict(Counter(anios // 10 * 10).most_common(3))
    cuerpo = NL.join([f"# Grupo {g}: {nu} usuarios, {len(Rg)} ratings", f"Nota media {Rg.rating.mean():.2f} (global {R.rating.mean():.2f}); ratings por usuario {len(Rg)/nu:.0f} (global {n_tot/10000:.0f}).",
        f"Géneros sobre-representados (x veces el reparto global): {gen_over}", f"Géneros infra-representados: {gen_under}",
        "Géneros que más valora (nota centrada): " + ", ".join(f"{k} {v:+.2f}" for k, v in gnota['mean'].iloc[::-1].head(4).items()),
        "Géneros que menos valora: " + ", ".join(f"{k} {v:+.2f}" for k, v in gnota['mean'].head(3).items()), f"Décadas de las películas que ve: {dec}", "",
        "## 25 películas más sobre-representadas en el grupo (las ve mucho más que el resto)", lista(sobre.index), "",
        "## 25 mejor valoradas por el grupo respecto a la media de cada usuario", lista(cen.index[:25]), "",
        "## 25 más vistas dentro del grupo", lista(pop.index), ""])
    open(f"{S}_usuarios/grupo_{g}.md", "w", encoding="utf-8").write(cuerpo)
json.dump({"labels": lab.tolist(), "personas": {a.replace("_api", ""): g for a, g in asig.items()}, "tam": np.bincount(lab).tolist()}, open(S + "_usuarios.json", "w"))
print("ok")
