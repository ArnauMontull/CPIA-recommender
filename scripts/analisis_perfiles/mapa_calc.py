import numpy as np, pandas as pd, torch, yaml, json, sys
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.datos import cargar_features
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ
K = int(sys.argv[1]); OUT = sys.argv[2]
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8"))
base = DATA/cfg["letterboxd"]["carpeta"]
feats = cargar_features(PROCESADOS/"completo")
m = DosTorres.desde_config(feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
with torch.no_grad():
    V = m.vectores_items(); us = {}
    for al in tuple(__import__("os").environ["PERSONAS"].split(",")):
        ld = lambda n: np.load(base/al/n)
        us[al] = m.vectores_usuarios(V, torch.tensor([0]), bolsa(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), bolsa(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy()
V = V.numpy()
pca = PCA(n_components=3, random_state=0).fit(V)
P = pca.transform(V); print("varianza explicada PC1-3:", pca.explained_variance_ratio_.round(3))
from src.datos import cargar
datos = cargar(PROCESADOS/"completo")
with torch.no_grad():
    Vt = torch.from_numpy(V)
    Ab, Bb = Bolsa(*datos.hist[("train","A")]), Bolsa(*datos.hist[("train","B")])
    idx = np.sort(np.random.default_rng(0).choice(datos.n_usuarios, 10000, replace=False))
    Uu = torch.cat([m.vectores_usuarios(Vt, torch.from_numpy(idx[i:i+2500]), Ab, Bb) for i in range(0, len(idx), 2500)]).numpy()
USU = pca.transform(Uu)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(V); lab = km.labels_
mapeo = pd.read_parquet(PROCESADOS/"dev"/"mapeo_peliculas.parquet").set_index("item_idx")
mv = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId","titulo","anio","generos","n_ratings"]).set_index("movieId")
df = mv.loc[mapeo.loc[np.arange(len(V)), "movieId"].to_numpy()].reset_index()
df["c"] = lab; df["x"] = P[:,0]; df["y"] = P[:,1]; df["z"] = P[:,2]
# --- dónde cae cada persona: proyección de u en el PCA, y grupo de centroide más cercano (coseno) ---
cn = km.cluster_centers_/np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
gente = {}
for al, u in us.items():
    sc = V@u; sg = pd.Series(sc).groupby(lab).mean()
    gente[al] = {"xy": pca.transform(u[None])[0].tolist(), "grupo_centroide": int((cn@u).argmax()), "afinidad_grupos": sg.round(3).to_dict(),
                 "top_grupos_por_score_medio": [int(x) for x in sg.sort_values(ascending=False).index[:3]]}
    print(al, "xy", np.round(gente[al]["xy"],3), "| grupo más afín (score medio):", gente[al]["top_grupos_por_score_medio"])
# --- descripción de cada grupo ---
for c in range(K):
    d = df[df.c==c]; g = Counter(x for s in d.generos for x in (s if not isinstance(s,str) else s.split("|")) if x)
    dec = Counter((d.anio.dropna()//10*10).astype(int))
    print(f"\n### grupo {c}: {len(d)} pelis | géneros: {', '.join(f'{k} {v/len(d):.0%}' for k,v in g.most_common(4))} | décadas: {dict(dec.most_common(2))}")
    print("   centrales:", "; ".join(f"{r.titulo} ({int(r.anio) if pd.notna(r.anio) else ''})" for r in d.assign(dist=np.linalg.norm(V[d.index]-km.cluster_centers_[c],axis=1)).sort_values("dist").head(8).itertuples()))
    print("   populares:", "; ".join(f"{r.titulo}" for r in d.sort_values("n_ratings", ascending=False).head(8).itertuples()))
import os; os.makedirs(OUT+"_grupos", exist_ok=True)
NL = chr(10)
for c in range(K):
    d = df[df.c==c]; g = Counter(x for s_ in d.generos for x in (s_ if not isinstance(s_,str) else s_.split("|")) if x)
    cen = d.assign(dist=np.linalg.norm(V[d.index]-km.cluster_centers_[c],axis=1)).sort_values("dist").head(25)
    pop = d.sort_values("n_ratings", ascending=False).head(25)
    f = lambda t: NL.join("- %s (%s)" % (r.titulo, int(r.anio) if pd.notna(r.anio) else "?") for r in t.itertuples())
    gen = ", ".join("%s %.0f%%" % (k, 100*v/len(d)) for k,v in g.most_common(6))
    dec = dict(Counter((d.anio.dropna()//10*10).astype(int)).most_common(3))
    txt = NL.join(["# Grupo %d: %d peliculas" % (c, len(d)), "Generos: " + gen, "Decadas mas comunes: %s" % dec, "", "## 25 mas centrales", f(cen), "", "## 25 mas votadas", f(pop), ""])
    open("%s_grupos/grupo_%d.md" % (OUT, c), "w", encoding="utf-8").write(txt)
df.to_parquet(OUT+".parquet"); json.dump({"usuarios": np.round(USU,3).tolist(), "gente": gente, "var": pca.explained_variance_ratio_.tolist(), "K": K}, open(OUT+".json","w"))
