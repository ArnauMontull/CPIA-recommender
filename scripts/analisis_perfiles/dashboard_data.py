"""Todos los datos del dashboard: perfiles, ángulos y percentiles, espacio de embedding, grupos, recomendaciones y curiosidades."""
import json, os, sys
from collections import Counter
import numpy as np, pandas as pd, torch, yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.datos import cargar
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ

S = sys.argv[1]
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
ALIAS = os.environ["PERSONAS"].split(","); NOM = [a.replace("_api", "").capitalize() for a in ALIAS]; P = len(NOM)
jr = lambda n: json.load(open(f"{S}/{n}", encoding="utf-8"))
NF, NU = jr("nombres15.json"), jr("nombres_usuarios5.json")
R2 = lambda x, k=2: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)

# ---------- modelo, vectores ----------
datos = cargar(PROCESADOS/"completo")
m = DosTorres.desde_config(datos.feats, cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
with torch.no_grad():
    Vt = m.vectores_items()
    Ab, Bb = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    Uall = torch.cat([m.vectores_usuarios(Vt, torch.arange(i, min(i + 4000, datos.n_usuarios)), Ab, Bb) for i in range(0, datos.n_usuarios, 4000)]).numpy()
    Up = []
    for al in ALIAS:
        ld = lambda n: np.load(base/al/n)
        Up.append(m.vectores_usuarios(Vt, torch.tensor([0]), bolsa(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), bolsa(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy())
V = Vt.numpy(); Up = np.array(Up); NUS = len(Uall)
ang = lambda c: np.degrees(np.arccos(np.clip(c, -1, 1)))
mu = pd.read_parquet(PROCESADOS/"completo"/"mapeo_usuarios.parquet").set_index("user_idx").userId

# ---------- catálogo y perfiles ----------
mv = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId", "titulo", "anio", "generos", "n_ratings"]).set_index("movieId")
mapeo = pd.read_parquet(PROCESADOS/"dev"/"mapeo_peliculas.parquet").set_index("item_idx")
labF = pd.read_parquet(f"{S}/k15.parquet", columns=["c"]).c.to_numpy()
gens = lambda mid: [g for g in (mv.generos.get(mid) if not isinstance(mv.generos.get(mid), str) else mv.generos.get(mid).split("|")) if g and g != "(no genres listed)"]
nom = lambda mid: f"{mv.titulo[mid]} ({'' if pd.isna(mv.anio[mid]) else int(mv.anio[mid])})"
E = [pd.read_parquet(base/al/"emparejado.parquet") for al in ALIAS]
Mt = []
for e in E:
    t = e.dropna(subset=["movieId"]).drop_duplicates("movieId").copy(); t["movieId"] = t.movieId.astype(int); t["anio"] = t.movieId.map(mv.anio); t["pop"] = t.movieId.map(mv.n_ratings)
    Mt.append(t)
gcount = Counter(g for t in Mt for mid in t.movieId for g in gens(mid)); GEN = [g for g, _ in gcount.most_common(18)]
DEC = list(range(1920, 2030, 10))
perfiles = []
for i, (e, t) in enumerate(zip(E, Mt)):
    r = e.rating.astype(float); h = [int((r == v).sum()) for v in np.arange(0.5, 5.01, 0.5)]
    dec = Counter(int(a) // 10 * 10 for a in t.anio.dropna()); gshare = Counter(g for mid in t.movieId for g in gens(mid))
    gnota = {}
    for g in GEN:
        v = [float(rt) - float(r.mean()) for mid, rt in zip(t.movieId, t.rating) if g in gens(mid)]
        gnota[g] = R2(np.mean(v)) if len(v) >= 3 else None
    fav = t[t.rating >= t.rating.max()].sort_values("pop", ascending=False).head(8)
    peor = t[t.rating <= 2].sort_values("pop", ascending=False).head(5)
    an = t.anio.dropna().astype(int)
    perfiles.append({"nombre": NOM[i], "n": int(len(e)), "emparejadas": int(len(t)), "pct_emp": R2(100 * len(t) / len(e), 1), "media": R2(r.mean()), "sd": R2(r.std()), "mediana": R2(r.median()),
        "hist": h, "pct45": R2(100 * (r >= 4.5).mean(), 1), "pct_bajas": R2(100 * (r <= 2).mean(), 1), "n_cinco": int((r == 5).sum()),
        "decadas": [int(dec.get(d, 0)) for d in DEC], "generos": [int(gshare.get(g, 0)) for g in GEN], "genero_nota": [gnota[g] for g in GEN],
        "anio_medio": R2(an.mean(), 1), "anio_mediana": int(an.median()), "pct_pre1980": R2(100 * (an < 1980).mean(), 1), "pct_post2015": R2(100 * (an >= 2015).mean(), 1),
        "mas_antigua": nom(int(t.loc[t.anio.idxmin(), "movieId"])), "mas_reciente": nom(int(t.loc[t.anio.idxmax(), "movieId"])),
        "pop_mediana": int(t["pop"].median()), "pct_blockbuster": R2(100 * (t["pop"] > 10000).mean(), 1), "pct_nicho": R2(100 * (t["pop"] < 1000).mean(), 1),
        "favoritas": [nom(int(x)) + f" ({rt:g}★)" for x, rt in zip(fav.movieId, fav.rating)], "peores": [nom(int(x)) + f" ({rt:g}★)" for x, rt in zip(peor.movieId, peor.rating)]})

# ---------- pares: ángulos, percentiles, en común ----------
A = ang(Up @ Up.T); np.fill_diagonal(A, 0)
rng = np.random.default_rng(5); ia, ib = rng.integers(0, NUS, 400000), rng.integers(0, NUS, 400000); ok = ia != ib
RP = ang((Uall[ia[ok]] * Uall[ib[ok]]).sum(1))
PCT = np.array([[100 * float((RP < A[i, j]).mean()) for j in range(P)] for i in range(P)])
bins = np.arange(0, 181, 2.5); hist_rp = np.histogram(RP, bins)[0] / len(RP) * 100
sets = [set(t.movieId) for t in Mt]; rat = [dict(zip(t.movieId, t.rating)) for t in Mt]
COM, JAC, COR, DIF = (np.zeros((P, P)) for _ in range(4)); COR[:] = np.nan
for i in range(P):
    for j in range(P):
        c = sets[i] & sets[j]; COM[i, j] = len(c); JAC[i, j] = 100 * len(c) / max(len(sets[i] | sets[j]), 1)
        if i != j and len(c) >= 5:
            a = np.array([rat[i][x] for x in c]); b = np.array([rat[j][x] for x in c]); COR[i, j] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan; DIF[i, j] = np.abs(a - b).mean()
D = squareform(A, checks=False); Z = linkage(D, "average"); dg = dendrogram(Z, no_plot=True, labels=NOM)
Dm = A ** 2; J = np.eye(P) - 1 / P; Bm = -0.5 * J @ Dm @ J; w, v = np.linalg.eigh(Bm); o = np.argsort(-w)[:2]; mds = v[:, o] * np.sqrt(np.maximum(w[o], 0))
# ---------- cada persona contra los 200.948 usuarios ----------
ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
vecindad, hist_u, resumen_u, top_ids = [], [], [], []
for i in range(P):
    c = Uall @ Up[i]; a = ang(c); s = np.sort(a)
    vecindad.append([float(s[k - 1]) for k in ks]); hist_u.append((np.histogram(a, bins)[0] / NUS * 100).tolist())
    resumen_u.append({"media": R2(a.mean(), 1), "mediana": R2(np.median(a), 1), "p5": R2(np.percentile(a, 5), 1), "p95": R2(np.percentile(a, 95), 1), "min": R2(s[0], 1),
                      "n_lt40": int((a < 40).sum()), "n_lt45": int((a < 45).sum()), "n_lt50": int((a < 50).sum()), "n_lt60": int((a < 60).sum()), "pct_lt60": R2(100 * (a < 60).mean(), 2)})
    top_ids.append(set(np.argsort(a)[:3000].tolist()))
inter = Counter(j for s in top_ids for j in s); compart = [(int(j), n) for j, n in inter.items() if n >= 3]
compart.sort(key=lambda x: -x[1]); Rt = pd.read_parquet(PROCESADOS/"ratings.parquet", columns=["userId", "movieId", "rating"])
vecinos_comp = []
for j, n in compart[:8]:
    r = Rt[Rt.userId == mu[j]]; fav = sorted(r[r.rating >= 4.5].movieId, key=lambda x: -mv.n_ratings.get(x, 0))[:4]
    vecinos_comp.append({"userId": int(mu[j]), "n_personas": n, "n_ratings": int(len(r)), "media": R2(r.rating.mean()), "favoritas": [nom(x) for x in fav],
                         "personas": [NOM[i] for i in range(P) if j in top_ids[i]]})
# ---------- espacio: PCA, grupos, afinidades ----------
idx = np.sort(np.random.default_rng(0).choice(NUS, 10000, replace=False)); U10 = Uall[idx]
pv, pu = PCA(20, random_state=0).fit(V), PCA(20, random_state=0).fit(Uall)
km5 = KMeans(5, n_init=10, random_state=0).fit(U10); cu = km5.cluster_centers_ / np.linalg.norm(km5.cluster_centers_, axis=1, keepdims=True)
sc = V @ Up.T  # películas x personas
afin_f = [[R2(sc[labF == g, i].mean(), 3) for g in range(len(NF))] for i in range(P)]
afin_u = [[R2(float(ang(cu[g] @ Up[i])), 1) for g in range(5)] for i in range(P)]
combo = jr("k15_combo.json")
sub = np.random.default_rng(2).choice(len(combo["films"]), 9000, replace=False)
espacio = {"films": [[R2(combo["films"][k][0], 3), R2(combo["films"][k][1], 3), R2(combo["films"][k][3], 3), R2(combo["films"][k][4], 3), combo["films"][k][6]] for k in sub],
           "users": [[R2(u[0], 3), R2(u[1], 3), R2(u[3], 3), R2(u[4], 3), u[6]] for u in combo["users"]],
           "people": {k: [R2(v[0], 3), R2(v[1], 3), R2(v[3], 3), R2(v[4], 3), v[6]] for k, v in combo["people"].items()}, "varV": combo["varV"], "varU": combo["varU"]}
barrido = jr("mapa12_barrido.json")
tam_f = np.bincount(labF, minlength=len(NF)).tolist(); tam_u = np.bincount(km5.labels_, minlength=5).tolist()
# estadística de los grupos de películas
mvg = pd.DataFrame({"anio": [mv.anio.get(int(mapeo.loc[i, "movieId"])) for i in range(len(V))], "pop": [mv.n_ratings.get(int(mapeo.loc[i, "movieId"])) for i in range(len(V))], "c": labF})
gf = mvg.groupby("c").agg(anio=("anio", "median"), pop=("pop", "median")).reset_index()
grupos_f = [{"nombre": NF[int(r.c)], "n": tam_f[int(r.c)], "anio_mediano": R2(r.anio, 0), "pop_mediana": R2(r["pop"], 0)} for _, r in gf.iterrows()]
# ---------- recomendaciones ----------
an = jr("analisis_personas.json"); recs = {n: an[n]["recs"] for n in NOM}
rl = {n: {t for t, _, _ in recs[n]} for n in NOM}
overlap = [[len(rl[a] & rl[b]) for b in NOM] for a in NOM]
cnt = Counter(t for n in NOM for t in rl[n]); mas_rec = [(t, c, [n for n in NOM if t in rl[n]]) for t, c in cnt.most_common(12) if c >= 2]
rec_grupo = [[sum(1 for _, _, g in recs[n] if g == NF[k]) for k in range(len(NF))] for n in NOM]
vecinos_p = {n: an[n]["vecinos"] for n in NOM}
# ---------- curiosidades ----------
cur = []
iu = [(i, j) for i in range(P) for j in range(i + 1, P)]; imin = min(iu, key=lambda p: A[p]); imax = max(iu, key=lambda p: A[p])
cur += [("Pareja más cercana", f"{NOM[imin[0]]} y {NOM[imin[1]]}", f"{A[imin]:.1f}° · más cerca que el {100 - PCT[imin]:.1f}% de pares de usuarios"),
        ("Pareja más lejana", f"{NOM[imax[0]]} y {NOM[imax[1]]}", f"{A[imax]:.1f}° · más lejos que el {PCT[imax]:.0f}% de pares de usuarios")]
gen_i = int(np.argmax([p["media"] for p in perfiles])); exi_i = int(np.argmin([p["media"] for p in perfiles]))
cur += [("Quien puntúa más alto", NOM[gen_i], f"media {perfiles[gen_i]['media']:.2f}"), ("Quien puntúa más bajo", NOM[exi_i], f"media {perfiles[exi_i]['media']:.2f}")]
mm = int(np.argmax([p["pop_mediana"] for p in perfiles])); nn = int(np.argmin([p["pop_mediana"] for p in perfiles]))
cur += [("Gustos más mainstream", NOM[mm], f"mediana de {perfiles[mm]['pop_mediana']:,} valoraciones por película en el catálogo".replace(",", ".")),
        ("Gustos más de nicho", NOM[nn], f"mediana de {perfiles[nn]['pop_mediana']:,} valoraciones por película".replace(",", "."))]
ant = min(range(P), key=lambda i: perfiles[i]["anio_medio"]); mod = max(range(P), key=lambda i: perfiles[i]["anio_medio"])
cur += [("Año medio más antiguo", NOM[ant], f"{perfiles[ant]['anio_medio']:.0f}"), ("Año medio más moderno", NOM[mod], f"{perfiles[mod]['anio_medio']:.0f}")]
tt = Counter(x for s in sets for x in s); top_vistas = tt.most_common(1)[0]
cur.append(("Película más vista entre nosotros", nom(top_vistas[0]), f"{top_vistas[1]} de {P} personas la han valorado"))
todas = set.intersection(*sets); cur.append(("Películas valoradas por las 7 personas", str(len(todas)), "; ".join(nom(x) for x in list(todas)[:4]) if todas else "ninguna"))
multi = [mid for mid, c in tt.items() if c >= 4]
dis = sorted(multi, key=lambda mid: -np.std([rat[i][mid] for i in range(P) if mid in rat[i]]))[:1]
if dis:
    mid = dis[0]; cur.append(("La película que más nos divide", nom(mid), ", ".join(f"{NOM[i]} {rat[i][mid]:g}★" for i in range(P) if mid in rat[i])))
cons = sorted(multi, key=lambda mid: np.std([rat[i][mid] for i in range(P) if mid in rat[i]]))[:1]
if cons:
    mid = cons[0]; cur.append(("La película que más nos une", nom(mid), ", ".join(f"{NOM[i]} {rat[i][mid]:g}★" for i in range(P) if mid in rat[i])))
c5 = int(np.argmax([p["n_cinco"] for p in perfiles])); cur.append(("Más 5★ repartidos", NOM[c5], f"{perfiles[c5]['n_cinco']} películas con 5★"))
raro = int(np.argmax([r["media"] for r in resumen_u])); tip = int(np.argmin([r["media"] for r in resumen_u]))
cur += [("Gusto más raro respecto al dataset", NOM[raro], f"ángulo medio a los {NUS:,} usuarios: {resumen_u[raro]['media']}°".replace(",", ".")), ("Gusto más típico respecto al dataset", NOM[tip], f"ángulo medio: {resumen_u[tip]['media']}°")]
cur.append(("Usuario del dataset más compartido", str(vecinos_comp[0]["userId"]) if vecinos_comp else "—", f"vecino cercano de {vecinos_comp[0]['n_personas']} de nosotros" if vecinos_comp else ""))
resumen = {"n_personas": P, "n_notas": int(sum(p["n"] for p in perfiles)), "n_emp": int(sum(p["emparejadas"] for p in perfiles)), "n_pelis_dist": int(len(set().union(*sets))),
           "n_usuarios": int(NUS), "ang_min": R2(A[imin], 1), "ang_max": R2(A[imax], 1), "ang_pares_medio": R2(np.mean([A[p] for p in iu]), 1), "ang_random_medio": R2(RP.mean(), 1), "ang_random_p5": R2(np.percentile(RP, 5), 1),
           "ang_random_p50": R2(np.percentile(RP, 50), 1), "ang_random_p95": R2(np.percentile(RP, 95), 1), "n_catalogo": int(len(V))}
comunes = [{"t": nom(mid), "c": c, "r": [rat[i].get(mid) for i in range(P)], "sd": R2(np.std([rat[i][mid] for i in range(P) if mid in rat[i]]))} for mid, c in tt.most_common(60) if c >= 3]
out = {"comunes": comunes, "nombres": NOM, "perfiles": perfiles, "GEN": GEN, "DEC": DEC, "A": A.round(1).tolist(), "PCT": PCT.round(1).tolist(), "COM": COM.astype(int).tolist(), "JAC": JAC.round(1).tolist(),
       "COR": [[R2(x, 2) for x in r] for r in COR], "DIF": DIF.round(2).tolist(), "bins": bins.tolist(), "hist_rp": hist_rp.round(3).tolist(), "hist_u": [[round(x, 3) for x in h] for h in hist_u],
       "ks": ks, "vecindad": [[round(x, 1) for x in v] for v in vecindad], "resumen_u": resumen_u, "dendro": {"icoord": dg["icoord"], "dcoord": dg["dcoord"], "leaves": dg["leaves"]},
       "mds": mds.round(2).tolist(), "vecinos_comp": vecinos_comp, "scree_v": pv.explained_variance_ratio_.round(4).tolist(), "scree_u": pu.explained_variance_ratio_.round(4).tolist(),
       "afin_f": afin_f, "afin_u": afin_u, "NF": NF, "NU": NU, "tam_f": tam_f, "tam_u": tam_u, "grupos_f": grupos_f, "espacio": espacio, "barrido": barrido, "recs": recs, "overlap": overlap,
       "mas_rec": mas_rec, "rec_grupo": rec_grupo, "vecinos_p": vecinos_p, "grupos_pelis_aff": {n: an[n]["grupos_pelis"] for n in NOM}, "curiosidades": cur, "resumen": resumen}
json.dump(out, open(S + "/dashboard_data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("ok", os.path.getsize(S + "/dashboard_data.json") // 1024, "KB; curiosidades:", len(cur))
