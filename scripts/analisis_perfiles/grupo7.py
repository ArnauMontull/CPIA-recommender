import json, os, numpy as np, pandas as pd, torch, yaml
from src.datos import cargar_features
from src.demo_pareja import scores_pareja, combinar
from src.modelo import DosTorres
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
m = DosTorres.desde_config(cargar_features(PROCESADOS/"completo"), cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
with torch.no_grad(): V = m.vectores_items()
al = os.environ["PERSONAS"].split(","); nom = [a.replace("_api", "").capitalize() for a in al]
perfiles = []
for a in al:
    ld = lambda n, a=a: np.load(base/a/n)
    perfiles.append({"A": (ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), "B": (ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")), "valorados": ld("valorados_indices.npy")})
S, excl, N = scores_pareja(m, V, perfiles)
movies = pd.read_parquet(PROCESADOS/"movies.parquet", columns=["movieId", "titulo", "anio"]).set_index("movieId")
mapeo = pd.read_parquet(PROCESADOS/"dev"/"mapeo_peliculas.parquet").set_index("item_idx")
nombre = lambda i: (lambda f: f"{f.titulo} ({'' if pd.isna(f.anio) else int(f.anio)})")(movies.loc[int(mapeo.loc[int(i), 'movieId'])])
print(f"excluidas (vistas por alguien): {len(excl)} de {N}")
for regla in ("minimo", "media"):
    top, c = combinar(S, regla, 12)
    print(f"\n== regla {regla} ==")
    for r, i in enumerate(top, 1):
        print(f"{r:2d}. {nombre(i)} | combinado {c[r-1]:.3f} | " + " ".join(f"{n[:3]} {S[k, i]:.2f}" for k, n in enumerate(nom)))
# películas que están en el top-100 individual de al menos 5 de las 7 personas
tops = [set(np.argsort(-s)[:100].tolist()) for s in S]
from collections import Counter
cn = Counter(i for t in tops for i in t); print("\nen el top-100 de >=5 personas:", [(nombre(i), c) for i, c in cn.most_common(12) if c >= 5])
