import numpy as np, torch, yaml
from src.datos import cargar_features
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
m = DosTorres.desde_config(cargar_features(PROCESADOS/"completo"), cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
b = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw); us = {}
with torch.no_grad():
    V = m.vectores_items()
    for al in tuple(__import__("os").environ["PERSONAS"].split(",")):
        ld = lambda n: np.load(base/al/n)
        us[al] = m.vectores_usuarios(V, torch.tensor([0]), b(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), b(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy()
a = lambda x, y: np.degrees(np.arccos(np.clip(us[x]@us[y], -1, 1)))
nombres = list(us); print()
print("%-8s" % "", *["%8s" % n.split("_")[0] for n in nombres])
for x in nombres: print("%-8s" % x.split("_")[0], *["%8.1f" % a(x, y) for y in nombres])
