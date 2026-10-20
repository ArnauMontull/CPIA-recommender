import os, numpy as np, torch, yaml
from scipy.optimize import nnls
from src.datos import cargar_features
from src.modelo import DosTorres, Bolsa
from src.preprocesamiento.comun import DATA, PROCESADOS, RAIZ
cfg = yaml.safe_load((RAIZ/"config.yaml").read_text(encoding="utf-8")); base = DATA/cfg["letterboxd"]["carpeta"]
m = DosTorres.desde_config(cargar_features(PROCESADOS/"completo"), cfg).eval()
m.load_state_dict(torch.load("modelos/completo_tau0.05_s42/mejor.pt", map_location="cpu", weights_only=True)["modelo"])
b = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw); al = os.environ["PERSONAS"].split(","); nom = [a.replace("_api", "").capitalize() for a in al]; us = []
with torch.no_grad():
    V = m.vectores_items()
    for a in al:
        ld = lambda n, a=a: np.load(base/a/n)
        us.append(m.vectores_usuarios(V, torch.tensor([0]), b(ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), b(ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")))[0].numpy().astype(np.float64))
U = np.array(us); P = len(U)
print(f"{'':8s} {'dist. subespacio':>16s} {'ángulo':>8s} {'% de u fuera':>12s} | {'dist. cono (coef>=0)':>20s} {'ángulo cono':>11s} | ángulo mínimo a otra persona")
for i in range(P):
    O = np.delete(U, i, 0); Q, _ = np.linalg.qr(O.T)            # base ortonormal del subespacio de los 6 restantes
    res = U[i] - Q @ (Q.T @ U[i]); d = np.linalg.norm(res)       # |u|=1  => d = sin(ángulo con el subespacio)
    c, rn = nnls(O.T, U[i])                                      # cono: combinaciones con coeficientes >= 0
    dc = rn; amin = min(np.degrees(np.arccos(np.clip(U[i] @ U[j], -1, 1))) for j in range(P) if j != i)
    print(f"{nom[i]:8s} {d:16.3f} {np.degrees(np.arcsin(min(d,1))):7.1f}° {100*d*d:11.1f}% | {dc:20.3f} {np.degrees(np.arcsin(min(dc,1))):10.1f}° | {amin:6.1f}°")
k = P - 1                                                         # coeficientes del cono para la última persona de PERSONAS
print(f"\ncoeficientes del cono para {nom[k]}:", dict(zip(nom[:k], np.round(nnls(U[:k].T, U[k])[0], 2))))
