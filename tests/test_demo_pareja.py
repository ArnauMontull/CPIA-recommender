"""Reglas para juntar dos personas: intersección de top-k, mínimo y media, con scores a mano."""
import numpy as np

from src.demo_pareja import combinar, interseccion


def test_interseccion_y_reglas_a_mano():
    # 5 películas; la 4 ya la vio alguien (-inf en las dos filas)
    S = np.array([[0.9, 0.8, 0.1, 0.5, -np.inf],
                  [0.1, 0.7, 0.9, 0.6, -np.inf]])
    inter = interseccion(S, ks=(2, 3))
    assert inter[2].tolist() == [1] and sorted(inter[3].tolist()) == [1, 3]       # top-2: {0,1} y {2,1}; top-3: {0,1,3} y {2,1,3}
    top, c = combinar(S, "minimo", 3)
    assert top.tolist() == [1, 3, 0] and np.allclose(c, [0.7, 0.5, 0.1])           # min: 0.1, 0.7, 0.1, 0.5
    top, c = combinar(S, "media", 2)
    assert top.tolist() == [1, 3] and np.allclose(c, [0.75, 0.55])
