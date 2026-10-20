"""Ensambla todos los resultados de `reports/` y genera el informe en LaTeX (PDF) y el informe interactivo (HTML autocontenido).

Entradas (las que falten se omiten y el informe lo dice):
  reports/resultados_completo.md            semilla de referencia (42): popularidad, EASE y Two-Tower
  reports/resultados_completo_s<N>.md       otras semillas (solo Two-Tower)
  reports/semillas/s<N>_log.json            curvas de entrenamiento por semilla
  reports/bootstrap_<etiqueta>.json         IC por bootstrap de usuarios
  reports/resultados_dev_frio.json          experimento de arranque en frío (G6)
  reports/watchlist_<perfil>_<etiqueta>.json  recuentos de la watchlist (sin títulos)
  reports/comprobaciones_<etiqueta>.json    comprobaciones sobre datos reales + latencias
  reports/pruebas.json                      {"pasados": N, "saltados": M, "fallos": K}

Salida en ENTREGAS/E3_RESULTADOS/: datos.json, generado.tex, informe_resultados.tex (+ .pdf con --compilar) e informe_interactivo.html.

Uso: python -m src.informe_final [--compilar]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .preprocesamiento.comun import INFORMES, RAIZ
from .glosario import GLOSARIO, GRUPOS
from .resumen_semillas import leer_tablas

SALIDA = RAIZ / "ENTREGAS" / "E3_RESULTADOS"
SECCIONES = ["Val: positivo (C2)", "Val: rating ≥ 4 (C7)", "Test: positivo (C2)", "Test: rating ≥ 4 (C7)"]
METRICAS = ["Recall@10", "Recall@20", "Recall@50", "Recall@200", "NDCG@10", "NDCG@20", "NDCG@50", "NDCG@200", "Cobertura@20", "Novedad@20"]
PRINCIPALES = ["Recall@10", "Recall@50", "Recall@200", "NDCG@20", "Cobertura@20"]


# ---------------------------------------------------------------------------------------------- ensamblado
def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def ensamblar(rep: Path = INFORMES) -> dict:
    informes = {}
    ref = rep / "resultados_completo.md"
    if ref.exists():
        informes["42"] = leer_tablas(ref.read_text(encoding="utf-8"))
    for p in sorted(rep.glob("resultados_completo_s*.md")):
        informes[re.search(r"_s(\d+)\.md$", p.name).group(1)] = leer_tablas(p.read_text(encoding="utf-8"))
    semillas = sorted(informes, key=int)
    metodos: dict = {}
    for sec in SECCIONES:
        metodos[sec] = {}
        for m in METRICAS:
            tt = [informes[s][(sec, m)][-1] for s in semillas if (sec, m) in informes[s]]
            if not tt:
                continue
            fila = {"Two-Tower": {"media": float(np.mean(tt)), "sd": float(np.std(tt, ddof=1)) if len(tt) > 1 else None,
                                  "semillas": dict(zip([s for s in semillas if (sec, m) in informes[s]], tt))}}
            base = informes.get("42", {}).get((sec, m))
            if base and len(base) == 3:
                fila["Popularidad"], fila["EASE"] = base[0], base[1]
            elif base:
                fila["Popularidad"] = base[0]
            metodos[sec][m] = fila
    curvas = {}
    for s in semillas:
        log = _json(rep / "semillas" / f"s{s}_log.json")
        if log:
            curvas[s] = [{"epoca": x["epoca"], "perdida": x["perdida"], "recall200": x.get("recall@200/pos"),
                          "minutos": round(x["segundos"] / 60, 1)} for x in log]
    comprob = {p.stem.replace("comprobaciones_", ""): _json(p) for p in sorted(rep.glob("comprobaciones_*.json"))}
    return {
        "generado": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "semillas": semillas, "metodos": metodos, "curvas": curvas,
        "bootstrap": _json(rep / "bootstrap_s42.json"),
        "frio": _json(rep / "resultados_dev_frio.json"),
        "watchlist": {p.stem.replace("watchlist_", ""): _json(p) for p in sorted(rep.glob("watchlist_perfil*_*.json"))},
        "comprobaciones": comprob,
        "pruebas": _json(rep / "pruebas.json"),
        "glosario": [list(x) for x in GLOSARIO], "grupos": GRUPOS,
    }


# ---------------------------------------------------------------------------------------------- LaTeX
def tex(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("≥", r"$\geq$").replace("−", "$-$").replace("±", r"$\pm$").replace("→", r"$\to$"))


_GRIEGO = {"τ": r"$\tau$", "λ": r"$\lambda$", "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "×": r"$\times$", "−": "$-$", "±": r"$\pm$",
           "→": r"$\to$", "·": r"$\cdot$", "₂": r"\textsubscript{2}", "∞": r"$\infty$"}


def tex_gl(s: str) -> str:
    """Escapa un texto del glosario y traduce sus símbolos Unicode a LaTeX."""
    s = s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    s = s.replace("10⁻⁴", "$10^{-4}$").replace("10⁻³", "$10^{-3}$").replace("10⁻⁶", "$10^{-6}$")
    for k, v in _GRIEGO.items():
        s = s.replace(k, v)
    return s.replace("/", "/\\allowbreak{}")


def glosario_tex() -> str:
    out = ["\\section{Glosario}\nCada métrica, variable y término técnico del informe, tal como se usa en este proyecto. Ordenado por temas.\n"]
    for g in GRUPOS:
        out.append("\\subsection*{" + tex(g) + "}\n\\begin{description}\\setlength{\\itemsep}{2pt}")
        out += ["\\item[" + tex_gl(t) + "] " + tex_gl(d) for gr, t, d in GLOSARIO if gr == g]
        out.append("\\end{description}\n")
    return "\n".join(out)



def f4(x) -> str:
    return "--" if x is None else f"{x:.4f}"


def celda_tt(v: dict) -> str:
    return f"{v['media']:.4f}" + (rf" $\pm$ {v['sd']:.4f}" if v.get("sd") is not None else "")


def tabla_metodos(datos: dict, sec: str, titulo: str, etiqueta: str) -> str:
    filas = []
    for m in METRICAS:
        v = datos["metodos"].get(sec, {}).get(m)
        if not v:
            continue
        mejor = max([x for x in (v.get("Popularidad"), v.get("EASE"), v["Two-Tower"]["media"]) if x is not None])
        marca = lambda x, t: (r"\textbf{" + t + "}") if x is not None and abs(x - mejor) < 1e-12 else t
        filas.append(f"{m} & {f4(v.get('Popularidad'))} & {f4(v.get('EASE'))} & "
                     + marca(v["Two-Tower"]["media"], celda_tt(v["Two-Tower"])) + r" \\")
    if not filas:
        return ""
    return ("\\begin{table}[H]\\centering\\small\n\\begin{tabular}{lrrr}\\toprule\n"
            "Métrica & Popularidad & EASE & Two-Tower (media $\\pm$ desv.) \\\\\\midrule\n" + "\n".join(filas)
            + f"\n\\bottomrule\\end{{tabular}}\n\\caption{{{titulo}}}\\label{{{etiqueta}}}\n\\end{{table}}\n")


def grafica_curvas(datos: dict) -> str:
    if not datos["curvas"]:
        return ""
    colores = ["blue!70!black", "red!70!black", "green!50!black", "orange!90!black", "violet!80!black"]
    series = []
    for i, (s, c) in enumerate(sorted(datos["curvas"].items(), key=lambda kv: int(kv[0]))):
        pts = " ".join(f"({x['epoca']},{x['recall200']})" for x in c if x["recall200"] is not None)
        series.append(f"\\addplot[color={colores[i % 5]},mark=*,mark size=1.2pt] coordinates {{{pts}}};\n\\addlegendentry{{semilla {s}}}")
    return ("\\begin{figure}[H]\\centering\\begin{tikzpicture}\\begin{axis}[width=0.85\\linewidth,height=6.5cm,xlabel={Época},"
            "ylabel={Recall@200 (validación)},grid=both,legend pos=south east,legend style={font=\\small}]\n"
            + "\n".join(series) + "\n\\end{axis}\\end{tikzpicture}\n"
            "\\caption{Recall@200 en validación por época, una curva por semilla.}\\label{fig:curvas}\\end{figure}\n")


def grafica_barras(datos: dict) -> str:
    sec = "Test: positivo (C2)"
    ms = [m for m in ("Recall@10", "Recall@50", "Recall@200", "NDCG@20") if m in datos["metodos"].get(sec, {})]
    if not ms:
        return ""
    def coords(clave):
        return " ".join(f"({m},{datos['metodos'][sec][m][clave] if clave != 'Two-Tower' else datos['metodos'][sec][m]['Two-Tower']['media']:.4f})"
                        for m in ms if clave in datos["metodos"][sec][m])
    tiene_ease = any("EASE" in datos["metodos"][sec][m] for m in ms)
    plots = [("Popularidad", "gray!70"), ("EASE", "orange!80"), ("Two-Tower", "blue!70!black")]
    cuerpo = "\n".join(f"\\addplot[fill={c},draw=none] coordinates {{{coords(n)}}};\\addlegendentry{{{n}}}" for n, c in plots if n != "EASE" or tiene_ease)
    return ("\\begin{figure}[H]\\centering\\begin{tikzpicture}\\begin{axis}[ybar,width=0.9\\linewidth,height=6.5cm,symbolic x coords={"
            + ",".join(ms) + "},xtick=data,ymin=0,bar width=9pt,legend pos=north west,legend style={font=\\small},grid=major,ylabel={valor}]\n"
            + cuerpo + "\n\\end{axis}\\end{tikzpicture}\n\\caption{Test, relevante = positivo: tres métodos en las métricas principales.}\\label{fig:barras}\\end{figure}\n")


def tabla_semillas(datos: dict) -> str:
    sec = "Test: positivo (C2)"
    if len(datos["semillas"]) < 1 or "Recall@200" not in datos["metodos"].get(sec, {}):
        return ""
    ms = [m for m in ("Recall@10", "Recall@200", "NDCG@20", "Cobertura@20") if m in datos["metodos"][sec]]
    cab = " & ".join(rf"\texttt{{{m}}}" for m in ms)
    filas = []
    for s in datos["semillas"]:
        vs = [datos["metodos"][sec][m]["Two-Tower"]["semillas"].get(s) for m in ms]
        filas.append(f"{s} & " + " & ".join(f4(v) for v in vs) + r" \\")
    return ("\\begin{table}[H]\\centering\\small\n\\begin{tabular}{l" + "r" * len(ms) + "}\\toprule\nSemilla & " + cab + " \\\\\\midrule\n" + "\n".join(filas)
            + "\n\\bottomrule\\end{tabular}\n\\caption{Two-Tower por semilla en test (relevante = positivo).}\\label{tab:semillas}\n\\end{table}\n")


def tabla_bootstrap(datos: dict) -> str:
    b = datos["bootstrap"]
    if not b:
        return "No se ha calculado el bootstrap de usuarios en esta ejecución.\n"
    filas = []
    for clave, v in b["diferencias"].items():
        comp, met = clave.split(" | ")
        excl = "sí" if v["ic_inf"] > 0 or v["ic_sup"] < 0 else "no"
        filas.append(f"{tex(comp)} & {met} & {v['dif']:+.4f} & [{v['ic_inf']:+.4f}, {v['ic_sup']:+.4f}] & {excl} \\\\")
    return (f"Bootstrap sobre {b['n_usuarios']} usuarios de test (2000 remuestreos, misma lista de usuarios para todos los métodos).\n"
            "\\begin{table}[H]\\centering\\small\n\\begin{tabular}{llrcc}\\toprule\nComparación & Métrica & Diferencia & IC 95\\,\\% & ¿Excluye el 0? \\\\\\midrule\n"
            + "\n".join(filas) + "\n\\bottomrule\\end{tabular}\n\\caption{Diferencias Two-Tower menos baseline con intervalo de confianza por bootstrap.}\\end{table}\n")


def tabla_frio(datos: dict) -> str:
    f = datos["frio"]
    if not f:
        return "El experimento de arranque en frío no se ha ejecutado en esta versión del informe.\n"
    res = f.get("test") or f
    filas = []
    for m, clave in (("Recall@10", "recall@10/pos"), ("Recall@50", "recall@50/pos"), ("Recall@200", "recall@200/pos"), ("NDCG@20", "ndcg@20/pos")):
        g = lambda met: f4(res.get(met, {}).get(clave))
        filas.append(f"{m} & {g('pop')} & {g('ease')} & {g('modelo')} \\\\")
    return ("\\begin{table}[H]\\centering\\small\n\\begin{tabular}{lrrr}\\toprule\nMétrica & Popularidad & EASE & Two-Tower \\\\\\midrule\n" + "\n".join(filas)
            + "\n\\bottomrule\\end{tabular}\n\\caption{Arranque en frío (10\\,\\% de usuarios fuera del entrenamiento; se les da el 80\\,\\% de su historia y se predice el 20\\,\\% final). "
              "Muestra \\texttt{dev}.}\\end{table}\n")


def tabla_watchlist(datos: dict) -> str:
    if not datos["watchlist"]:
        return "No se ha evaluado la watchlist en esta ejecución.\n"
    filas = []
    for clave, w in datos["watchlist"].items():
        tt, pop = w["Two-Tower"], w["Popularidad"]
        for k in (10, 50, 200):
            filas.append(f"{tex(clave)} & {k} & {tt[f'hits@{k}']} & {pop[f'hits@{k}']} & {tt[f'lift@{k}']:.1f}$\\times$ & {pop[f'lift@{k}']:.1f}$\\times$ \\\\")
    return ("\\begin{table}[H]\\centering\\small\n\\begin{tabular}{lrrrrr}\\toprule\nPerfil / semilla & $k$ & Aciertos TT & Aciertos pop. & Lift TT & Lift pop. \\\\\\midrule\n"
            + "\n".join(filas) + "\n\\bottomrule\\end{tabular}\n\\caption{Aciertos del top-$k$ en la watchlist del usuario (que el modelo nunca ve). "
            "Lift = precisión dividida por la tasa de aciertos del azar.}\\end{table}\n")


def tabla_comprobaciones(datos: dict) -> str:
    c = datos["comprobaciones"]
    if not c:
        return "No se han ejecutado las comprobaciones sobre datos reales.\n"
    nombres = sorted({k for v in c.values() for k in v["comprobaciones"]})
    filas = []
    for n in nombres:
        celdas = ["OK" if c[e]["comprobaciones"].get(n, {}).get("ok") else "FALLA" for e in sorted(c)]
        filas.append(f"{tex(n.replace('_', ' '))} & " + " & ".join(celdas) + r" \\")
    cab = " & ".join(tex(e) for e in sorted(c))
    return ("\\begin{table}[H]\\centering\\small\n\\begin{tabular}{l" + "c" * len(c) + "}\\toprule\nComprobación & " + cab + " \\\\\\midrule\n" + "\n".join(filas)
            + "\n\\bottomrule\\end{tabular}\n\\caption{Comprobaciones sobre datos reales, por checkpoint.}\\end{table}\n")


def frases_resumen(datos: dict) -> list[str]:
    out = []
    sec = "Test: positivo (C2)"
    m = datos["metodos"].get(sec, {})
    if m:
        gana = [k for k, v in m.items() if "EASE" in v and k not in ("Novedad@20",) and v["Two-Tower"]["media"] > v["EASE"]]
        tot = [k for k, v in m.items() if "EASE" in v and k not in ("Novedad@20",)]
        out.append(f"En test (relevante = positivo) el Two-Tower supera a EASE en {len(gana)} de {len(tot)} métricas, promediando {len(datos['semillas'])} semilla(s).")
        r = m.get("Recall@200")
        if r and "EASE" in r:
            sd = f" ± {r['Two-Tower']['sd']:.4f}" if r["Two-Tower"]["sd"] is not None else ""
            out.append(f"Recall@200: Two-Tower {r['Two-Tower']['media']:.4f}{sd}, EASE {r['EASE']:.4f}, popularidad {r['Popularidad']:.4f}.")
    b = datos.get("bootstrap")
    if b:
        ee = {k: v for k, v in b["diferencias"].items() if k.startswith("Two-Tower − EASE")}
        sig = [k for k, v in ee.items() if v["ic_inf"] > 0]
        out.append(f"Bootstrap de usuarios (semilla 42): la diferencia Two-Tower − EASE es positiva con el IC del 95 % por encima de 0 en {len(sig)} de {len(ee)} comparaciones.")
    f = datos.get("frio")
    if f:
        t = (f.get("test") or f)
        mets = ("recall@10/pos", "recall@50/pos", "recall@200/pos", "ndcg@20/pos")
        ease_gana = sum(1 for k in mets if t["ease"][k] > t["modelo"][k])
        out.append(f"Arranque en frío (muestra dev, modelo pequeño de 5 épocas cortas): EASE supera al Two-Tower en {ease_gana} de {len(mets)} métricas; "
                   "no es el resultado del conjunto completo y no confirma ventaja en frío.")
    w = datos.get("watchlist") or {}
    if w:
        perfiles = sorted({k.split("_")[0] for k in w})
        partes = []
        for pf in perfiles:
            ws = [v for k, v in w.items() if k.startswith(pf + "_")]
            m50 = sum(1 for v in ws if v["Two-Tower"]["hits@50"] > v["Popularidad"]["hits@50"])
            m200 = sum(1 for v in ws if v["Two-Tower"]["hits@200"] > v["Popularidad"]["hits@200"])
            partes.append(f"{pf}: el Two-Tower acierta más que popularidad en el top-50 en {m50} de {len(ws)} semillas y en el top-200 en {m200} de {len(ws)}")
        out.append("Watchlist (señal débil, solo dos perfiles) — " + "; ".join(partes) + ".")
    if datos["pruebas"]:
        p = datos["pruebas"]
        out.append(f"Suite de pruebas: {p['pasados']} pasadas, {p['saltados']} saltadas, {p['fallos']} fallos.")
    return out


PLANTILLA = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[spanish,es-nodecimaldot,es-noshorthands]{babel}
\usepackage{amsmath,amssymb}
\usepackage[left=2.1cm,right=2.1cm,top=2.2cm,bottom=2.2cm]{geometry}
\usepackage{booktabs,float,graphicx,caption,microtype,xcolor}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\title{Resultados del recomendador Two-Tower\\\large CPIA --- Motor de recomendación cinematográfica}
\author{Arnau Montull \and Uriel Martínez}
\date{@@FECHA@@}
\begin{document}
\maketitle
\input{generado.tex}
\end{document}
"""


def cuerpo_tex(datos: dict) -> str:
    n = len(datos["semillas"])
    p = []
    p.append("\\section{Resumen}\n" + "\n\n".join(tex(x) for x in frases_resumen(datos)) + "\n")
    p.append("""\\section{Datos y protocolo}
Recomendador de películas \\emph{Two-Tower} entrenado con MovieLens~32M (200\\,948 usuarios, 23\\,335 películas, 31\\,7~M de valoraciones) y
metadatos de TMDB. El reparto es \\textbf{por usuario y por tiempo}: el 80\\,\\% más antiguo de cada usuario es entrenamiento, el 10\\,\\% siguiente validación y el último 10\\,\\% test.
Todo lo que se aprende (media de cada usuario, popularidad, vocabularios) sale solo de entrenamiento. En validación el historial de entrada es train; en test, train y validación.
En ambas particiones se excluye del ranking todo lo ya valorado. Las métricas se calculan sobre el catálogo completo (sin muestrear negativos), con una submuestra fija de
20\\,000 usuarios elegibles, la misma para los tres métodos. «Relevante» tiene dos definiciones: \\emph{positivo} (valoración $\\geq \\min(4, \\bar r_u)$) y valoración $\\geq 4$.
Baselines: popularidad (lo más visto en train) y EASE (Steck, 2019, $\\lambda=500$) sobre las 10\\,000 películas más populares.
""")
    p.append(f"\\section{{Resultados principales}}\nEl Two-Tower se ha entrenado con {n} semilla(s): {', '.join(datos['semillas'])}. Cada celda del Two-Tower es la media $\\pm$ desviación típica muestral entre semillas; en negrita el mejor método.\n")
    p.append(tabla_metodos(datos, "Test: positivo (C2)", "Test, relevante = positivo.", "tab:test_pos"))
    p.append(tabla_metodos(datos, "Test: rating ≥ 4 (C7)", "Test, relevante = valoración $\\geq 4$.", "tab:test_r4"))
    p.append(grafica_barras(datos))
    p.append(tabla_metodos(datos, "Val: positivo (C2)", "Validación, relevante = positivo.", "tab:val_pos"))
    p.append("\\subsection{Variabilidad entre semillas}\n" + tabla_semillas(datos))
    p.append("\\subsection{Entrenamiento}\n" + grafica_curvas(datos))
    p.append("\\section{Significación estadística}\n" + tabla_bootstrap(datos))
    p.append("\\section{Arranque en frío de usuarios}\n" + tabla_frio(datos))
    p.append("\\section{Validación con la watchlist}\nLa watchlist son películas que el usuario marcó como «quiero ver»; el modelo no la usa como entrada. "
             "Es una señal débil (intereses, no gustos) y tiende a contener clásicos populares, por eso se compara con popularidad. "
             "Se excluyen las ya vistas, y solo se cuentan las emparejadas con el catálogo.\n" + tabla_watchlist(datos))
    p.append("\\section{Comprobaciones y pruebas}\n")
    if datos["pruebas"]:
        pr = datos["pruebas"]
        p.append(f"Suite de pruebas automáticas: {pr['pasados']} pasadas, {pr['saltados']} saltadas y {pr['fallos']} fallos. "
                 "Incluye invariantes sobre las salidas reales del conjunto completo (sin fugas temporales, medias solo de entrenamiento, logQ exacto, historiales sin test).\n")
    p.append(tabla_comprobaciones(datos))
    lat = next((v["latencia"] for v in datos["comprobaciones"].values() if v and "latencia" in v), None)
    if lat:
        p.append(f"\\subsection{{Latencia de inferencia (CPU, Python)}}\nUn usuario (torre de usuario, puntuar las 23\\,335 películas y top-200 exacto): {lat['ms_por_usuario_solo']}~ms. "
                 f"En lotes de 512: {lat['ms_por_usuario_en_lote_512']}~ms por usuario. Calcular la torre de ítem completa: {lat['ms_torre_de_items_completa']}~ms (una vez por actualización del modelo).\n")
    p.append("""\\section{Limitaciones}
\\begin{itemize}
\\item EASE se calcula sobre las 10\\,000 películas más populares (la inversión densa de las 23\\,335 necesita unos 9~GB); con el catálogo completo podría mejorar algo.
\\item Las métricas se calculan sobre una submuestra fija de 20\\,000 usuarios, no sobre todos los elegibles.
\\item El mismo hiperparámetro $\\tau=0{,}05$ en todas las semillas: no se ha ajustado por semilla ni se ha barrido en el conjunto completo (solo en \\texttt{dev}).
\\item La validación con la watchlist y las demos con perfiles de Letterboxd se hacen con dos usuarios: son ilustrativas, no estadísticas.
\\end{itemize}
\\section{Reproducibilidad}
Código y resultados en el repositorio \\texttt{CPIA-recommender}. Entrenamiento: \\texttt{python -m src.entrenar --muestra completo --device cuda --tau 0.05 --semilla N}. Evaluación: \\texttt{python -m src.evaluar}.
Comprobaciones: \\texttt{python -m src.comprobaciones}. Este informe: \\texttt{python -m src.informe\\_final --compilar}.
""")
    p.append(glosario_tex())
    return "\n".join(x for x in p if x)


# ---------------------------------------------------------------------------------------------- HTML interactivo
PLANTILLA_HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Informe interactivo del recomendador CPIA</title>
<style>
:root{--bg:#f7f7f4;--fg:#1c1f26;--mut:#5d6470;--card:#ffffff;--line:#d9dbe0;--a:#2f5fd0;--b:#d98324;--c:#8d939e;--ok:#177a45;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--fg:#e8eaee;--mut:#9aa1ad;--card:#1d2027;--line:#333844;--a:#7aa2ff;--b:#f0a54a;--c:#8f96a3;--ok:#4cc38a;--bad:#ff7b72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px max(16px,env(safe-area-inset-left))}
h1{font:600 28px/1.2 Georgia,serif;margin:.2em 0}h2{font:600 19px/1.3 system-ui;margin:1.6em 0 .5em}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.tile .n{font:600 28px Georgia,serif}.tile .l{color:var(--mut);font-size:13px}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}label{font-size:14px}select,button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:5px 8px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.sc{overflow-x:auto}svg{width:100%;height:auto;display:block}svg text{fill:var(--mut);font:12px system-ui}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:13px;display:none;z-index:9}
.leg span{display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:13px}.leg i{width:11px;height:11px;border-radius:2px;display:inline-block}
</style></head><body><main>
<h1>Recomendador Two-Tower: informe interactivo</h1>
<div class="mut" id="sub"></div>
<div class="tiles" id="tiles"></div>
<div class="card" id="concl"></div>

<h2>Comparación de métodos</h2>
<div class="card"><div class="ctl">
<label>Partición <select id="selPart"><option>Test</option><option>Val</option></select></label>
<label>Relevante <select id="selDef"><option value="positivo (C2)">positivo (C2)</option><option value="rating ≥ 4 (C7)">rating ≥ 4 (C7)</option></select></label>
<label>Métrica <select id="selFam"><option value="Recall">Recall@k</option><option value="NDCG">NDCG@k</option></select></label></div>
<div class="leg" id="legM"></div><div class="sc"><svg id="barras" viewBox="0 0 820 300"></svg></div>
<div class="sc"><table id="tabM"></table></div></div>

<h2>Variabilidad entre semillas</h2>
<div class="card"><div class="ctl"><label>Métrica <select id="selSem"></select></label></div>
<div class="sc"><svg id="semillas" viewBox="0 0 820 220"></svg></div><div class="sc"><table id="tabS"></table></div></div>

<h2>Curvas de entrenamiento</h2>
<div class="card"><div class="ctl"><label>Mostrar <select id="selCur"><option value="recall200">Recall@200 (validación)</option><option value="perdida">Pérdida</option></select></label></div>
<div class="leg" id="legC"></div><div class="sc"><svg id="curvas" viewBox="0 0 820 300"></svg></div></div>

<h2>Significación (bootstrap de usuarios)</h2><div class="card" id="boot"></div>
<h2>Arranque en frío</h2><div class="card" id="frio"></div>
<h2>Watchlist</h2><div class="card" id="wl"></div>
<h2>Comprobaciones</h2><div class="card" id="comp"></div>
<h2>Glosario</h2><div class="card"><div class="ctl"><label>Buscar <input id="busca" type="search" placeholder="métrica, variable o término…" style="min-width:260px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px"></label></div><div id="glo"></div></div>
<div class="tip" id="tip"></div>
<p class="mut" id="pie"></p>
</main>
<script id="datos" type="application/json">@@DATOS@@</script>
<script>
const D=JSON.parse(document.getElementById('datos').textContent);
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',tip=$('tip');
const COL={'Popularidad':'var(--c)','EASE':'var(--b)','Two-Tower':'var(--a)'},SC=['#2f5fd0','#d98324','#177a45','#b3261e','#7d4cc0'];
const f4=x=>x==null?'—':x.toFixed(4);
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function txt(p,x,y,s,a){const e=el('text',Object.assign({x,y},a||{}),p);e.textContent=s;return e}
function showTip(ev,html){tip.innerHTML=html;tip.style.display='block';tip.style.left=Math.min(ev.clientX+12,innerWidth-190)+'px';tip.style.top=(ev.clientY+12)+'px'}
function hideTip(){tip.style.display='none'}
$('sub').textContent='Generado el '+D.generado+' · '+D.semillas.length+' semilla(s): '+D.semillas.join(', ')+' · MovieLens 32M, catálogo de 23 335 películas';
const TS='Test: positivo (C2)',met=(s,m)=>(D.metodos[s]||{})[m];
// tiles
(function(){const r=met(TS,'Recall@200'),n=met(TS,'NDCG@20'),t=[];
 if(r){t.push(['Recall@200 (test) Two-Tower',f4(r['Two-Tower'].media)+(r['Two-Tower'].sd!=null?' ± '+r['Two-Tower'].sd.toFixed(4):''),'media entre semillas']);
  if(r.EASE!=null)t.push(['Recall@200 EASE',f4(r.EASE),'baseline fuerte, 10 000 películas']);t.push(['Recall@200 popularidad',f4(r['Popularidad']),'lo más visto en train'])}
 if(D.pruebas)t.push(['Pruebas automáticas',D.pruebas.pasados+' ok',D.pruebas.saltados+' saltadas · '+D.pruebas.fallos+' fallos']);
 $('tiles').innerHTML=t.map(x=>'<div class="card tile"><div class="l">'+x[0]+'</div><div class="n">'+x[1]+'</div><div class="l">'+x[2]+'</div></div>').join('')})();
$('concl').innerHTML='<b>Conclusiones</b><ul style="margin:.4em 0 0 1.1em;padding:0">'+(D.frases||[]).map(x=>'<li>'+x+'</li>').join('')+'</ul>';
// barras agrupadas
function barras(){const sec=$('selPart').value+': '+$('selDef').value,fam=$('selFam').value,ks=[10,20,50,200];
 const ms=ks.map(k=>fam+'@'+k).filter(m=>met(sec,m));const svg=$('barras');svg.innerHTML='';
 const L=50,R=800,T=16,B=36,H=300,names=['Popularidad','EASE','Two-Tower'].filter(n=>ms.some(m=>met(sec,m)[n]!=null));
 let mx=0;ms.forEach(m=>names.forEach(n=>{const v=met(sec,m)[n];const x=n=='Two-Tower'?v.media+(v.sd||0):v;if(x>mx)mx=x}));mx=Math.max(mx*1.08,1e-9);
 const y=v=>H-B-(v/mx)*(H-B-T);
 for(let i=0;i<=4;i++){const v=mx*i/4;el('line',{x1:L,x2:R,y1:y(v),y2:y(v),stroke:'var(--line)'},svg);txt(svg,L-6,y(v)+4,v.toFixed(2),{'text-anchor':'end'})}
 const gw=(R-L)/ms.length,bw=Math.min(34,gw/(names.length+1));
 ms.forEach((m,i)=>{const cx=L+gw*(i+.5);txt(svg,cx,H-B+18,m,{'text-anchor':'middle'});
  names.forEach((n,j)=>{const raw=met(sec,m)[n];if(raw==null)return;const v=n=='Two-Tower'?raw.media:raw,x=cx+(j-(names.length-1)/2)*(bw+4)-bw/2;
   const r=el('rect',{x,y:y(v),width:bw,height:H-B-y(v),fill:COL[n],rx:3},svg);
   if(n=='Two-Tower'&&raw.sd!=null){el('line',{x1:x+bw/2,x2:x+bw/2,y1:y(v-raw.sd),y2:y(v+raw.sd),stroke:'var(--fg)','stroke-width':1.5},svg)}
   r.onmousemove=ev=>showTip(ev,'<b>'+n+'</b> · '+m+'<br>'+f4(v)+(n=='Two-Tower'&&raw.sd!=null?' ± '+raw.sd.toFixed(4):''));r.onmouseleave=hideTip})});
 $('legM').innerHTML=names.map(n=>'<span><i style="background:'+COL[n]+'"></i>'+n+'</span>').join('');
 const rows=METRICAS.filter(m=>met(sec,m));let h='<tr><th>Métrica</th><th>Popularidad</th><th>EASE</th><th>Two-Tower</th></tr>';
 rows.forEach(m=>{const v=met(sec,m),tt=v['Two-Tower'],mx=Math.max(tt.media,v['Popularidad']==null?-1:v['Popularidad'],v.EASE==null?-1:v.EASE),B=(x,t)=>x!=null&&Math.abs(x-mx)<1e-12?'<b>'+t+'</b>':t;h+='<tr><td>'+m+'</td><td>'+B(v['Popularidad'],f4(v['Popularidad']))+'</td><td>'+B(v.EASE,f4(v.EASE))+'</td><td>'+B(tt.media,f4(tt.media))+(tt.sd!=null?' ± '+tt.sd.toFixed(4):'')+'</td></tr>'});$('tabM').innerHTML=h}
const METRICAS=@@METRICAS@@;['selPart','selDef','selFam'].forEach(i=>$(i).onchange=barras);barras();
// semillas
$('selSem').innerHTML=METRICAS.filter(m=>met(TS,m)).map(m=>'<option>'+m+'</option>').join('');$('selSem').value='Recall@200';
function sem(){const m=$('selSem').value,v=met(TS,m),svg=$('semillas');svg.innerHTML='';if(!v){return}
 const ss=Object.entries(v['Two-Tower'].semillas),vals=ss.map(x=>x[1]);if(v.EASE!=null)vals.push(v.EASE);if(v['Popularidad']!=null)vals.push(v['Popularidad']);
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo||hi*.1)*.15;lo-=pad;hi+=pad;const L=60,R=800,T=18,B=34,H=220,x=v=>L+(v-lo)/(hi-lo)*(R-L);
 for(let i=0;i<=5;i++){const t=lo+(hi-lo)*i/5;el('line',{x1:x(t),x2:x(t),y1:T,y2:H-B,stroke:'var(--line)'},svg);txt(svg,x(t),H-B+16,t.toFixed(3),{'text-anchor':'middle'})}
 const ref=(val,col,name,dy)=>{if(val==null)return;el('line',{x1:x(val),x2:x(val),y1:T,y2:H-B,stroke:col,'stroke-width':2,'stroke-dasharray':'5 4'},svg);txt(svg,x(val)+4,T+dy,name,{fill:col})};
 ref(v.EASE,'var(--b)','EASE',12);ref(v['Popularidad'],'var(--c)','Popularidad',26);
 const mean=v['Two-Tower'].media;el('line',{x1:x(mean),x2:x(mean),y1:T,y2:H-B,stroke:'var(--a)','stroke-width':2},svg);txt(svg,x(mean)+4,H-B-6,'media',{fill:'var(--a)'});
 ss.forEach(([s,val],i)=>{const cy=T+20+i*((H-B-T-30)/Math.max(ss.length,1));const c=el('circle',{cx:x(val),cy,r:6,fill:'var(--a)'},svg);txt(svg,x(val)+10,cy+4,'semilla '+s+' · '+f4(val));
  c.onmousemove=ev=>showTip(ev,'semilla '+s+'<br>'+m+' = '+f4(val));c.onmouseleave=hideTip});
 let h='<tr><th>Semilla</th><th>'+m+'</th></tr>';ss.forEach(([s,val])=>h+='<tr><td>'+s+'</td><td>'+f4(val)+'</td></tr>');
 h+='<tr><td><b>media ± desv.</b></td><td><b>'+f4(mean)+(v['Two-Tower'].sd!=null?' ± '+v['Two-Tower'].sd.toFixed(4):'')+'</b></td></tr>';$('tabS').innerHTML=h}
$('selSem').onchange=sem;sem();
// curvas
function cur(){const k=$('selCur').value,svg=$('curvas');svg.innerHTML='';const ss=Object.keys(D.curvas);if(!ss.length){svg.outerHTML='<p class="mut">Sin curvas de entrenamiento.</p>';return}
 const all=ss.flatMap(s=>D.curvas[s].map(p=>p[k]).filter(v=>v!=null));let lo=Math.min(...all),hi=Math.max(...all);const pad=(hi-lo)*.1||.01;lo-=pad;hi+=pad;
 const maxE=Math.max(...ss.flatMap(s=>D.curvas[s].map(p=>p.epoca))),L=60,R=800,T=16,B=34,H=300,x=e=>L+(e-1)/(Math.max(maxE-1,1))*(R-L),y=v=>H-B-(v-lo)/(hi-lo)*(H-B-T);
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;el('line',{x1:L,x2:R,y1:y(v),y2:y(v),stroke:'var(--line)'},svg);txt(svg,L-6,y(v)+4,v.toFixed(3),{'text-anchor':'end'})}
 for(let e=1;e<=maxE;e++)txt(svg,x(e),H-B+16,e,{'text-anchor':'middle'});txt(svg,(L+R)/2,H-4,'época',{'text-anchor':'middle'});
 ss.forEach((s,i)=>{const pts=D.curvas[s].filter(p=>p[k]!=null);el('polyline',{points:pts.map(p=>x(p.epoca)+','+y(p[k])).join(' '),fill:'none',stroke:SC[i%5],'stroke-width':2,'stroke-linejoin':'round'},svg);
  pts.forEach(p=>{const c=el('circle',{cx:x(p.epoca),cy:y(p[k]),r:3.5,fill:SC[i%5]},svg);c.onmousemove=ev=>showTip(ev,'semilla '+s+' · época '+p.epoca+'<br>'+k+' = '+p[k].toFixed(4)+'<br>'+p.minutos+' min');c.onmouseleave=hideTip})});
 $('legC').innerHTML=ss.map((s,i)=>'<span><i style="background:'+SC[i%5]+'"></i>semilla '+s+'</span>').join('')}
$('selCur').onchange=cur;cur();
// bootstrap
(function(){const b=D.bootstrap;if(!b){$('boot').innerHTML='<span class="mut">No calculado en esta ejecución.</span>';return}
 const e=Object.entries(b.diferencias);let h='<p class="mut">'+b.n_usuarios+' usuarios de test · 2000 remuestreos · IC 95 %. Si el intervalo no contiene el 0, la diferencia no depende de qué usuarios entraron en la muestra.</p><div class="sc"><table><tr><th>Comparación</th><th>Métrica</th><th>Diferencia</th><th>IC 95 %</th><th>¿Excluye 0?</th></tr>';
 e.forEach(([k,v])=>{const [c,m]=k.split(' | ');const ex=v.ic_inf>0||v.ic_sup<0;h+='<tr><td>'+c+'</td><td>'+m+'</td><td>'+(v.dif>=0?'+':'')+v.dif.toFixed(4)+'</td><td>['+v.ic_inf.toFixed(4)+', '+v.ic_sup.toFixed(4)+']</td><td class="'+(ex?'ok':'bad')+'">'+(ex?'sí':'no')+'</td></tr>'});$('boot').innerHTML=h+'</table></div>'})();
// frio
(function(){const f=D.frio;if(!f){$('frio').innerHTML='<span class="mut">No ejecutado en esta versión.</span>';return}const t=f.test||f;
 let h='<p class="mut">10 % de usuarios fuera del entrenamiento; se les da el 80 % de su historia y se predice el 20 % final (muestra dev).</p><div class="sc"><table><tr><th>Métrica</th><th>Popularidad</th><th>EASE</th><th>Two-Tower</th></tr>';
 [['Recall@10','recall@10/pos'],['Recall@50','recall@50/pos'],['Recall@200','recall@200/pos'],['NDCG@20','ndcg@20/pos']].forEach(([n,k])=>{const v=[(t.pop||{})[k],(t.ease||{})[k],(t.modelo||{})[k]],mx=Math.max(...v.filter(x=>x!=null)),c=v.map(x=>x!=null&&x===mx?'<b>'+f4(x)+'</b>':f4(x));h+='<tr><td>'+n+'</td><td>'+c[0]+'</td><td>'+c[1]+'</td><td>'+c[2]+'</td></tr>'});$('frio').innerHTML=h+'</table></div>'})();
// watchlist
(function(){const w=D.watchlist,ks=Object.keys(w||{});if(!ks.length){$('wl').innerHTML='<span class="mut">No evaluada.</span>';return}
 let h='<p class="mut">Aciertos del top-k en la watchlist del usuario (no se usa como entrada). Lift = precisión / tasa del azar.</p><div class="sc"><table><tr><th>Perfil / semilla</th><th>k</th><th>Aciertos Two-Tower</th><th>Aciertos popularidad</th><th>Lift TT</th><th>Lift pop.</th></tr>';
 ks.forEach(c=>[10,50,200].forEach(k=>{const t=w[c]['Two-Tower'],p=w[c]['Popularidad'];h+='<tr><td>'+c+'</td><td>'+k+'</td><td>'+t['hits@'+k]+'</td><td>'+p['hits@'+k]+'</td><td>'+t['lift@'+k].toFixed(1)+'×</td><td>'+p['lift@'+k].toFixed(1)+'×</td></tr>'}));$('wl').innerHTML=h+'</table></div>'})();
// comprobaciones
(function(){const c=D.comprobaciones,ks=Object.keys(c||{});if(!ks.length){$('comp').innerHTML='<span class="mut">No ejecutadas.</span>';return}
 const nombres=[...new Set(ks.flatMap(k=>Object.keys(c[k].comprobaciones)))];let h='<div class="sc"><table><tr><th>Comprobación</th>'+ks.map(k=>'<th>'+k+'</th>').join('')+'</tr>';
 nombres.forEach(n=>h+='<tr><td>'+n.replace(/_/g,' ')+'</td>'+ks.map(k=>{const x=c[k].comprobaciones[n];return '<td class="'+(x&&x.ok?'ok':'bad')+'" title="'+(x?x.detalle.replace(/"/g,'&quot;'):'')+'">'+(x&&x.ok?'OK':'FALLA')+'</td>'}).join('')+'</tr>');
 h+='</table></div>';const l=(c[ks[0]]||{}).latencia;if(l)h+='<p class="mut">Latencia en CPU (Python): '+l.ms_por_usuario_solo+' ms por usuario solo, '+l.ms_por_usuario_en_lote_512+' ms por usuario en lotes de 512, torre de ítem completa '+l.ms_torre_de_items_completa+' ms.</p>';$('comp').innerHTML=h})();
(function(){const g=D.glosario;function pinta(){const q=$('busca').value.trim().toLowerCase();let h='';D.grupos.forEach(gr=>{const it=g.filter(x=>x[0]==gr&&(!q||(x[1]+' '+x[2]).toLowerCase().includes(q)));if(!it.length)return;h+='<h3 style="margin:1em 0 .3em;font-size:15px">'+gr+'</h3><dl style="margin:0">'+it.map(x=>'<dt style="font-weight:600;margin-top:.5em">'+x[1]+'</dt><dd style="margin:0 0 0 1em;color:var(--mut)">'+x[2]+'</dd>').join('')+'</dl>'});$('glo').innerHTML=h||'<span class="mut">Sin resultados.</span>'}
 $('busca').oninput=pinta;pinta()})();
$('pie').textContent='Informe autogenerado por src/informe_final.py a partir de reports/. Los datos están incrustados en este archivo.';
</script></body></html>
"""


def generar(datos: dict, salida: Path = SALIDA, compilar: bool = False) -> dict:
    datos = {**datos, "frases": frases_resumen(datos)}
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "datos.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    (salida / "generado.tex").write_text(cuerpo_tex(datos), encoding="utf-8")
    (salida / "informe_resultados.tex").write_text(PLANTILLA.replace("@@FECHA@@", datos["generado"]), encoding="utf-8")
    html = PLANTILLA_HTML.replace("@@DATOS@@", json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")).replace("@@METRICAS@@", json.dumps(METRICAS))
    (salida / "informe_interactivo.html").write_text(html, encoding="utf-8")
    res = {"html": str(salida / "informe_interactivo.html"), "tex": str(salida / "informe_resultados.tex")}
    if compilar:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "informe_resultados.tex"], cwd=salida, capture_output=True, text=True)
        res["latex_ok"] = r.returncode == 0
        res["latex_log"] = (r.stdout + r.stderr)[-1500:]
        res["pdf"] = str(salida / "informe_resultados.pdf")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compilar", action="store_true")
    ap.add_argument("--reports", type=Path, default=INFORMES)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args(argv)
    r = generar(ensamblar(a.reports), a.salida, a.compilar)
    print(json.dumps({k: v for k, v in r.items() if k != "latex_log"}, indent=1))
    if a.compilar and not r["latex_ok"]:
        print(r["latex_log"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
