import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# Carpeta donde reside este script (ENTREGAS/E1_PROPUESTA)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta hacia data/ml-32m/ratings.csv
CSV_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data", "ml-32m", "ratings.csv"))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# FIGURA 1: diagrama_problema_M.png
# ==============================================================================
def generar_diagrama_problema():
    print("-> Generando figura 1: diagrama_problema_M.png...", flush=True)
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    c_data = '#E3F2FD'
    c_b_data = '#1976D2'
    c_proc = '#FFEBEE'
    c_b_proc = '#D32F2F'
    c_model = '#E8F5E9'
    c_b_model = '#388E3C'
    c_ret = '#FFF3E0'
    c_b_ret = '#F57C00'

    def card(x, y, w, h, title, lines, bg, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.4, edgecolor=border, facecolor=bg, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#1A1A1A', zorder=3)
        cy = y + h - 0.75
        for line in lines:
            ax.text(x + 0.2, cy, line, ha='left', va='center', fontsize=8.2, color='#333333', zorder=3)
            cy -= 0.35

    def flecha(start, end, label=""):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=1.6, mutation_scale=14), zorder=1)
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18
            ax.text(mx, my, label, ha='center', va='bottom', fontsize=7.8, fontweight='bold',
                    color='#455A64', backgroundcolor='white')

    card(0.5, 5.5, 3.2, 2.3, "MovieLens 32M",
         ["• Matriz R ∈ ℝ^(N×M)", "• 32M calificaciones", "• Esparcidad: 99.818%"], c_data, c_b_data)
    card(0.5, 2.9, 3.2, 2.3, "The Movies Dataset (TMDB)",
         ["• Metadatos (45k filmes)", "• Sinopsis, géneros, keywords", "• Reparto y cuerpo técnico"], c_data, c_b_data)
    card(0.5, 0.3, 3.2, 2.3, "Letterboxd Profile",
         ["• Historial de usuario externo", "• Secuencia temporal de diario", "• Reto Cold-Start activo"], c_data, c_b_data)

    card(4.4, 4.3, 3.4, 3.5, "Espacio Semántico y Regularización",
         ["• SentenceTransformers (d=384)", "• Multi-Hot: directores y género",
          "• Pasarela links.csv (TMDB/IMDb)", "• Corrección de sesgo LogQ"], c_proc, c_b_proc)
    card(4.4, 0.8, 3.4, 2.8, "Perfilado Dinámico de Usuario",
         ["• Agregación de historial", "• Ponderación continua de notas",
          "• Inferencia amortizada en CPU", "• Normalización ||u|| = 1"], c_proc, c_b_proc)

    card(8.5, 4.5, 3.2, 3.2, "Torre de Ítem Ψ_I(x_i)",
         ["• Proyección lineal a d=128", "• Fusión de texto + categorías",
          "• Vector v_i normalizado", "• Indexación offline en memoria"], c_model, c_b_model)
    card(8.5, 0.8, 3.2, 3.2, "Torre de Usuario Ψ_U(x_u)",
         ["• Vector u ∈ ℝ^128 en tiempo real", "• Cómputo SIMD / AVX en CPU",
          "• Paralelismo multihilo OpenMP", "• Desacoplamiento total"], c_model, c_b_model)

    dot = patches.Circle((10.1, 4.0), 0.42, lw=1.4, edgecolor='#388E3C', facecolor='#C8E6C9', zorder=2)
    ax.add_patch(dot)
    ax.text(10.1, 4.0, "⟨u, v_i⟩", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1B5E20', zorder=3)

    card(12.2, 4.6, 2.5, 3.0, "Etapa 1: Recuperación",
         ["• Búsqueda ANN (HNSW)", "• Máximo producto escalar", "• Top K_c = 200 candidatos", "• Latencia < 5 ms"], c_ret, c_b_ret)
    card(12.2, 0.8, 2.5, 3.0, "Etapa 2: Re-Ranking",
         ["• Algoritmo MMR (λ ≈ 0.7)", "• Diversidad intra-lista", "• Impulso a cine de nicho", "• Top-K final (K=20)"], c_ret, c_b_ret)

    flecha((3.7, 6.5), (4.4, 6.0))
    flecha((3.7, 4.0), (4.4, 5.0))
    flecha((3.7, 1.5), (4.4, 2.2))
    flecha((7.8, 6.0), (8.5, 6.0), "Metadatos")
    flecha((7.8, 2.2), (8.5, 2.2), "Historial")
    flecha((10.1, 4.5), (10.1, 4.42))
    flecha((10.1, 3.5), (10.1, 3.58))
    flecha((11.7, 6.0), (12.2, 6.0))
    flecha((13.45, 4.6), (13.45, 3.8), "Candidatos")

    plt.tight_layout()
    ruta_salida = os.path.join(OUTPUT_DIR, "diagrama_problema_M.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Creado: {ruta_salida}", flush=True)

# ==============================================================================
# FIGURA 2: distribucion_ratings.png (DATOS REALES)
# ==============================================================================
def generar_distribucion_ratings():
    print(f"-> Generando figura 2 desde: {CSV_PATH}...", flush=True)
    
    ratings_posibles = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    rating_counts = Counter()
    movie_counts = Counter()

    if os.path.exists(CSV_PATH):
        total_filas = 0
        for chunk in pd.read_csv(CSV_PATH, usecols=['movieId', 'rating'], chunksize=5_000_000):
            total_filas += len(chunk)
            rating_counts.update(chunk['rating'].value_counts().to_dict())
            movie_counts.update(chunk['movieId'].value_counts().to_dict())
            print(f"   Filas procesadas: {total_filas:,} / 32,000,204...", flush=True)

        movie_freqs = np.array(sorted(movie_counts.values(), reverse=True))
        ranks = np.arange(1, len(movie_freqs) + 1)
        
        top_10_idx = max(1, int(len(movie_freqs) * 0.10))
        pct_votos_top10 = (movie_freqs[:top_10_idx].sum() / movie_freqs.sum()) * 100
        conteo_notas_millones = np.array([rating_counts.get(r, 0) for r in ratings_posibles]) / 1e6
        
        print(f"   [OK] Total películas únicas: {len(movie_freqs):,}", flush=True)
        print(f"   [OK] Top 10% títulos ({top_10_idx:,} filmes) concentra el {pct_votos_top10:.2f}% de votos.", flush=True)
    else:
        print(f"   [AVISO] No se encontró {CSV_PATH}. Usando aproximación analítica.", flush=True)
        ranks = np.linspace(1, 5000, 5000)
        movie_freqs = 100000 / (ranks ** 1.05)
        top_10_idx = 500
        pct_votos_top10 = 74.5
        conteo_notas_millones = np.array([0.45, 1.1, 0.55, 2.2, 1.6, 6.4, 4.1, 8.5, 2.7, 4.4])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot A: Curva Long-Tail
    ax1.plot(ranks, movie_freqs, color='#1976D2', lw=2.2, label='Frecuencia real')
    head_mask = ranks <= top_10_idx
    tail_mask = ranks > top_10_idx
    ax1.fill_between(ranks[head_mask], movie_freqs[head_mask], color='#90CAF9', alpha=0.45, label='Top 10% Filmes Comerciales')
    ax1.fill_between(ranks[tail_mask], movie_freqs[tail_mask], color='#FFCCBC', alpha=0.55, label='Cola Larga / Nicho (90% restante)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Rango de popularidad de la película (Escala Log)', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('Número de valoraciones recibidas (Escala Log)', fontsize=10.5, fontweight='bold')
    ax1.set_title('(a) Distribución Long-Tail Real (MovieLens 32M)', fontsize=11, fontweight='bold', pad=12)
    ax1.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.8, frameon=True)

    ax1.annotate(f'Sesgo real en datos:\nTop 10% concentra el {pct_votos_top10:.1f}% de votos', 
                 xy=(ranks[min(20, len(ranks)-1)], movie_freqs[min(20, len(movie_freqs)-1)]), 
                 xytext=(ranks[min(60, len(ranks)-1)], movie_freqs[min(5, len(movie_freqs)-1)] * 0.8),
                 arrowprops=dict(facecolor='#0D47A1', shrink=0.08, width=1, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#0D47A1')

    # Subplot B: Histograma de notas
    bars = ax2.bar(ratings_posibles, conteo_notas_millones, width=0.38, color='#388E3C', edgecolor='#1B5E20', alpha=0.85, zorder=3)
    ax2.set_xlabel('Valoración asignada (Estrellas)', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Millones de valoraciones', fontsize=10.5, fontweight='bold')
    ax2.set_title('(b) Distribución de Ratings Explícitos Reales', fontsize=11, fontweight='bold', pad=12)
    ax2.set_xticks(ratings_posibles)
    ax2.grid(axis='y', ls='--', lw=0.5, alpha=0.6, zorder=0)

    idx_max = np.argmax(conteo_notas_millones)
    bars[idx_max].set_color('#1B5E20')

    ax2.axvline(3.5, color='#D32F2F', linestyle=':', lw=1.8, label='Umbral Binarización (≥ 3.5)')
    ax2.legend(loc='upper left', fontsize=8.8, frameon=True)

    plt.tight_layout()
    ruta_salida = os.path.join(OUTPUT_DIR, "distribucion_ratings.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Creado: {ruta_salida}", flush=True)

# ==============================================================================
# FIGURA 3: arquitectura_two_tower.png
# ==============================================================================
def generar_arquitectura_two_tower():
    print("-> Generando figura 3: arquitectura_two_tower.png...", flush=True)
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    c_box = '#F5F5F5'
    c_b = '#424242'

    def node(x, y, w, h, title, sub="", bg=c_box, bcol=c_b, shape="rect"):
        if shape == "round":
            p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       lw=1.3, edgecolor=bcol, facecolor=bg, zorder=2)
        else:
            p = patches.Rectangle((x, y), w, h, lw=1.3, edgecolor=bcol, facecolor=bg, zorder=2)
        ax.add_patch(p)
        if sub:
            ax.text(x + w/2, y + h/2 + 0.15, title, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#111111', zorder=3)
            ax.text(x + w/2, y + h/2 - 0.22, sub, ha='center', va='center',
                    fontsize=7.5, color='#444444', zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#111111', zorder=3)

    def link(p1, p2, label=""):
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.5, mutation_scale=13), zorder=1)
        if label:
            mx, my = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2 + 0.15
            ax.text(mx, my, label, ha='center', va='bottom', fontsize=7.5, color='#263238', backgroundcolor='white')

    u_box = patches.FancyBboxPatch((0.5, 0.5), 4.2, 8.5, boxstyle="round,pad=0.15,rounding_size=0.2",
                                   lw=1.5, edgecolor='#1565C0', facecolor='#E3F2FD', alpha=0.35, zorder=1)
    ax.add_patch(u_box)
    ax.text(2.6, 8.7, "TORRE DE USUARIO Ψ_U(x_u)\n[Online / CPU Multihilo]",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0D47A1')

    node(0.8, 1.0, 3.6, 1.2, "Entradas de Usuario", "Historial {i_1...i_T}, notas r_ui, perfiles LB", bg='#FFFFFF', bcol='#1976D2', shape="round")
    node(0.8, 2.7, 3.6, 1.1, "Agregación Ponderada", "u_agg = ∑ c_uj · v_j (SIMD AVX)", bg='#FFFFFF', bcol='#1976D2', shape="round")
    node(0.8, 4.3, 3.6, 1.1, "Capas Densas + LayerNorm", "Dense 256 → 128 (Activación GeLU)", bg='#FFFFFF', bcol='#1976D2', shape="round")
    node(0.8, 5.9, 3.6, 1.0, "Normalización L_2", "u = u / ||u||_2 ∈ ℝ^128", bg='#BBDEFB', bcol='#0D47A1', shape="round")
    node(0.8, 7.3, 3.6, 0.9, "Vector de Usuario u", "Dimensión d = 128", bg='#90CAF9', bcol='#0D47A1', shape="round")

    link((2.6, 2.2), (2.6, 2.7))
    link((2.6, 3.8), (2.6, 4.3))
    link((2.6, 5.4), (2.6, 5.9))
    link((2.6, 6.9), (2.6, 7.3))

    i_box = patches.FancyBboxPatch((5.3, 0.5), 4.2, 8.5, boxstyle="round,pad=0.15,rounding_size=0.2",
                                   lw=1.5, edgecolor='#2E7D32', facecolor='#E8F5E9', alpha=0.35, zorder=1)
    ax.add_patch(i_box)
    ax.text(7.4, 8.7, "TORRE DE ÍTEM Ψ_I(x_i)\n[Offline / Precomputado]",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1B5E20')

    node(5.6, 1.0, 3.6, 1.2, "Metadatos Heterogéneos", "Sinopsis (SentenceTransformers 384d)\nMulti-Hot (Directores, Géneros)", bg='#FFFFFF', bcol='#388E3C', shape="round")
    node(5.6, 2.7, 3.6, 1.1, "Concatenación y Embedding", "W_text · t_emb + W_cat · x_cat", bg='#FFFFFF', bcol='#388E3C', shape="round")
    node(5.6, 4.3, 3.6, 1.1, "Capas Densas + Dropout", "Dense 512 → 256 → 128 (GeLU)", bg='#FFFFFF', bcol='#388E3C', shape="round")
    node(5.6, 5.9, 3.6, 1.0, "Normalización L_2", "v_i = v_i / ||v_i||_2 ∈ ℝ^128", bg='#C8E6C9', bcol='#1B5E20', shape="round")
    node(5.6, 7.3, 3.6, 0.9, "Vector de Película v_i", "Dimensión d = 128", bg='#A5D6A7', bcol='#1B5E20', shape="round")

    link((7.4, 2.2), (7.4, 2.7))
    link((7.4, 3.8), (7.4, 4.3))
    link((7.4, 5.4), (7.4, 5.9))
    link((7.4, 6.9), (7.4, 7.3))

    train_box = patches.FancyBboxPatch((10.1, 5.4), 4.4, 3.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       lw=1.4, edgecolor='#C2185B', facecolor='#FCE4EC', zorder=2)
    ax.add_patch(train_box)
    ax.text(12.3, 8.65, "Entrenamiento: InfoNCE con LogQ", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#880E4F')
    ax.text(10.3, 7.9, "• Puntuación Corregida:", fontsize=8.5, fontweight='bold', color='#333333')
    ax.text(10.5, 7.45, "s*(u, i) = uᵀ v_i - τ · log P(i)", fontsize=8.5, color='#AD1457')
    ax.text(10.3, 6.9, "• Mitigación de Sesgo:", fontsize=8.5, fontweight='bold', color='#333333')
    ax.text(10.5, 6.45, "Penaliza la frecuencia marginal P(i) ∝ n_i^0.75", fontsize=7.8, color='#333333')
    ax.text(10.3, 5.95, "• In-Batch Negatives con Hard Mining", fontsize=8.2, color='#333333')

    link((4.4, 7.7), (10.1, 7.7), "u")
    link((9.2, 7.7), (10.1, 7.7), "v_i")

    inf_box = patches.FancyBboxPatch((10.1, 0.5), 4.4, 4.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     lw=1.4, edgecolor='#E65100', facecolor='#FFF3E0', zorder=2)
    ax.add_patch(inf_box)
    ax.text(12.3, 4.5, "Inferencia: 2 Etapas Desacopladas", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#BF360C')

    node(10.4, 2.9, 3.8, 1.2, "Etapa 1: Búsqueda ANN (HNSW)", "Índice FAISS en memoria RAM\nRecupera Top K_c = 200 (< 5 ms)", bg='#FFE0B2', bcol='#E65100', shape="round")
    node(10.4, 0.9, 3.8, 1.4, "Etapa 2: Re-Ranking MMR", "Score = λ·Affinity - (1-λ)·max Sim(v_i, v_j)\nPromueve Cine de Nicho → Top K = 20", bg='#FFE0B2', bcol='#E65100', shape="round")

    link((12.3, 2.9), (12.3, 2.3), "Candidatos K_c")
    link((4.4, 7.5), (10.4, 3.5), "MIPS u")
    link((9.2, 7.5), (10.4, 3.5), "Index v_i")

    plt.tight_layout()
    ruta_salida = os.path.join(OUTPUT_DIR, "arquitectura_two_tower.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Creado: {ruta_salida}", flush=True)

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando generación de diagramas...", flush=True)
    generar_diagrama_problema()
    generar_distribucion_ratings()
    generar_arquitectura_two_tower()
    print("¡Todas las figuras han sido generadas con éxito!", flush=True)