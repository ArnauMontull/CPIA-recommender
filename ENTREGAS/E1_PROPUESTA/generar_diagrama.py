import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuración de figura y lienzo
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Paleta de colores sobria y académica
c_data = '#E3F2FD'     # Azul claro (Datos)
c_border_data = '#1976D2'
c_model = '#E8F5E9'    # Verde claro (Modelos/Proceso)
c_border_model = '#388E3C'
c_retrieval = '#FFF3E0'# Naranja suave (Inferencia/Ranking)
c_border_retrieval = '#F57C00'
c_challenge = '#FFEBEE'# Rojo suave (Retos matemáticos)
c_border_challenge = '#D32F2F'

def draw_card(ax, x, y, w, h, title, text_lines, bg_col, border_col):
    rect = patches.FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.1,rounding_size=0.15",
                                  linewidth=1.5, edgecolor=border_col,
                                  facecolor=bg_col, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#1A1A1A', zorder=3)
    
    current_y = y + h - 0.75
    for line in text_lines:
        ax.text(x + 0.2, current_y, line, ha='left', va='center',
                fontsize=8.5, color='#333333', zorder=3)
        current_y -= 0.35

def draw_arrow(ax, start, end, label=""):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=1.8, mutation_scale=15),
                zorder=1)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.2
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#455A64', backgroundcolor='white')

# --- 1. COLUMNA: ENTRADAS HETEROGÉNEAS ---
draw_card(ax, 0.5, 5.5, 3.2, 2.3, "MovieLens 32M", 
          ["• Matriz R ∈ ℝ^(N×M)", "• 32M valoraciones ordinales", "• Esparcidad crítica: 99.818%"], 
          c_data, c_border_data)

draw_card(ax, 0.5, 2.9, 3.2, 2.3, "The Movies Dataset (TMDB)", 
          ["• Metadatos crudos (45k filmes)", "• Sinopsis, géneros, keywords", "• Reparto y cuerpo técnico"], 
          c_data, c_border_data)

draw_card(ax, 0.5, 0.3, 3.2, 2.3, "Letterboxd Profile", 
          ["• Historial de usuario externo", "• Secuencia temporal de diario", "• Reto de Cold-Start activo"], 
          c_data, c_border_data)

# --- 2. COLUMNA: PROCESAMIENTO Y MITIGACIÓN ---
draw_card(ax, 4.4, 4.3, 3.4, 3.5, "Espacio Semántico y Regularización", 
          ["• SentenceTransformers:", "   Incrustaciones densas d=384", 
           "• Codificación Multi-Hot:", "   Géneros y directores", 
           "• Mitigación LogQ:", "   Penalización de popularidad",
           "• Pasarela Relacional:", "   Mapeo unívoco vía links.csv"], 
          c_challenge, c_border_challenge)

draw_card(ax, 4.4, 0.8, 3.4, 2.8, "Perfilado Dinámico de Usuario", 
          ["• Agregación de historial", "• Ponderación por rating continuo", 
           "• Inferencia amortizada en CPU", "• Normalización en hiperesfera ||u||=1"], 
          c_challenge, c_border_challenge)

# --- 3. COLUMNA: NÚCLEO TWO-TOWER DESACOPLADO ---
draw_card(ax, 8.5, 4.5, 3.2, 3.2, "Torre de Ítem Ψ_I(x_i)", 
          ["• Proyección lineal a d=128", "• Fusión de texto + categorías", 
           "• v_i normalizado (||v_i||=1)", "• Indexación offline en memoria"], 
          c_model, c_border_model)

draw_card(ax, 8.5, 0.8, 3.2, 3.2, "Torre de Usuario Ψ_U(x_u)", 
          ["• Vector de preferencias u ∈ ℝ^128", "• Cálculo en tiempo real", 
           "• Cómputo SIMD / AVX en CPU", "• Desacoplamiento total"], 
          c_model, c_border_model)

# Bloque Producto Interno
rect_dot = patches.Circle((10.1, 4.0), 0.45, linewidth=1.5, edgecolor='#388E3C', facecolor='#C8E6C9', zorder=2)
ax.add_patch(rect_dot)
ax.text(10.1, 4.0, "⟨u, v_i⟩", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1B5E20', zorder=3)

# --- 4. COLUMNA: RECUPERACIÓN Y RE-RANKING ---
draw_card(ax, 12.2, 4.6, 2.5, 3.0, "Etapa 1: Recuperación", 
          ["• Búsqueda ANN (HNSW)", "• Máximo producto escalar", "• Filtro top K_c = 200", "• Latencia < 5 ms en CPU"], 
          c_retrieval, c_border_retrieval)

draw_card(ax, 12.2, 0.8, 2.5, 3.0, "Etapa 2: Re-Ranking", 
          ["• Algoritmo MMR (λ ≈ 0.7)", "• Diversidad intra-lista", "• Impulso a cine de nicho", "• Lista final Top-K (K=20)"], 
          c_retrieval, c_border_retrieval)

# --- FLECHAS DE CONEXIÓN ---
draw_arrow(ax, (3.7, 6.5), (4.4, 6.0))
draw_arrow(ax, (3.7, 4.0), (4.4, 5.0))
draw_arrow(ax, (3.7, 1.5), (4.4, 2.2))

draw_arrow(ax, (7.8, 6.0), (8.5, 6.0), "Metadatos")
draw_arrow(ax, (7.8, 2.2), (8.5, 2.2), "Historial")

draw_arrow(ax, (10.1, 4.5), (10.1, 4.45))
draw_arrow(ax, (10.1, 3.5), (10.1, 3.55))

draw_arrow(ax, (11.7, 6.0), (12.2, 6.0))
draw_arrow(ax, (13.45, 4.6), (13.45, 3.8), "K_c candidatos")

plt.tight_layout()
plt.savefig("diagrama_problema.png", dpi=300, bbox_inches='tight')
print("Imagen generada con éxito: diagrama_problema.png")