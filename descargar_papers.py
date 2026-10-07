import os
import urllib.request

DEST_DIR = os.path.join("docs", "papers")
os.makedirs(DEST_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

PAPERS = {
    "1998_Carbonell_MMR_ReRanking.pdf": "http://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_1998.pdf",
    "2009_Koren_Matrix_Factorization_Netflix.pdf": "https://www.cs.rochester.edu/u/ehart/csc298/papers/netflix.pdf",
    "2015_Harper_MovieLens_Datasets_History.pdf": "https://par.nsf.gov/servlets/purl/10041285",
    "2016_Covington_YouTube_DNN_Recommendations.pdf": "https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf",
    "2017_He_NCF_Neural_Collaborative_Filtering.pdf": "https://arxiv.org/pdf/1708.05031.pdf",
    "2018_Liang_MultiVAE_Collaborative_Filtering.pdf": "https://arxiv.org/pdf/1802.05814.pdf",
    "2018_Kang_SASRec_Sequential_Recommendation.pdf": "https://arxiv.org/pdf/1808.09781.pdf",
    "2019_Steck_EASE_Shallow_Autoencoders.pdf": "https://arxiv.org/pdf/1905.03375.pdf",
    "2019_Yi_TwoTower_LogQ_SamplingBias.pdf": "https://storage.googleapis.com/pub-tools-public-publication-data/pdf/48840.pdf",
    "2020_He_LightGCN_Graph_Convolution.pdf": "https://arxiv.org/pdf/2002.02126.pdf",
}

def descargar():
    print(f"Descargando artículos en: {os.path.abspath(DEST_DIR)}\n")
    for nombre, url in PAPERS.items():
        destino = os.path.join(DEST_DIR, nombre)
        if os.path.exists(destino):
            print(f"[Existente] {nombre}")
            continue

        print(f"[Descargando] {nombre}...")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp, open(destino, "wb") as f:
                f.write(resp.read())
            print(f"   ✓ Guardado correctamente.")
        except Exception as e:
            print(f"   ✗ Error en {nombre}: {e}")

    print("\n[✓] Descargas finalizadas en docs/papers/")

if __name__ == "__main__":
    descargar()
