"""Empaqueta los datos procesados en un zip para subirlo a Drive (H8).

Uso: python scripts/empaquetar_datos.py --muestra {dev,completo}

Crea dist/datos_<muestra>.zip con data/processed/<muestra>/, data/processed/emb_texto.npy
y data/processed/movies.parquet (rutas relativas a la raiz del repo), y dist/datos_<muestra>.sha256.
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    args = ap.parse_args()

    proc = RAIZ / "data" / "processed"
    carpeta = proc / args.muestra
    sueltos = [proc / "emb_texto.npy", proc / "movies.parquet"]
    faltan = [p for p in [carpeta, *sueltos] if not p.exists()]
    if faltan:
        print("Faltan: " + ", ".join(str(p.relative_to(RAIZ)) for p in faltan), file=sys.stderr)
        return 1

    archivos = sorted(p for p in carpeta.rglob("*") if p.is_file()) + sueltos
    dist = RAIZ / "dist"
    dist.mkdir(exist_ok=True)
    destino = dist / f"datos_{args.muestra}.zip"
    # Los .npy y .parquet ya van comprimidos o casi: ZIP_STORED es mas rapido.
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED) as z:
        for p in archivos:
            z.write(p, p.relative_to(RAIZ).as_posix())

    digest = sha256_de(destino)
    (dist / f"datos_{args.muestra}.sha256").write_text(f"{digest}  {destino.name}\n", encoding="utf-8")
    print(f"Archivos: {len(archivos)}")
    print(f"Zip: {destino}")
    print(f"Tamano: {destino.stat().st_size / 1e6:.1f} MB")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
