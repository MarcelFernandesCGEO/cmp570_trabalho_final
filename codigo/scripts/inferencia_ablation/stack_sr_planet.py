#!/usr/bin/env python3
"""
Stack de imagens PlanetScope para Super-Resolução (adaptado de stack_sr.py).

Diferenças do original (Sentinel-2, 4 imagens, 9.555m):
  - Processa uma pasta PLANA com N TIFs Planet (não subpastas)
  - Stacks com 16 imagens → 48 bandas (16 × RGB)
  - Pixel alvo: 4.778m (resolução nativa PlanetScope confirmada via gdalinfo)
  - Sem parser de nome Sentinel — aceita qualquer TIF RGB

Saída: um único GeoTIFF com N*3 bandas, reprojetado para EPSG:3857.

Uso:
  python stack_sr_planet.py \
      --input_dir /dados/user_data/marcel/cmp570_final/trabalho_final/data/planet \
      --output    /dados/user_data/marcel/cmp570_final/trabalho_final/data/planet_stack.tif

  # Ver quantos TIFs foram encontrados (sem gerar saída):
  python stack_sr_planet.py --input_dir /caminho/planet --dry_run
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

# -----------------------------------------------------------------------
# Parâmetros — AJUSTE SE NECESSÁRIO
# -----------------------------------------------------------------------
TARGET_SRS = "EPSG:3857"
PIXEL_SIZE  = 4.778480889   # metros — resolução nativa PlanetScope PSScene
                            # Confirmado via: gdalinfo → Pixel Size = (4.778480889,-4.777927189)
RESAMPLE    = "bilinear"
EXTENSIONS  = (".tif", ".tiff")
N_EXPECTED  = 16            # número de imagens Planet esperadas
# -----------------------------------------------------------------------


def list_planet_tifs(folder: Path) -> list[Path]:
    """Lista todos os TIFs da pasta, ordenados por nome."""
    tifs = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in EXTENSIONS and f.is_file()
    )
    return tifs


def stack_planet(input_dir: str, output_path: str, n_expected: int = N_EXPECTED,
                 dry_run: bool = False) -> bool:
    folder = Path(input_dir)
    images = list_planet_tifs(folder)

    print(f"Pasta: {folder}")
    print(f"TIFs encontrados: {len(images)}")
    for i, img in enumerate(images):
        print(f"  [{i:02d}] {img.name}")

    if len(images) == 0:
        print("ERRO: nenhum TIF encontrado.")
        return False

    if len(images) != n_expected:
        print(f"AVISO: encontrados {len(images)} TIFs, esperava {n_expected}.")
        print(f"       Continuando com {len(images)} imagens → {len(images)*3} bandas.")

    if dry_run:
        print("Dry run — saindo sem gerar arquivo.")
        return True

    output = Path(output_path)
    if output.exists():
        print(f"AVISO: saída já existe, sobrescrevendo: {output}")

    n_imgs = len(images)
    print(f"\nCriando stack: {n_imgs} imagens × 3 bandas = {n_imgs*3} bandas")
    print(f"SRS alvo:  {TARGET_SRS}")
    print(f"Pixel:     {PIXEL_SIZE} m")
    print(f"Saída:     {output}\n")

    with tempfile.TemporaryDirectory() as tmp:
        # Expandir cada imagem RGB em 3 VRTs single-band
        # Ordem: img0_R, img0_G, img0_B, img1_R, img1_G, img1_B, ...
        band_vrts = []
        for idx, img in enumerate(images):
            ds = gdal.Open(str(img))
            nb = ds.RasterCount
            ds = None

            if nb < 3:
                print(f"  ERRO: {img.name} tem apenas {nb} banda(s) — esperava >= 3.")
                return False

            for b in range(1, 4):   # bandas 1, 2, 3 (R, G, B)
                vrt_path = os.path.join(tmp, f"img{idx:02d}_b{b}.vrt")
                gdal.Translate(vrt_path, str(img), bandList=[b], format="VRT")
                band_vrts.append(vrt_path)

            print(f"  [{idx:02d}] {img.name} ({nb} bandas → usando RGB)")

        # Stack todas as bandas num único VRT
        stack_vrt = os.path.join(tmp, "planet_stack.vrt")
        ds = gdal.BuildVRT(
            stack_vrt, band_vrts,
            options=gdal.BuildVRTOptions(separate=True)
        )
        ds.FlushCache()
        ds = None

        # Reprojetar para TARGET_SRS com pixel PIXEL_SIZE
        print(f"\nReprojetando para {TARGET_SRS} @ {PIXEL_SIZE}m...")
        ds = gdal.Warp(
            str(output),
            stack_vrt,
            options=gdal.WarpOptions(
                dstSRS=TARGET_SRS,
                xRes=PIXEL_SIZE,
                yRes=PIXEL_SIZE,
                resampleAlg=RESAMPLE,
                outputType=gdal.GDT_Byte,
                format="GTiff",
                creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
            ),
        )
        ds.FlushCache()
        ds = None

    size_mb = output.stat().st_size / 1e6
    print(f"\nOK: {output.name} ({n_imgs*3} bandas, {size_mb:.1f} MB)")
    return True


def main():
    global PIXEL_SIZE

    parser = argparse.ArgumentParser(
        description="Stack de imagens Planet para SR (adaptado para 16 imagens / 48 bandas)"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Pasta com os TIFs Planet (ex: data/planet)")
    parser.add_argument("--output",     default=None,
                        help="Caminho do TIF de saída. Default: input_dir/../planet_stack.tif")
    parser.add_argument("--n_expected", type=int, default=N_EXPECTED,
                        help=f"Número esperado de imagens (default: {N_EXPECTED})")
    parser.add_argument("--pixel_size", type=float, default=PIXEL_SIZE,
                        help=f"Resolução alvo em metros (default: {PIXEL_SIZE})")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Apenas listar TIFs, sem gerar saída")
    args = parser.parse_args()

    PIXEL_SIZE = args.pixel_size

    if args.output is None:
        out = Path(args.input_dir).parent / "planet_stack.tif"
    else:
        out = Path(args.output)

    ok = stack_planet(args.input_dir, str(out), args.n_expected, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
