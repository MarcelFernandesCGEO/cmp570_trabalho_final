"""
prepare_chips.py - Prepara chips de treinamento a partir dos TIFs locais.

Entrada:
  planet_dir/*.tif  - cenas PlanetScope TCI (multi-temporal, RGB+alpha)
  gt_tif            - aerofoto GT (RGB ou RGBA, qualquer resolucao)

Saida (formato exato do S2NAIPDataset do SATLAS):
  out_dir/
  |- planet_chips/{xi:04d}_{yi:04d}/tci.png   [T*32 x 32 px, RGB]
  +- gt_chips/{xi:04d}_{yi:04d}/rgb.png        [128 x 128 px, RGB]

Alinhamento:
  A interseccao geografica entre todos os TIFs Planet e o GT e calculada
  automaticamente. A menor area comum define o bounding box de trabalho.
  Dentro dessa area, o GT define o grid de chips (resolucao e dimensoes).
  O Planet e reamostrado para cobrir exatamente a mesma area com relacao 4:1.

Uso:
  python prepare_chips.py --planet_dir data/planet --gt_tif data/gt/gt.tif
  python prepare_chips.py --planet_dir data/planet --gt_tif data/gt/gt.tif \
                          --out_dir /dados/chips --chip_lr 32 --chip_hr 128
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from PIL import Image


def intersection_bounds(bounds_list):
    """Retorna a interseccao geografica de uma lista de BoundingBox."""
    left   = max(b.left   for b in bounds_list)
    bottom = max(b.bottom for b in bounds_list)
    right  = min(b.right  for b in bounds_list)
    top    = min(b.top    for b in bounds_list)
    if left >= right or bottom >= top:
        print("[ERRO] Sem area de interseccao entre Planet e GT.")
        sys.exit(1)
    return left, bottom, right, top


def read_cropped_rgb(path, bbox, out_w, out_h):
    """Le bandas RGB de um TIF recortado para bbox e reamostrado para out_w x out_h."""
    left, bottom, right, top = bbox
    with rasterio.open(path) as src:
        win = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(
            [1, 2, 3],
            window=win,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.lanczos
        )
    return arr  # [3, H, W] uint8


def prepare_chips(planet_dir: Path, gt_tif: Path, out_dir: Path,
                  chip_lr: int = 32, chip_hr: int = 128):

    # ------------------------------------------------------------------
    # 1. Coletar bounds de todos os arquivos e calcular interseccao
    # ------------------------------------------------------------------
    planet_paths = sorted({p.resolve() for p in planet_dir.glob("*.tif")} |
                          {p.resolve() for p in planet_dir.glob("*.TIF")})
    if not planet_paths:
        print(f"[ERRO] Nenhum .tif em {planet_dir}"); sys.exit(1)

    all_bounds = []
    with rasterio.open(gt_tif) as src:
        gt_res = src.res
        all_bounds.append(src.bounds)

    for p in planet_paths:
        with rasterio.open(p) as src:
            all_bounds.append(src.bounds)

    bbox = intersection_bounds(all_bounds)
    left, bottom, right, top = bbox
    area_w = right - left
    area_h = top   - bottom

    print(f"Interseccao geografica:")
    print(f"  left={left:.1f}  bottom={bottom:.1f}  right={right:.1f}  top={top:.1f}")
    print(f"  Area: {area_w/1000:.2f} km x {area_h/1000:.2f} km")

    # ------------------------------------------------------------------
    # 2. Dimensionar grid a partir do GT recortado
    # ------------------------------------------------------------------
    # Pixels do GT na area de interseccao
    gt_crop_w = int(round(area_w / gt_res[0]))
    gt_crop_h = int(round(area_h / gt_res[1]))

    n_x = gt_crop_w // chip_hr
    n_y = gt_crop_h // chip_hr
    if n_x == 0 or n_y == 0:
        print(f"[ERRO] Area de interseccao muito pequena para gerar chips ({gt_crop_w}x{gt_crop_h} px GT).")
        sys.exit(1)

    usable_w = n_x * chip_hr
    usable_h = n_y * chip_hr
    lr_w = n_x * chip_lr
    lr_h = n_y * chip_lr

    # Ajustar bbox para multiplo exato de chip_hr
    usable_geo_w = usable_w * gt_res[0]
    usable_geo_h = usable_h * gt_res[1]
    bbox_usable = (left, top - usable_geo_h, left + usable_geo_w, top)

    print(f"\nGrid de chips: {n_x} x {n_y} = {n_x * n_y} pares")
    print(f"GT usavel:  {usable_w} x {usable_h} px  (de {gt_crop_w}x{gt_crop_h} na interseccao)")
    print(f"LR alvo:    {lr_w} x {lr_h} px  (Planet reamostrado)")
    print(f"Escala efetiva: {(usable_w/lr_w):.2f}x")

    # ------------------------------------------------------------------
    # 3. Carregar GT recortado
    # ------------------------------------------------------------------
    print("\nCarregando GT...", end=" ", flush=True)
    gt_arr = read_cropped_rgb(gt_tif, bbox_usable, usable_w, usable_h)
    print(f"OK - shape {gt_arr.shape}")

    # ------------------------------------------------------------------
    # 4. Carregar e reamostrar Planet TIFs
    # ------------------------------------------------------------------
    T = len(planet_paths)
    if T < 16:
        print(f"[AVISO] Apenas {T} Planet TIFs (recomendado >= 16)")

    print(f"Carregando {T} Planet TIFs -> {lr_w}x{lr_h}...")
    planet_arrays = []
    for p in planet_paths:
        arr = read_cropped_rgb(p, bbox_usable, lr_w, lr_h)
        planet_arrays.append(arr)
        print(f"  {p.name}: {arr.shape}")
    print(f"Todos os {T} TIFs carregados.\n")

    # ------------------------------------------------------------------
    # 5. Criar diretorios de saida
    # ------------------------------------------------------------------
    (out_dir / "planet_chips").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_chips").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Extrair e salvar chips
    # ------------------------------------------------------------------
    n_saved = n_skipped_gt = n_skipped_lr = 0
    total = n_x * n_y
    print(f"Gerando {total} pares de chips...")

    for yi in range(n_y):
        for xi in range(n_x):
            chip_id = f"{xi:04d}_{yi:04d}"

            y0_hr = yi * chip_hr
            x0_hr = xi * chip_hr
            gt_chip = gt_arr[:, y0_hr:y0_hr + chip_hr, x0_hr:x0_hr + chip_hr]

            if gt_chip.mean() < 10:
                n_skipped_gt += 1
                continue

            y0_lr = yi * chip_lr
            x0_lr = xi * chip_lr

            lr_frames = []
            all_black = True
            for arr in planet_arrays:
                frame = arr[:, y0_lr:y0_lr + chip_lr, x0_lr:x0_lr + chip_lr]
                if frame.mean() > 5:
                    all_black = False
                lr_frames.append(frame)

            if all_black:
                n_skipped_lr += 1
                continue

            lr_stacked = np.concatenate(lr_frames, axis=1)  # [3, T*chip_lr, chip_lr]
            lr_png = lr_stacked.transpose(1, 2, 0).astype(np.uint8)
            gt_png = gt_chip.transpose(1, 2, 0).astype(np.uint8)

            chip_dir = out_dir / "planet_chips" / chip_id
            chip_dir.mkdir(exist_ok=True)
            Image.fromarray(lr_png).save(chip_dir / "tci.png")

            gt_chip_dir = out_dir / "gt_chips" / chip_id
            gt_chip_dir.mkdir(exist_ok=True)
            Image.fromarray(gt_png).save(gt_chip_dir / "rgb.png")

            n_saved += 1

        if (yi + 1) % 10 == 0 or yi == n_y - 1:
            print(f"  Progresso: {yi+1}/{n_y} linhas - {n_saved} chips salvos")

    # ------------------------------------------------------------------
    # 7. Relatorio
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print(f"Chips salvos:          {n_saved}")
    print(f"Ignorados (GT preto):  {n_skipped_gt}")
    print(f"Ignorados (LR preto):  {n_skipped_lr}")
    print(f"Output: {out_dir}")
    print()
    print(f"Estrutura gerada:")
    print(f"  planet_chips/  - {n_saved} dirs, cada um com tci.png [{T*chip_lr}x{chip_lr}px, RGB]")
    print(f"  gt_chips/      - {n_saved} dirs, cada um com rgb.png [{chip_hr}x{chip_hr}px, RGB]")
    print()
    print("Proximo passo: rodar validate_chips.py para confirmar integridade.")


if __name__ == "__main__":
    BASE = Path(__file__).parents[1]

    parser = argparse.ArgumentParser(description="Prepara chips Planet+GT para treinamento SATLAS")
    parser.add_argument("--planet_dir", default=str(BASE / "data" / "planet"),
                        help="Pasta com os TIFs Planet")
    parser.add_argument("--gt_tif",     default=str(BASE / "data" / "gt" / "gt.tif"),
                        help="Caminho para o TIF GT aerofoto")
    parser.add_argument("--out_dir",    default=str(Path(r"C:\Users\marcel.1CGEO\Desktop\zero_shot_data")),
                        help="Diretorio de saida para os chips")
    parser.add_argument("--chip_lr",    type=int, default=32,
                        help="Tamanho do chip LR em pixels (default: 32)")
    parser.add_argument("--chip_hr",    type=int, default=128,
                        help="Tamanho do chip HR em pixels (default: 128)")
    args = parser.parse_args()

    prepare_chips(
        planet_dir = Path(args.planet_dir),
        gt_tif     = Path(args.gt_tif),
        out_dir    = Path(args.out_dir),
        chip_lr    = args.chip_lr,
        chip_hr    = args.chip_hr,
    )
