"""
validate_chips.py — Valida a estrutura e integridade dos chips antes do treino.

Formato esperado pelo S2NAIPDataset do SATLAS:

  planet_chips/
  └── {chip_id}/
      └── tci.png     ← PNG com T frames temporais empilhadas verticalmente
                        shape: [T*32, 32, 3]  (ex: 16 frames → 512×32 pixels)

  gt_chips/
  └── {chip_id}/
      └── rgb.png     ← aerofoto GT, shape: [128, 128, 3]

Uso:
    python validate_chips.py --data_root C:/Users/marcel.1CGEO/Desktop/zero_shot_data
    python validate_chips.py --data_root C:/Users/marcel.1CGEO/Desktop/zero_shot_data --n_s2 16
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Instale Pillow: pip install Pillow")
    sys.exit(1)


def validate(data_root: Path, n_s2: int = 16):
    planet_dir = data_root / "planet_chips"
    gt_dir     = data_root / "gt_chips"

    if not planet_dir.exists():
        print(f"[ERRO] Não encontrado: {planet_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"[ERRO] Não encontrado: {gt_dir}")
        sys.exit(1)

    chip_dirs = sorted([d for d in planet_dir.iterdir() if d.is_dir()])
    print(f"Chips encontrados em planet_chips/: {len(chip_dirs)}")

    if not chip_dirs:
        print("[ERRO] Nenhum subdiretório em planet_chips/")
        sys.exit(1)

    errors, warnings = [], []
    ok_count = 0

    for chip_dir in chip_dirs:
        chip_id = chip_dir.name
        chip_ok = True

        # --- Verificar LR (tci.png empilhado) ---
        tci_path = chip_dir / "tci.png"
        if not tci_path.exists():
            errors.append(f"{chip_id}: faltando planet_chips/{chip_id}/tci.png")
            chip_ok = False
        else:
            try:
                img = Image.open(tci_path)
                w, h = img.size   # PIL: (width, height)
                expected_h = n_s2 * 32
                if w != 32:
                    warnings.append(f"{chip_id}/tci.png: largura={w}, esperado 32px")
                if h < expected_h:
                    errors.append(f"{chip_id}/tci.png: altura={h}px, precisa de pelo menos {expected_h}px ({n_s2} frames × 32px)")
                    chip_ok = False
                elif h % 32 != 0:
                    warnings.append(f"{chip_id}/tci.png: altura={h} não é múltiplo de 32")
                if img.mode != "RGB":
                    errors.append(f"{chip_id}/tci.png: modo={img.mode}, esperado RGB")
                    chip_ok = False
            except Exception as e:
                errors.append(f"{chip_id}/tci.png: erro ao abrir — {e}")
                chip_ok = False

        # --- Verificar GT (rgb.png) ---
        # O dataset busca recursivamente e usa Path(n).parent.name como chip_id
        # Estrutura mínima: gt_chips/{chip_id}/rgb.png
        gt_path = gt_dir / chip_id / "rgb.png"
        if not gt_path.exists():
            errors.append(f"{chip_id}: GT não encontrado em gt_chips/{chip_id}/rgb.png")
            chip_ok = False
        else:
            try:
                gt = Image.open(gt_path)
                w, h = gt.size
                if (w, h) != (128, 128):
                    warnings.append(f"gt/{chip_id}/rgb.png: tamanho={w}×{h}, esperado 128×128")
                if gt.mode != "RGB":
                    errors.append(f"gt/{chip_id}/rgb.png: modo={gt.mode}, esperado RGB")
                    chip_ok = False
            except Exception as e:
                errors.append(f"gt/{chip_id}/rgb.png: erro ao abrir — {e}")
                chip_ok = False

        if chip_ok:
            ok_count += 1

    # --- Relatório ---
    print(f"\nResultado: {ok_count}/{len(chip_dirs)} chips válidos\n")

    if warnings:
        print(f"[AVISO] {len(warnings)} avisos:")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... e mais {len(warnings) - 10}")
        print()

    if errors:
        print(f"[ERRO] {len(errors)} erros:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... e mais {len(errors) - 20}")
        print("\nCorrija os erros antes de iniciar o treino.")
        sys.exit(1)

    print("Todos os chips estão válidos.")
    n = len(chip_dirs)
    print(f"\nEstimativas com dataset_enlarge_ratio=50, batch=4, total_iter=5000:")
    print(f"  Amostras virtuais: {n * 50}")
    print(f"  Imagens processadas no treino: {5000 * 4}")
    if n < 20:
        print(f"\n  [AVISO] Só {n} chips — alto risco de overfitting.")
        print(f"  Considere aumentar dataset_enlarge_ratio ou coletar mais dados.")
    elif n >= 100:
        print(f"\n  {n} chips — suficiente para ablation significativo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--n_s2", type=int, default=16, help="Número de frames temporais por chip")
    args = parser.parse_args()
    validate(Path(args.data_root), args.n_s2)
