"""
infer_chips.py - Inferencia com checkpoints treinados sobre chips planet.

Uso:
    python infer_chips.py --checkpoint path/to/net_g_5000.pth
                          --planet_dir path/to/planet_chips
                          --out_dir    path/to/sr_output
                          [--gt_dir    path/to/gt_chips]   # cria gt_flat/ junto
                          [--n_s2 16]

Saida:
    out_dir/{chip_id}.png     <- SR gerada pelo modelo (128x128 RGB)
    out_dir/../gt_flat/{chip_id}.png  <- GT copiada em estrutura plana (se --gt_dir fornecido)
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_model(checkpoint_path: Path, device: torch.device):
    # PYTHONPATH deve conter o diretorio satlas-super-resolution (setado pelo run script)
    import ssr.archs
    from basicsr.utils.registry import ARCH_REGISTRY

    net_g = ARCH_REGISTRY.get('SSR_RRDBNet')(
        num_in_ch=48, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    ).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else list(ckpt.keys())[0])
    result = net_g.load_state_dict(ckpt[key], strict=False)
    print(f"Checkpoint carregado: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    net_g.eval()
    return net_g


def chip_to_tensor(tci_path: Path, n_s2: int, device: torch.device) -> torch.Tensor:
    """Le tci.png [T*32, 32, 3] e converte para tensor [1, n_s2*3, 32, 32]."""
    img = np.array(Image.open(tci_path))          # [T*32, 32, 3]
    chunks = img.reshape(-1, 32, 32, 3)            # [T, 32, 32, 3]
    T = chunks.shape[0]
    # Selecionar n_s2 frames (preferir nao-pretos)
    goods = [i for i in range(T) if chunks[i].mean() > 5]
    bads  = [i for i in range(T) if chunks[i].mean() <= 5]
    if len(goods) >= n_s2:
        import random; idx = random.sample(goods, n_s2)
    else:
        import random; idx = goods + random.sample(bads, max(0, n_s2 - len(goods)))
    idx = sorted(idx)
    selected = chunks[idx]                         # [n_s2, 32, 32, 3]
    # [n_s2, 32, 32, 3] -> [n_s2, 3, 32, 32] -> [n_s2*3, 32, 32]
    t = torch.from_numpy(selected).permute(0, 3, 1, 2).reshape(-1, 32, 32)
    return (t.float() / 255.0).unsqueeze(0).to(device)  # [1, n_s2*3, 32, 32]


def run_inference(checkpoint: Path, planet_dir: Path, out_dir: Path,
                  gt_dir: Path | None, n_s2: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    net_g = load_model(checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_dirs = sorted([d for d in planet_dir.iterdir() if d.is_dir()])
    print(f"Chips a processar: {len(chip_dirs)}")

    gt_flat = None
    if gt_dir is not None:
        gt_flat = out_dir.parent / 'gt_flat'
        gt_flat.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for chip_dir in chip_dirs:
        chip_id = chip_dir.name
        tci = chip_dir / 'tci.png'
        if not tci.exists():
            continue

        with torch.no_grad():
            lr = chip_to_tensor(tci, n_s2, device)
            sr = net_g(lr)                          # [1, 3, 128, 128]

        sr_np = (sr.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(sr_np).save(out_dir / f"{chip_id}.png")

        # Copiar GT para estrutura plana (uma unica vez)
        if gt_flat is not None:
            gt_src = gt_dir / chip_id / 'rgb.png'
            gt_dst = gt_flat / f"{chip_id}.png"
            if gt_src.exists() and not gt_dst.exists():
                import shutil; shutil.copy2(gt_src, gt_dst)

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(chip_dirs)} chips processados")

    print(f"Inferencia concluida: {n_done} chips -> {out_dir}")
    if gt_flat:
        gt_count = len(list(gt_flat.glob('*.png')))
        print(f"GT plana: {gt_count} chips -> {gt_flat}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--planet_dir',  required=True)
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--gt_dir',      default=None)
    parser.add_argument('--n_s2',        type=int, default=16)
    args = parser.parse_args()

    run_inference(
        checkpoint  = Path(args.checkpoint),
        planet_dir  = Path(args.planet_dir),
        out_dir     = Path(args.out_dir),
        gt_dir      = Path(args.gt_dir) if args.gt_dir else None,
        n_s2        = args.n_s2,
    )
