"""
evaluate_ffl.py — Avaliação comparativa do ablation study (Projeto Final CMP570)

Compara as saídas SR de cada variante do ablation contra o ground truth (aerofoto)
usando métricas no espaço de imagem (SSIM, cPSNR) e no domínio de frequência
(PSD_L2), além de gerar curvas PSD radiais para a apresentação.

Uso:
    python evaluate_ffl.py --sr_dirs <dir_baseline> <dir_ffl_w01> <dir_ffl_w10> \
                           --gt_dir  <dir_gt>        \
                           --names   baseline ffl_w01 ffl_w10 \
                           --out_dir results/evaluation

As imagens SR e GT devem ser PNGs RGB (128x128 por chip).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_ssim(img, gt):
    return ssim(img, gt, channel_axis=-1, data_range=1.0)


def compute_cpsnr(img, gt):
    """cPSNR: PSNR com ajuste afim por canal (tolerante a offset cross-sensor)."""
    best = -np.inf
    for c in range(img.shape[2]):
        ic = img[:, :, c].flatten().astype(np.float64)
        gc = gt[:, :, c].flatten().astype(np.float64)
        # ajuste afim: ic_adj = a*ic + b minimiza MSE contra gc
        A = np.vstack([ic, np.ones_like(ic)]).T
        a, b = np.linalg.lstsq(A, gc, rcond=None)[0]
        ic_adj = np.clip(a * ic + b, 0, 1)
        mse = np.mean((ic_adj - gc) ** 2)
        if mse < 1e-10:
            return 100.0
        best = max(best, 10 * np.log10(1.0 / mse))
    return best


def radial_psd(img_gray):
    """PSD radial média de uma imagem em escala de cinza [H,W] ∈ [0,1]."""
    f = np.fft.fft2(img_gray)
    f = np.fft.fftshift(f)
    psd2d = np.abs(f) ** 2
    H, W = psd2d.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cy, cx)
    radial = np.array([psd2d[r == ri].mean() if (r == ri).any() else 0.0
                       for ri in range(r_max)])
    return radial


def psd_l2(img, gt):
    """PSD_L2: distância L2 no log₁₀ do PSD radial médio (todas as bandas)."""
    total = 0.0
    for c in range(img.shape[2]):
        p_img = radial_psd(img[:, :, c])
        p_gt  = radial_psd(gt[:, :, c])
        # evitar log(0)
        p_img = np.clip(p_img, 1e-10, None)
        p_gt  = np.clip(p_gt,  1e-10, None)
        total += np.mean((np.log10(p_img) - np.log10(p_gt)) ** 2)
    return total / img.shape[2]


# ---------------------------------------------------------------------------
# Carregamento de imagens
# ---------------------------------------------------------------------------

def load_images(directory):
    """Carrega todos os PNGs de um diretório como float [0,1]."""
    paths = sorted(Path(directory).glob("**/*.png"))
    imgs = []
    for p in paths:
        img = io.imread(str(p)).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        imgs.append((p.stem, img))
    return imgs


# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------

def evaluate(sr_dirs, gt_dir, names, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    gt_imgs = dict(load_images(gt_dir))

    results = {}  # name → {ssim, cpsnr, psd_l2}
    psd_curves = {}  # name → array radial médio

    for name, sr_dir in zip(names, sr_dirs):
        sr_imgs = dict(load_images(sr_dir))
        common = sorted(set(sr_imgs) & set(gt_imgs))
        if not common:
            print(f"[AVISO] Nenhuma imagem em comum entre {sr_dir} e {gt_dir}")
            continue

        ssim_vals, cpsnr_vals, psd_vals = [], [], []
        radials = []

        for stem in common:
            sr  = sr_imgs[stem]
            gt  = gt_imgs[stem]

            # redimensionar se necessário
            if sr.shape != gt.shape:
                from skimage.transform import resize
                sr = resize(sr, gt.shape[:2], anti_aliasing=True)

            ssim_vals.append(compute_ssim(sr, gt))
            cpsnr_vals.append(compute_cpsnr(sr, gt))
            psd_vals.append(psd_l2(sr, gt))

            # acumular PSD radial para plotagem
            for c in range(sr.shape[2]):
                radials.append(radial_psd(sr[:, :, c]))

        results[name] = {
            'ssim':   np.mean(ssim_vals),
            'cpsnr':  np.mean(cpsnr_vals),
            'psd_l2': np.mean(psd_vals),
            'n':      len(common),
        }
        psd_curves[name] = np.mean(radials, axis=0)
        print(f"[{name}] n={len(common)} | SSIM={results[name]['ssim']:.4f} | "
              f"cPSNR={results[name]['cpsnr']:.2f} dB | PSD_L2={results[name]['psd_l2']:.4f}")

    # Adicionar GT como referência na curva PSD
    gt_radials = []
    for _, img in gt_imgs.items():
        for c in range(img.shape[2]):
            gt_radials.append(radial_psd(img[:, :, c]))
    psd_curves['GT'] = np.mean(gt_radials, axis=0)

    # ------------------------------------------------------------------
    # Tabela de resultados (Markdown)
    # ------------------------------------------------------------------
    table_path = os.path.join(out_dir, "results_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Resultados do Ablation Study - Focal Frequency Loss\n\n")
        f.write("| Variante | N | SSIM (maior melhor) | cPSNR dB (maior melhor) | PSD_L2 (menor melhor) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['n']} | {r['ssim']:.4f} | "
                    f"{r['cpsnr']:.2f} | {r['psd_l2']:.4f} |\n")
    print(f"\nTabela salva em {table_path}")

    # ------------------------------------------------------------------
    # Figura: curvas PSD radiais
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    for i, (name, curve) in enumerate(psd_curves.items()):
        freqs = np.arange(len(curve)) / len(curve)
        label = name if name != 'GT' else 'GT (referência)'
        ls = '-' if name == 'GT' else linestyles[i % len(linestyles)]
        lw = 2.5 if name == 'GT' else 1.5
        ax.semilogy(freqs, curve + 1e-10, label=label,
                    color=colors[i % len(colors)], linestyle=ls, linewidth=lw)

    ax.set_xlabel("Frequência normalizada (ciclos/pixel)", fontsize=12)
    ax.set_ylabel("PSD (log)", fontsize=12)
    ax.set_title("Comparação de Espectros de Potência (PSD)\nFocal Frequency Loss Ablation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "psd_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figura PSD salva em {fig_path}")

    # ------------------------------------------------------------------
    # Figura: barras comparativas (SSIM e PSD_L2)
    # ------------------------------------------------------------------
    names_list = list(results.keys())
    ssim_list  = [results[n]['ssim']   for n in names_list]
    psd_list   = [results[n]['psd_l2'] for n in names_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(names_list, ssim_list, color=colors[:len(names_list)])
    ax1.set_title("SSIM ↑ (maior é melhor)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("SSIM")
    for i, v in enumerate(ssim_list):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)

    ax2.bar(names_list, psd_list, color=colors[:len(names_list)])
    ax2.set_title("PSD\\_L2 ↓ (menor é melhor)")
    ax2.set_ylabel("PSD L2")
    for i, v in enumerate(psd_list):
        ax2.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9)

    fig.suptitle("Ablation Study — Focal Frequency Loss no SATLAS", fontsize=13)
    fig.tight_layout()
    bar_path = os.path.join(out_dir, "metrics_comparison.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Figura de barras salva em {bar_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia ablation FFL vs baseline")
    parser.add_argument("--sr_dirs", nargs="+", required=True,
                        help="Diretórios com imagens SR de cada variante")
    parser.add_argument("--gt_dir",  required=True,
                        help="Diretório com imagens ground truth (aerofoto)")
    parser.add_argument("--names",   nargs="+", required=True,
                        help="Nomes das variantes (mesmo orden que --sr_dirs)")
    parser.add_argument("--out_dir", default="results/evaluation",
                        help="Diretório de saída para tabela e figuras")
    args = parser.parse_args()

    if len(args.sr_dirs) != len(args.names):
        parser.error("--sr_dirs e --names devem ter o mesmo número de elementos")

    evaluate(args.sr_dirs, args.gt_dir, args.names, args.out_dir)
