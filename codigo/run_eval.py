"""
run_eval.py — Avaliação direta das visualizações salvas pelo treino SATLAS.

Uso (no servidor):
    python run_eval.py

Usa o iter 20000 (final) para todos os variantes — mesmo orçamento de treino,
comparação justa para ablation study. Plota também as curvas de SSIM e PSNR
por iter (do log) para verificar convergência.

Salva em results/evaluation/:
    results_table.md
    psd_comparison.png
    metrics_comparison.png
    convergence_curves.png
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from skimage.metrics import structural_similarity as ssim

BASE = Path("/dados/user_data/marcel/cmp570_final/trabalho_final/zero_shot/results")
VARIANTS = ["baseline", "ffl_w01", "ffl_w10"]
ITER = 20000
OUT_DIR = BASE / "evaluation"


def parse_log(variant):
    """Retorna dict {iter: {'psnr': float, 'ssim': float}} do log do BasicSR."""
    log_dir = BASE / variant / f"{variant}_from_pretrained"
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        print(f"  [AVISO] Nenhum log em {log_dir}")
        return {}

    data = {}
    pattern = re.compile(r"Iter:\s*(\d+).*?psnr:\s*([\d.]+).*?ssim:\s*([\d.]+)", re.IGNORECASE)
    for log_path in logs:
        for line in log_path.read_text(errors="ignore").splitlines():
            m = pattern.search(line)
            if m:
                it = int(m.group(1))
                data[it] = {'psnr': float(m.group(2)), 'ssim': float(m.group(3))}
    return data


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_ssim(img, gt):
    return ssim(img, gt, channel_axis=-1, data_range=1.0)

def compute_cpsnr(img, gt):
    best = -np.inf
    for c in range(img.shape[2]):
        ic = img[:, :, c].flatten().astype(np.float64)
        gc = gt[:, :, c].flatten().astype(np.float64)
        A = np.vstack([ic, np.ones_like(ic)]).T
        a, b = np.linalg.lstsq(A, gc, rcond=None)[0]
        ic_adj = np.clip(a * ic + b, 0, 1)
        mse = np.mean((ic_adj - gc) ** 2)
        if mse < 1e-10:
            return 100.0
        best = max(best, 10 * np.log10(1.0 / mse))
    return best

def radial_psd(img_gray):
    f = np.fft.fftshift(np.fft.fft2(img_gray))
    psd2d = np.abs(f) ** 2
    H, W = psd2d.shape
    cy, cx = H // 2, W // 2
    r = np.sqrt((np.ogrid[:H, :W][1] - cx) ** 2 +
                (np.ogrid[:H, :W][0] - cy) ** 2).astype(int)
    r_max = min(cy, cx)
    return np.array([psd2d[r == ri].mean() if (r == ri).any() else 0.0
                     for ri in range(r_max)])

def psd_l2(img, gt):
    total = 0.0
    for c in range(img.shape[2]):
        p_img = np.clip(radial_psd(img[:, :, c]), 1e-10, None)
        p_gt  = np.clip(radial_psd(gt[:, :, c]),  1e-10, None)
        total += np.mean((np.log10(p_img) - np.log10(p_gt)) ** 2)
    return total / img.shape[2]


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_variant(variant, it):
    viz_dir = BASE / variant / f"{variant}_from_pretrained" / "visualization"
    if not viz_dir.exists():
        print(f"  [ERRO] Não encontrado: {viz_dir}")
        return [], []

    sr_list, gt_list = [], []
    for chip_dir in sorted(viz_dir.iterdir()):
        if not chip_dir.is_dir():
            continue
        chip_id = chip_dir.name
        sr_path = chip_dir / f"{chip_id}_{it}.png"
        gt_path = chip_dir / f"{chip_id}_{it}_gt.png"
        if sr_path.exists() and gt_path.exists():
            sr_list.append(io.imread(str(sr_path)).astype(np.float32) / 255.0)
            gt_list.append(io.imread(str(gt_path)).astype(np.float32) / 255.0)

    print(f"  {variant}: {len(sr_list)} chips (iter {it})")
    return sr_list, gt_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}
psd_curves = {}
log_curves = {}
gt_radials_saved = False
gt_psd_accum = []

for variant in VARIANTS:
    print(f"\n[{variant}]")
    log_curves[variant] = parse_log(variant)
    sr_imgs, gt_imgs = load_variant(variant, ITER)
    if not sr_imgs:
        continue

    ssim_vals, cpsnr_vals, psd_vals, radials = [], [], [], []
    for sr, gt in zip(sr_imgs, gt_imgs):
        if sr.ndim == 2:
            sr = np.stack([sr] * 3, axis=-1)
        if gt.ndim == 2:
            gt = np.stack([gt] * 3, axis=-1)
        ssim_vals.append(compute_ssim(sr, gt))
        cpsnr_vals.append(compute_cpsnr(sr, gt))
        psd_vals.append(psd_l2(sr, gt))
        for c in range(sr.shape[2]):
            radials.append(radial_psd(sr[:, :, c]))
        if not gt_radials_saved:
            for c in range(gt.shape[2]):
                gt_psd_accum.append(radial_psd(gt[:, :, c]))

    gt_radials_saved = True
    results[variant] = {
        'ssim':   np.mean(ssim_vals),
        'cpsnr':  np.mean(cpsnr_vals),
        'psd_l2': np.mean(psd_vals),
        'n':      len(sr_imgs),
    }
    psd_curves[variant] = np.mean(radials, axis=0)
    r = results[variant]
    print(f"  SSIM={r['ssim']:.4f} | cPSNR={r['cpsnr']:.2f} dB | PSD_L2={r['psd_l2']:.4f}")

psd_curves['GT'] = np.mean(gt_psd_accum, axis=0)

# ------------------------------------------------------------------
# Tabela
# ------------------------------------------------------------------
table_path = OUT_DIR / "results_table.md"
with open(table_path, "w") as f:
    f.write("# Resultados — Ablation Study Focal Frequency Loss\n\n")
    f.write(f"> Avaliação no iter final ({ITER}) — mesmo orçamento para todos os variantes.\n\n")
    f.write("| Variante | N | SSIM ↑ | cPSNR (dB) ↑ | PSD_L2 ↓ |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for name, r in results.items():
        f.write(f"| {name} | {r['n']} | {r['ssim']:.4f} | {r['cpsnr']:.2f} | {r['psd_l2']:.4f} |\n")
print(f"\nTabela: {table_path}")

# ------------------------------------------------------------------
# Curvas de convergência (do log)
# ------------------------------------------------------------------
colors = plt.cm.tab10.colors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for i, variant in enumerate(VARIANTS):
    data = log_curves.get(variant, {})
    if not data:
        continue
    iters = sorted(data.keys())
    ax1.plot(iters, [data[it]['psnr'] for it in iters],
             label=variant, color=colors[i], marker='o', markersize=3)
    ax2.plot(iters, [data[it]['ssim'] for it in iters],
             label=variant, color=colors[i], marker='o', markersize=3)

ax1.set_title("PSNR de validação por iter"); ax1.set_xlabel("Iter"); ax1.set_ylabel("PSNR (dB)")
ax2.set_title("SSIM de validação por iter"); ax2.set_xlabel("Iter"); ax2.set_ylabel("SSIM")
for ax in (ax1, ax2):
    ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle("Curvas de convergência — FFL Ablation")
fig.tight_layout()
conv_path = OUT_DIR / "convergence_curves.png"
fig.savefig(conv_path, dpi=150); plt.close()
print(f"Convergência: {conv_path}")

# ------------------------------------------------------------------
# Figura PSD
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for i, (name, curve) in enumerate(psd_curves.items()):
    freqs = np.arange(len(curve)) / len(curve)
    ls = '-' if name == 'GT' else ['--', '-.', ':'][i % 3]
    lw = 2.5 if name == 'GT' else 1.5
    ax.semilogy(freqs, curve + 1e-10, label=name, color=colors[i], linestyle=ls, linewidth=lw)
ax.set_xlabel("Frequência normalizada (ciclos/pixel)")
ax.set_ylabel("PSD (log)")
ax.set_title("Comparação de Espectros de Potência — FFL Ablation")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
psd_path = OUT_DIR / "psd_comparison.png"
fig.savefig(psd_path, dpi=150); plt.close()
print(f"PSD: {psd_path}")

# ------------------------------------------------------------------
# Figura barras
# ------------------------------------------------------------------
names_list = list(results.keys())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(names_list, [results[n]['ssim']   for n in names_list], color=colors[:len(names_list)])
ax1.set_title("SSIM ↑"); ax1.set_ylim(0, 1)
for i, n in enumerate(names_list):
    ax1.text(i, results[n]['ssim'] + 0.01, f"{results[n]['ssim']:.4f}", ha='center', fontsize=9)
ax2.bar(names_list, [results[n]['psd_l2'] for n in names_list], color=colors[:len(names_list)])
ax2.set_title("PSD_L2 ↓")
for i, n in enumerate(names_list):
    ax2.text(i, results[n]['psd_l2'] + 0.001, f"{results[n]['psd_l2']:.4f}", ha='center', fontsize=9)
fig.suptitle("Ablation Study — Focal Frequency Loss no SATLAS")
fig.tight_layout()
bar_path = OUT_DIR / "metrics_comparison.png"
fig.savefig(bar_path, dpi=150); plt.close()
print(f"Barras: {bar_path}")

print("\nConcluído! Arquivos em:", OUT_DIR)
