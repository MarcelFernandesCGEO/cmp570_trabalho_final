"""
focal_frequency_loss.py — Implementação standalone da Focal Frequency Loss
Projeto Final CMP570 — Fotografia Computacional — UFRGS 2026/1

Referência:
    Jiang, L., Dai, B., Wu, W., Loy, C.C. (2021).
    "Focal Frequency Loss for Image Reconstruction and Synthesis."
    ICCV 2021. https://arxiv.org/abs/2012.12821

Contexto do projeto:
    Diagnóstico (CP03–CP04): imagens PlanetScope reamostradas para 10 m têm
    ~100× mais energia em alta frequência do que a Sentinel-2 na mesma grade.
    O SATLAS, treinado para *criar* alta frequência ausente, se confunde quando
    ela já está presente — e de forma incorreta. O fine-tuning do CP06 reduz
    parte desse gap, mas usa apenas losses no espaço de pixel (L1, VGG, GAN)
    que não penalizam explicitamente o erro espectral.

    A Focal Frequency Loss opera diretamente no espaço de Fourier: calcula a
    FFT 2D de saída e GT, e penaliza o erro componente a componente, com peso
    adaptativo que cresce nas frequências onde o modelo persiste errando.

Conexão com CMP570 (Aulas 12–15):
    - torch.fft.fft2 implementa a DFT 2D (Aula 12/15)
    - norm='ortho' garante preservação de energia (unitária)
    - view_as_real separa Re e Im do espectro complexo
    - O mecanismo focal pondera frequências altas ↔ relação com aliasing (Aula 14)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss.

    Args:
        loss_weight: escalar λ que pondera a FFL no total da loss de treino.
        alpha: expoente do mecanismo focal (α=1 → linear; α>1 → mais agressivo).
    """

    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.weight: torch.Tensor | None = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] — saída do gerador em [0, 1]
            target: [B, C, H, W] — ground truth em [0, 1]
        Returns:
            Escalar (loss ponderada).
        """
        # --- Passo 1: FFT 2D ---
        # Resultado: tensor complexo [B, C, H, W]
        # norm='ortho' → ||FFT(x)||² = ||x||²  (preservação de energia)
        pred_freq   = torch.fft.fft2(pred,   norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')

        # --- Passo 2: erro complexo → Re e Im separados ---
        # diff_real: [B, C, H, W, 2]  última dim = [parte real, parte imaginária]
        diff = pred_freq - target_freq
        diff_real = torch.view_as_real(diff)

        # --- Passo 3: pesos uniformes na primeira iteração ---
        if self.weight is None or self.weight.shape != diff_real.shape:
            self.weight = torch.ones_like(diff_real)

        # --- Passo 4: loss ponderada ---
        # Frequências com w(u,v) alto já foram identificadas como difíceis.
        # .detach() → os pesos não participam do grafo computacional
        loss = torch.mean(self.weight.detach() * diff_real ** 2)

        # --- Passo 5: atualização focal ---
        # Frequências onde |diff| foi grande recebem peso maior na próxima iter.
        with torch.no_grad():
            new_weight = diff_real.detach().abs().pow(self.alpha)
            new_weight = new_weight / (new_weight.mean() + 1e-8)
            self.weight = new_weight

        return loss * self.loss_weight


# ---------------------------------------------------------------------------
# Função de diagnóstico: visualiza o peso w(u,v) após N iterações
# ---------------------------------------------------------------------------

def visualize_focal_weights(pred_list, target_list, alpha=1.0, save_path=None):
    """
    Simula N iterações da FFL e visualiza como os pesos w(u,v) evoluem.

    Args:
        pred_list:   lista de tensores [1, 3, H, W] (saídas do gerador)
        target_list: lista de tensores [1, 3, H, W] (ground truths)
        alpha:       expoente focal
        save_path:   se fornecido, salva a figura em disco
    """
    loss_fn = FocalFrequencyLoss(alpha=alpha)
    weight_snapshots = []

    for i, (pred, target) in enumerate(zip(pred_list, target_list)):
        _ = loss_fn(pred, target)
        if i in [0, len(pred_list) // 4, len(pred_list) // 2, len(pred_list) - 1]:
            w = loss_fn.weight.cpu().numpy()
            # média sobre batch, canais e Re/Im → [H, W]
            w_mean = w.mean(axis=(0, 1, 4))
            weight_snapshots.append((i, np.fft.fftshift(w_mean)))

    fig, axes = plt.subplots(1, len(weight_snapshots), figsize=(4 * len(weight_snapshots), 4))
    if len(weight_snapshots) == 1:
        axes = [axes]
    for ax, (it, w) in zip(axes, weight_snapshots):
        im = ax.imshow(np.log10(w + 1e-8), cmap='hot')
        ax.set_title(f"Iter {it}\nw(u,v) em log₁₀")
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"Evolução dos pesos focais (α={alpha})\n"
                 "Centro = baixa freq | Bordas = alta freq", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figura salva em {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Demo rápido (execução direta)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 128, 128

    # Simula saída SR e GT com gap em alta frequência
    gt   = torch.rand(B, C, H, W)
    pred = gt + 0.1 * torch.randn(B, C, H, W)  # pequeno ruído

    ffl = FocalFrequencyLoss(loss_weight=0.1, alpha=1.0)

    print("=== Demo: Focal Frequency Loss ===")
    for i in range(5):
        loss = ffl(pred, gt)
        print(f"Iter {i+1}: loss = {loss.item():.6f}")

    print("\nForma do tensor de pesos w(u,v):", ffl.weight.shape)
    print("Peso médio nas primeiras 10 freq:", ffl.weight[0, 0, :10, :10, 0].mean().item())
