# Trabalho Final CMP570 — Focal Frequency Loss + SATLAS

**Disciplina:** CMP570 — Fotografia Computacional (UFRGS 2026/1)
**Professor:** Manuel Menezes de Oliveira Neto
**Aluno:** Marcel Fernandes Gomes
**Orientador:** Prof. Dr. Eduardo Simões Lopes Gastal

---

## Tema

**"Superresolução no Domínio de Frequências: Fine-tuning do SATLAS com Focal
Frequency Loss para Correção do Gap Espectral Planet–Aerofoto"**

*Fine-tuning* do SATLAS (ESRGAN pré-treinado Sentinel-2 → NAIP, 4×) com pares locais
Planet → aerofoto, adicionando a **Focal Frequency Loss** (Jiang et al., ICCV 2021)
como *loss* auxiliar no domínio espectral. *Ablation study* com 3 variantes:
baseline (sem FFL), FFL λ=0,1, FFL λ=1,0.

---

## Estrutura do repositório (pacote de reprodutibilidade)

```
cmp570_trabalho_final/
├── CLAUDE.md                     # este arquivo
├── README.md                     # visão geral + resultado principal
├── revisao_apresentacao.md       # perguntas e respostas para a banca
├── relatorio/                    # relatório final (relatorio.tex + PDF)
├── apresentacao/                 # slides Beamer (apresentacao.tex + PDF)
├── proposta/                     # proposta aprovada (proposta.tex + PDF)
├── codigo/
│   ├── focal_frequency_loss.py   # implementação standalone da FFL + demo
│   ├── evaluate_ffl.py           # avaliação original (usada pelo run_pipeline)
│   ├── run_eval.py               # avaliação simplificada (lê visualization/ direto)
│   ├── configs/                  # YAMLs de finetune (raiz) + zero_shot/ (a partir do pretrained)
│   │   └── zero_shot/linux/      # YAMLs do servidor — rodada OFICIAL (20k iters)
│   └── scripts/                  # prepare_chips, infer_chips, run_ablation, run_pipeline, ...
├── resultados/
│   ├── evaluation/               # resultado OFICIAL — Linux 20.000 iters (1.476 chips)
│   └── zero_shot_5k/             # reprodução secundária — Windows 5.000 iters
└── dados/
    ├── planet/                   # 16 TIFs PlanetScope de amostra (entrada LR rodada 1)
    └── PROVENIENCIA.md           # o que está/não está versionado e como obter o resto
```

Os rasters grandes (aerofoto GT, saídas SR) e pesos não são versionados — ver
`dados/PROVENIENCIA.md`.

---

## Fundamentação (conexão com CMP570)

| Tópico | Papel no projeto |
|---|---|
| Aulas 12–13 — Fourier / DFT | `torch.fft.fft2` no núcleo da FFL |
| Aula 14 — Aliasing / Nyquist | Motivação do gap espectral Planet/Sentinel |
| Aula 15 — Teorema da Convolução | Relação PSF ↔ domínio de frequência |
| Semanas 10–11 — Filtro de Wiener | Baseline clássico de comparação conceitual |

A FFL opera no espaço de Fourier: calcula a DFT 2D (`norm='ortho'`, preserva energia)
da saída e do GT e penaliza o erro componente a componente, com um peso adaptativo
`w[u,v] = |F̂−F|^α / E[|F̂−F|^α]` (`.detach()`, fora do *backprop*) que cresce nas
frequências onde o modelo persiste errando — tipicamente as altas, que o L1 suaviza.

---

## Ablation study

| Variante | Losses | λ_FFL | α | Ponto de partida |
|---|---|---|---|---|
| **baseline** | L1 + Perceptual + GAN | — | — | esrgan_16S2.pth (SATLAS pretrained) |
| **ffl_w01** | L1 + Perceptual + GAN + FFL | 0,1 | 1,0 | esrgan_16S2.pth |
| **ffl_w10** | L1 + Perceptual + GAN + FFL | 1,0 | 1,0 | esrgan_16S2.pth |

**Config oficial (Linux):** `total_iter: 20000`, `batch_size_per_gpu: 4`,
`n_s2_images: 16`, `lr: 1e-4` (decai em 8k/16k), checkpoints a cada 1000 iters,
`CUDA_VISIBLE_DEVICES=0`. GPU Tesla V100-PCIE-32GB.
**Critério de seleção de checkpoint:** iter final (20000) fixo para todas as
variantes (mesmo orçamento de treino — evita viés de "peak por métrica").

---

## Resultados

### Rodada OFICIAL — Linux, 20.000 iters, 1.476 chips (`resultados/evaluation/`)

| Variante | SSIM ↑ | cPSNR (dB) ↑ | PSD_L2 ↓ |
|---|---:|---:|---:|
| baseline | 0,6352 | 29,09 | 0,2574 |
| ffl_w01 (λ=0,1) | 0,6368 | 29,08 | 0,2638 |
| ffl_w10 (λ=1,0) | 0,6291 | 29,03 | **0,2389** |

**Achados:**
- FFL λ=1,0 reduz PSD_L2 em **7,2%** vs baseline — a FFL recupera frequências altas.
- Custo de SSIM pequeno (Δ = −0,006) — trade-off favorável p/ fidelidade espectral.
- FFL λ=0,1 insuficiente (peso baixo não ativa o mecanismo focal).

### ⚠️ Reprodução secundária — Windows, 5.000 iters (`resultados/zero_shot_5k/`)

| Variante | SSIM ↑ | cPSNR (dB) ↑ | PSD_L2 ↓ |
|---|---:|---:|---:|
| baseline | 0,5968 | 28,22 | 0,3974 |
| ffl_w01 (λ=0,1) | 0,6000 | 28,26 | 0,4109 |
| ffl_w10 (λ=1,0) | 0,5923 | 28,20 | 0,4029 |

**A reprodução a 5k iters NÃO confirmou o ganho da FFL** (PSD_L2 da FFL λ=1,0 ficou
*acima* do baseline). Interpretação: o mecanismo focal da FFL precisa de iterações
para acumular peso nas frequências difíceis; com 1/4 do orçamento de treino o efeito
não se manifesta (e o modelo todo está pior — SSIM ~0,59 vs ~0,63). O relatório e os
slides reportam a **rodada oficial de 20k iters**. Para replicar, treinar 20.000 iters
(`codigo/configs/zero_shot/linux/`).

**Histórico:** a primeira rodada foi no servidor Linux (checkpoint CP06, máquina depois
perdida). A versão Windows reescreveu o pipeline para partir direto do `esrgan_16S2.pth`
público — ver `codigo/scripts/run_pipeline.ps1` e os configs `zero_shot/`.

---

## Métricas de avaliação

| Métrica | Descrição | Sentido |
|---|---|---|
| **SSIM** | Similaridade estrutural | maior melhor |
| **cPSNR** | PSNR com ajuste afim por canal (tolerante a offset cross-sensor) | maior melhor |
| **PSD_L2** | Distância L2 no log do PSD radial médio vs GT | menor melhor |

`run_eval.py` lê as imagens salvas pelo BasicSR durante o treino
(`visualization/{chip}/{chip}_20000.png`) e calcula as 3 métricas para as variantes.

---

## Patches no SATLAS (aplicar antes de treinar)

No repositório `satlas-super-resolution`:

**`ssr/losses/basic_loss.py`** — classe `FocalFrequencyLoss` registrada via
`@LOSS_REGISTRY.register()` (espelha `codigo/focal_frequency_loss.py`).

**`ssr/models/ssr_esrgan_model.py`** — suporte a `ffl_opt` no YAML (opcional,
retrocompatível): inicializa `self.cri_ffl` e aplica em `optimize_parameters`.

**`ssr/data/s2-naip_dataset.py`** — (Windows) correção de path: `Path().parent.name`.

Verificar:
```bash
grep -n "FocalFrequencyLoss" .../ssr/losses/basic_loss.py
grep -n "ffl_opt\|cri_ffl"   .../ssr/models/ssr_esrgan_model.py
```

---

## Dados

- **Entrada (LR):** stacks PlanetScope TCI (RGB) reamostrados para ~10 m/px, chips
  32×32 px. Amostra da rodada 1 em `dados/planet/check03_*.tif` (16 cenas mensais).
- **Ground truth (HR):** aerofoto ~0,35–0,5 m/px (Convênio Estado RS / Exército),
  chips 128×128 px (escala 4×). **Não versionada** (ver `dados/PROVENIENCIA.md`).

---

## Referências

- **FFL:** L. Jiang, B. Dai, W. Wu, C. C. Loy. *Focal Frequency Loss for Image
  Reconstruction and Synthesis.* ICCV 2021, pp. 13919–13929.
  <https://arxiv.org/abs/2012.12821>
- **SATLAS-SR:** P. Wolters, F. Bastani, A. Kembhavi. *Zooming Out on Zooming In:
  Advancing Super-Resolution for Remote Sensing.* arXiv:2311.18082, 2023.
  <https://github.com/allenai/satlas-super-resolution>
- **ESRGAN:** X. Wang et al. *ESRGAN: Enhanced Super-Resolution GANs.* ECCVW 2018.
- **SSIM:** Z. Wang et al. *Image Quality Assessment: From Error Visibility to
  Structural Similarity.* IEEE TIP 13(4), 2004.
- **BasicSR:** X. Wang et al. <https://github.com/XPixelGroup/BasicSR>
