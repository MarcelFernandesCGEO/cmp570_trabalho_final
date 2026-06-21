# Superresolução no Domínio de Frequências — FFL + SATLAS

Trabalho final da disciplina **CMP570 — Fotografia Computacional** (PPGC/UFRGS, 2026/1):
*fine-tuning* do gerador de super-resolução do **SATLAS** (ESRGAN / `SSR_RRDBNet`,
Sentinel-2 → NAIP, 4×) com pares locais **Planet → aerofoto**, adicionando a
**Focal Frequency Loss** (Jiang et al., ICCV 2021) como *loss* auxiliar no domínio
espectral. *Ablation study* de 3 variantes: baseline (sem FFL), FFL λ=0,1 e FFL λ=1,0.

Autor: **Marcel Fernandes Gomes** · marcelgfernandes@gmail.com
Prof. Manuel M. de Oliveira Neto · Orientador: Prof. Dr. Eduardo S. L. Gastal

## O que há neste repositório

Este é o **pacote de reprodutibilidade**. Os rasters grandes (aerofoto GT ~844 MB,
saídas SR ~400 MB) e os pesos não são versionados — apenas os TIFs Planet de amostra,
o código, as configs e todos os resultados. Ver [`dados/PROVENIENCIA.md`](dados/PROVENIENCIA.md).

```
relatorio/        relatório final (relatorio.tex + PDF)
apresentacao/     slides Beamer (apresentacao.tex + PDF)
proposta/         proposta aprovada (proposta.tex + PDF)
codigo/           focal_frequency_loss.py, evaluate_ffl.py, run_eval.py
codigo/configs/   YAMLs do ablation (raiz = finetune; zero_shot/ = a partir do pretrained)
codigo/scripts/   pipeline (prepare_chips, infer, run_ablation, run_pipeline, ...)
resultados/evaluation/    resultado OFICIAL — rodada Linux 20.000 iters (1.476 chips)
resultados/zero_shot_5k/  reprodução secundária — Windows 5.000 iters (ver nota abaixo)
dados/planet/     16 TIFs PlanetScope de amostra (entrada LR da rodada 1)
CLAUDE.md         documentação técnica detalhada
revisao_apresentacao.md   perguntas e respostas preparadas para a banca
```

## Resultado principal (rodada oficial — 20.000 iters, 1.476 chips)

| Variante | SSIM ↑ | cPSNR (dB) ↑ | PSD_L2 ↓ |
|---|---:|---:|---:|
| baseline | 0,6352 | 29,09 | 0,2574 |
| FFL λ=0,1 | **0,6368** | 29,08 | 0,2638 |
| FFL λ=1,0 | 0,6291 | 29,03 | **0,2389** |

FFL λ=1,0 reduz o **PSD_L2 em 7,2%** vs. baseline (recupera frequências altas), com
custo desprezível de SSIM (Δ = −0,006). FFL λ=0,1 é insuficiente para o gap Planet→aerofoto.

> ⚠️ **Nota de reprodutibilidade.** Os números acima vêm da rodada oficial executada
> no servidor Linux (20.000 iters; configs em `codigo/configs/zero_shot/linux/`).
> A reprodução posterior no Windows (`resultados/zero_shot_5k/`) usou **apenas 5.000
> iters** e **não** reproduziu o ganho da FFL (PSD_L2 da FFL λ=1,0 ficou ligeiramente
> acima do baseline). A diferença é atribuída ao **orçamento de treino** (5k × 20k):
> o mecanismo focal precisa de iterações para atuar. Para replicar o resultado
> oficial, treinar 20.000 iters. Discussão em [`CLAUDE.md`](CLAUDE.md).

## Reprodução (resumo)

1. Baixar `esrgan_16S2.pth` (SATLAS) e aplicar os 2 patches FFL (ver `CLAUDE.md`).
2. Gerar chips com `codigo/scripts/prepare_chips.py`.
3. Treinar as 3 variantes — `codigo/configs/zero_shot/linux/*.yml` (20.000 iters).
4. Avaliar com `codigo/run_eval.py` → `resultados/evaluation/`.

Detalhes de ambiente, parâmetros e armadilhas em [`CLAUDE.md`](CLAUDE.md).

## Referências

- L. Jiang, B. Dai, W. Wu, C. C. Loy. *Focal Frequency Loss for Image Reconstruction
  and Synthesis*. ICCV 2021. <https://arxiv.org/abs/2012.12821>
- P. Wolters, F. Bastani, A. Kembhavi. *Zooming Out on Zooming In: Advancing
  Super-Resolution for Remote Sensing*. arXiv:2311.18082, 2023.
  <https://github.com/allenai/satlas-super-resolution>
