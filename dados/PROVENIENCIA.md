# Dados — proveniência e o que está (ou não) versionado

Este repositório é o **pacote de reprodutibilidade** do trabalho. Os rasters grandes
não são versionados (limite de 100 MB/arquivo do GitHub e/ou restrição de
redistribuição). Abaixo, o que está incluído e como obter o resto.

## Versionado neste repositório

| Item | Caminho | Tamanho |
|---|---|---|
| TIFs PlanetScope de amostra (rodada 1, série temporal) | `dados/planet/check03_*.tif` | ~3,5 MB × 16 = ~54 MB |

Os 16 TIFs `check03_AAAAMM.tif` são cenas mensais PlanetScope TCI (RGB) da área de
diagnóstico CP03, reamostradas e usadas como entrada de baixa resolução (LR) da
rodada 1. Cada chip de entrada empilha esses frames (formato `[T×32, 32]`).

## NÃO versionado (obter/regenerar)

| Item | Origem / como obter |
|---|---|
| **Aerofoto GT** (`gt.tif`, ~99 MB rodada 1 / ~844 MB rodada 2) | Aerofoto ~0,35–0,5 m/px do Convênio Estado do RS / Exército Brasileiro (IEDE-RS / 1º Centro de Geoinformação). Não redistribuível. |
| **Saídas SR** (`baseline_sr.tif`, `ffl_w01_sr.tif`, `ffl_w10_sr.tif`, ~400 MB cada) | Regeneradas por inferência (`codigo/scripts/inferencia_ablation/`). |
| **Chips** (`planet_chips/`, `gt_chips/`) | Gerados por `codigo/scripts/prepare_chips.py` a partir dos TIFs Planet + GT. |
| **Pesos SATLAS** `esrgan_16S2.pth` | Público: <https://github.com/allenai/satlas-super-resolution> (README → download). |
| **Checkpoint CP06** (`9000.state`, `net_g_9000.pth`) | Perdido com a máquina Linux original; a rodada final parte do `esrgan_16S2.pth` público. |

## Reprodução resumida

1. Baixar `esrgan_16S2.pth` (SATLAS) e aplicar os 2 patches FFL ao repo
   `satlas-super-resolution` (ver `../CLAUDE.md` → "Patches no SATLAS").
2. Gerar chips: `codigo/scripts/prepare_chips.py` (Planet `dados/planet/` + GT).
3. Treinar as 3 variantes: configs em `codigo/configs/zero_shot/` (Windows) ou
   `codigo/configs/zero_shot/linux/` (servidor, 20k iters — rodada oficial).
4. Avaliar: `codigo/run_eval.py` (lê as `visualization/` do treino) →
   `resultados/evaluation/`.

Detalhes completos em [`../CLAUDE.md`](../CLAUDE.md).
