#!/bin/bash
# run_ablation.sh — Roda o ablation study do Projeto Final CMP570
#
# Uso: bash scripts/run_ablation.sh
#
# Cada variante parte do checkpoint 9000 do CP06 e treina mais 9k iters.
# Os resultados ficam em trabalho_final/results/<variante>/

set -e

SATLAS=/home/marcel/Desktop/allenai/satlas-super-resolution
CONFIGS=/home/marcel/Desktop/cmp570/trabalhos/trabalho_final/configs

cd "$SATLAS"

echo "=== Variante A: FFL w=0.1 ==="
python ssr/train.py --opt "$CONFIGS/ffl_w01_planet_finetune.yml"

echo "=== Variante B: FFL w=1.0 ==="
python ssr/train.py --opt "$CONFIGS/ffl_w10_planet_finetune.yml"

echo "=== Ablation concluído. Rode evaluate_ffl.py para gerar a tabela. ==="
