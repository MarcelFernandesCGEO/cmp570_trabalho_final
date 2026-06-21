#!/bin/bash
# run_baseline_inference.sh
# Gera imagens SR do baseline (CP06 sem FFL) para uso em evaluate_ffl.py.
# Deve ser rodado a partir do diretório satlas-super-resolution/.
#
# Uso:
#   cd /home/marcel/Desktop/allenai/satlas-super-resolution
#   bash /home/marcel/Desktop/cmp570/trabalhos/trabalho_final/scripts/run_baseline_inference.sh

set -e

SATLAS_DIR="/home/marcel/Desktop/allenai/satlas-super-resolution"
CONFIG="/home/marcel/Desktop/cmp570/trabalhos/trabalho_final/configs/baseline_test.yml"

echo "=== Inferência baseline (CP06 net_g_9000.pth) ==="
echo "Config: $CONFIG"
echo "Saída:  trabalho_final/results/baseline/"
echo

cd "$SATLAS_DIR"
python ssr/test.py -opt "$CONFIG"

echo
echo "=== Concluído. Imagens SR em: ==="
echo "    trabalho_final/results/baseline/visualization/"
