#!/bin/bash
# =============================================================
# rodar_infer_ablation.sh
# Inferência Planet → SR com os 3 pesos treinados no zero_shot
# (baseline / ffl_w01 / ffl_w10)
#
# Não usa Docker — roda direto no venv do servidor.
# Gera um GeoTIF de saída por variante (imagem completa, não chips).
#
# Uso:
#   bash rodar_infer_ablation.sh
#
# Para rodar em background:
#   nohup bash rodar_infer_ablation.sh > ~/infer_ablation.log 2>&1 &
#   tail -f ~/infer_ablation.log
# =============================================================

set -e

# =============================================================
# CAMINHOS FIXOS DO SERVIDOR — ajuste se necessário
# =============================================================
VENV=/dados/user_data/marcel/cmp570_final/venv
SATLAS=/dados/user_data/marcel/cmp570_final/satlas-super-resolution
TRABALHO=/dados/user_data/marcel/cmp570_final/trabalho_final
THIS_DIR="$TRABALHO/scripts/inferencia_ablation"

# Pasta com os 16 TIFs Planet originais
PLANET_DIR="$TRABALHO/data/planet"

# Stack multi-banda gerado pelo stack_sr_planet.py (criado antes da inferência)
PLANET_STACK="$TRABALHO/data/planet_stack.tif"

# Pasta de saída dos GeoTIFs SR
OUTPUT_DIR="$TRABALHO/results_infer"

# Config template (peso substituído por sed para cada variante)
CONFIG_TEMPLATE="$THIS_DIR/config_infer_planet.yml"

# Pipeline de inferência
PIPELINE="$THIS_DIR/pipeline_sr_multigpu.py"

# =============================================================
# PESOS TREINADOS — INFORME O CAMINHO DE CADA VARIANTE AQUI
# =============================================================
WEIGHTS_SATLAS_PRETRAINED="/dados/user_data/marcel/cmp570_final/satlas-super-resolution/pretrained_models/esrgan_16S2.pth"
WEIGHTS_BASELINE="/dados/user_data/marcel/cmp570_final/trabalho_final/zero_shot/results/baseline/baseline_from_pretrained/models/net_g_20000.pth"
WEIGHTS_FFL_W01="/dados/user_data/marcel/cmp570_final/trabalho_final/zero_shot/results/ffl_w01/ffl_w01_from_pretrained/models/net_g_20000.pth"
WEIGHTS_FFL_W10="/dados/user_data/marcel/cmp570_final/trabalho_final/zero_shot/results/ffl_w10/ffl_w10_from_pretrained/models/net_g_20000.pth"
# =============================================================

# Ativar venv
source "$VENV/bin/activate"
export PYTHONPATH="$SATLAS:/usr/lib/python3/dist-packages"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo " Inferência Ablation Study — CMP570 Trabalho Final"
echo "============================================================"
echo " SATLAS:   $SATLAS"
echo " Planet:   $PLANET_DIR"
echo " Stack:    $PLANET_STACK"
echo " Saídas:   $OUTPUT_DIR"
echo ""

# =============================================================
# PASSO 1: Gerar o stack Planet (16 TIFs → 48 bandas)
# =============================================================
if [ -f "$PLANET_STACK" ]; then
    echo "[1/4] Stack já existe: $PLANET_STACK"
    echo "      Para regenerar, delete o arquivo e rode novamente."
else
    echo "[1/4] Gerando stack Planet (16 TIFs → 48 bandas)..."
    python "$THIS_DIR/stack_sr_planet.py" \
        --input_dir "$PLANET_DIR" \
        --output    "$PLANET_STACK" \
        --n_expected 16
    echo "      Stack criado: $PLANET_STACK"
fi
echo ""

# =============================================================
# PASSO 2: Inferência com cada variante
# =============================================================
run_variant() {
    local NAME="$1"
    local WEIGHTS="$2"

    echo "------------------------------------------------------------"
    echo " Variante: $NAME"
    echo " Pesos:    $WEIGHTS"

    # Verificar se o peso existe
    if [ ! -f "$WEIGHTS" ]; then
        echo " [ERRO] Arquivo de pesos não encontrado: $WEIGHTS"
        echo "        Verifique o caminho na seção PESOS TREINADOS do script."
        return 1
    fi

    local OUTPUT="$OUTPUT_DIR/${NAME}_sr.tif"

    if [ -f "$OUTPUT" ]; then
        echo " [SKIP] Saída já existe: $OUTPUT"
        echo ""
        return 0
    fi

    local CONFIG_TMP="/tmp/config_infer_${NAME}.yml"

    # Substituir placeholder pelo caminho real do peso
    sed "s|__WEIGHTS_PATH__|${WEIGHTS}|g" "$CONFIG_TEMPLATE" > "$CONFIG_TMP"

    echo " Saída:    $OUTPUT"
    echo ""

    python "$PIPELINE" \
        -opt        "$CONFIG_TMP" \
        -input      "$PLANET_STACK" \
        -output     "$OUTPUT" \
        -imagens    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
        -tile_size  32 \
        -overlap    0.5 \
        -batch_size 512 \
        -ponderacao linear \
        -gpus       0

    rm -f "$CONFIG_TMP"
    echo " ✓ $NAME concluído → $OUTPUT"
    echo ""
}

echo "[2/5] Rodando satlas_pretrained (referência zero-shot)..."
run_variant "satlas_pretrained" "$WEIGHTS_SATLAS_PRETRAINED"

echo "[3/5] Rodando baseline..."
run_variant "baseline" "$WEIGHTS_BASELINE"

echo "[4/5] Rodando ffl_w01..."
run_variant "ffl_w01" "$WEIGHTS_FFL_W01"

echo "[5/5] Rodando ffl_w10..."
run_variant "ffl_w10" "$WEIGHTS_FFL_W10"

# =============================================================
# Resumo
# =============================================================
echo "============================================================"
echo " Concluído! GeoTIFs gerados em: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"/*.tif 2>/dev/null || echo " (nenhum arquivo encontrado)"
echo "============================================================"
