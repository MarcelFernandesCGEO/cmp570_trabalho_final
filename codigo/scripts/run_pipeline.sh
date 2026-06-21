#!/usr/bin/env bash
# run_pipeline.sh — Pipeline completo no Linux: prepare -> train -> infer -> evaluate
#
# Uso basico (dados e configs padrao):
#   bash run_pipeline.sh
#
# Com novos dados:
#   bash run_pipeline.sh --planet_dir /dados/planet --gt_tif /dados/gt.tif --out_data /dados/chips
#
# Flags de skip:
#   --skip_prepare   pula prepare_chips (chips ja existem)
#   --skip_train     pula treino (so infer + evaluate)
#   --skip_infer     pula inferencia (so evaluate)
#   --checkpoint N   qual checkpoint usar (default: 20000)

set -e

# -----------------------------------------------------------------------
# Paths fixos do servidor
# -----------------------------------------------------------------------
BASE=/dados/user_data/marcel/cmp570_final
SATLAS=$BASE/satlas-super-resolution
TRABALHO=$BASE/trabalho_final
SCRIPTS=$TRABALHO/zero_shot/scripts
CONFIGS=$TRABALHO/zero_shot/configs/linux
RESULTS=$TRABALHO/zero_shot/results
EVALSCRIPT=$TRABALHO/evaluate_ffl.py

# Defaults
PLANET_DIR=$TRABALHO/data/planet
GT_TIF=$TRABALHO/data/gt/gt.tif
OUT_DATA=$BASE/zero_shot_data
CHECKPOINT=20000
SKIP_PREPARE=0
SKIP_TRAIN=0
SKIP_INFER=0

# -----------------------------------------------------------------------
# Parsear argumentos
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --planet_dir)  PLANET_DIR="$2"; shift 2 ;;
        --gt_tif)      GT_TIF="$2";     shift 2 ;;
        --out_data)    OUT_DATA="$2";   shift 2 ;;
        --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
        --skip_prepare) SKIP_PREPARE=1; shift ;;
        --skip_train)  SKIP_TRAIN=1;   shift ;;
        --skip_infer)  SKIP_INFER=1;   shift ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

export PYTHONPATH=$SATLAS
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "========================================"
echo " Pipeline SATLAS + Focal Frequency Loss "
echo "========================================"
echo " SATLAS:      $SATLAS"
echo " Dados:       $OUT_DATA"
echo " Resultados:  $RESULTS"
echo ""

# -----------------------------------------------------------------------
# 1. Preparar chips
# -----------------------------------------------------------------------
if [[ $SKIP_PREPARE -eq 0 ]]; then
    echo "[1/4] Preparando chips..."
    python "$SCRIPTS/prepare_chips.py" \
        --planet_dir "$PLANET_DIR" \
        --gt_tif     "$GT_TIF" \
        --out_dir    "$OUT_DATA"
    echo ""
else
    echo "[1/4] Preparar chips: PULADO"
fi

# -----------------------------------------------------------------------
# 2. Treinar (3 variantes sequencialmente)
# -----------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "[2/4] Treinando variantes..."
    for variant in baseline ffl_w01 ffl_w10; do
        yml="$CONFIGS/${variant}_from_pretrained.yml"
        echo "  Treinando $variant ($yml)..."
        cd "$SATLAS"
        python -m ssr.train -opt "$yml"
        cd -
        echo "  Treino $variant concluido."
    done
    echo ""
else
    echo "[2/4] Treino: PULADO"
fi

# -----------------------------------------------------------------------
# 3. Inferencia (3 variantes)
# -----------------------------------------------------------------------
if [[ $SKIP_INFER -eq 0 ]]; then
    echo "[3/4] Rodando inferencia (checkpoint $CHECKPOINT)..."
    for variant in baseline ffl_w01 ffl_w10; do
        ckpt="$RESULTS/$variant/${variant}_from_pretrained/models/net_g_${CHECKPOINT}.pth"
        sr_out="$RESULTS/$variant/sr_flat"
        if [[ ! -f "$ckpt" ]]; then
            echo "  [AVISO] Checkpoint nao encontrado: $ckpt"
            continue
        fi
        echo "  $variant -> $sr_out"
        python "$SCRIPTS/infer_chips.py" \
            --checkpoint "$ckpt" \
            --planet_dir "$OUT_DATA/planet_chips" \
            --out_dir    "$sr_out" \
            --gt_dir     "$OUT_DATA/gt_chips"
    done
    echo ""
else
    echo "[3/4] Inferencia: PULADO"
fi

# -----------------------------------------------------------------------
# 4. Avaliacao
# -----------------------------------------------------------------------
echo "[4/4] Avaliando..."

SR_DIRS=()
NAMES=()
for variant in baseline ffl_w01 ffl_w10; do
    sr_dir="$RESULTS/$variant/sr_flat"
    if [[ -d "$sr_dir" ]]; then
        SR_DIRS+=("$sr_dir")
        NAMES+=("$variant")
    fi
done

GT_FLAT="$RESULTS/baseline/gt_flat"

if [[ ${#SR_DIRS[@]} -eq 0 ]]; then
    echo "[AVISO] Nenhum diretorio SR encontrado. Rode sem --skip_infer primeiro."
elif [[ ! -d "$GT_FLAT" ]]; then
    echo "[AVISO] gt_flat nao encontrado em $GT_FLAT"
    echo "        Rode infer_chips.py com --gt_dir para gera-lo."
else
    EVAL_OUT="$RESULTS/evaluation"
    python "$EVALSCRIPT" \
        --sr_dirs "${SR_DIRS[@]}" \
        --gt_dir  "$GT_FLAT" \
        --names   "${NAMES[@]}" \
        --out_dir "$EVAL_OUT"
    echo ""
    echo "Resultados em: $EVAL_OUT"
fi

echo ""
echo "========================================"
echo " Pipeline concluido!"
echo "========================================"
