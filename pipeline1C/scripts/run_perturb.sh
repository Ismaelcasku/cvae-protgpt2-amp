#!/bin/bash
# =============================================================================
# Pipeline 1C — Generación por perturbación alrededor de una seed AMP
# Modelo: cvae_v4_gate_run3_freebits01 (epoch 20)
#         val_lm=5.62  val_delta_shuffle=1.50  val_delta_zero=+0.57
#
# Uso:
#   bash run_perturb.sh "GIGKFLKKAKKFGKAFVKILKK"
#   bash run_perturb.sh "GIGKFLKKAKKFGKAFVKILKK" all 300 0.3
#   bash run_perturb.sh "GIGKFLKKAKKFGKAFVKILKK" ecoli 200 0.5 0.8
#
# Argumentos posicionales:
#   $1  seed_seq     (obligatorio)
#   $2  bacteria     (default: all)
#   $3  n            (default: 200)
#   $4  sigma        (default: 0.3)
#   $5  temperature  (default: 0.9)
# =============================================================================

set -e

if [ -z "$1" ]; then
    echo "ERROR: Debes proporcionar una secuencia seed."
    echo "Uso: bash run_perturb.sh <seed_seq> [bacteria] [n] [sigma] [temperature]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/fullvae_generate.py"

# ── Artifacts ────────────────────────────────────────────────────────────────
ARTIFACTS_DIR="$HOME/bhome/ML/pipeline1B/runs/cvae_v4_gate_run3_freebits01/artifacts"

# ── Salida ───────────────────────────────────────────────────────────────────
RUN_NAME="perturb_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$HOME/bhome/ML/pipeline1C/runs/$RUN_NAME"
OUTPUT="$OUT_DIR/results.csv"

# ── Parámetros ────────────────────────────────────────────────────────────────
SEED_SEQ="$1"
BACTERIA="${2:-all}"          # ecoli | kpneumoniae | paeruginosa | all
N="${3:-200}"                 # secuencias únicas por bacteria
SIGMA="${4:-0.3}"             # radio de perturbacion (0.1=cerca, 1.0=lejos)
TEMPERATURE="${5:-0.9}"
TOP_P="0.95"
MAX_NEW_TOKENS="60"
MIN_LEN="8"
MAX_LEN="60"
BATCH_SIZE="32"
SEED="42"
DEVICE="cuda"

# ── Referencia para novelty ──────────────────────────────────────────────────
REFERENCE_CSV="$HOME/bhome/ML/data/train.csv"

# ── Ejecución ────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "  Pipeline 1C  |  mode=perturb"
echo "  artifacts  : $ARTIFACTS_DIR"
echo "  bacteria   : $BACTERIA  |  n=$N"
echo "  seed_seq   : $SEED_SEQ"
echo "  sigma      : $SIGMA  |  temp=$TEMPERATURE"
echo "  output     : $OUTPUT"
echo "=============================================="

python "$SCRIPT" \
    --artifacts_dir   "$ARTIFACTS_DIR"   \
    --output          "$OUTPUT"          \
    --bacteria        "$BACTERIA"        \
    --mode            perturb            \
    --seed_seq        "$SEED_SEQ"        \
    --n               "$N"               \
    --sigma           "$SIGMA"           \
    --temperature     "$TEMPERATURE"     \
    --top_p           "$TOP_P"           \
    --max_new_tokens  "$MAX_NEW_TOKENS"  \
    --min_len         "$MIN_LEN"         \
    --max_len         "$MAX_LEN"         \
    --batch_size      "$BATCH_SIZE"      \
    --max_tries_factor 40               \
    --reference_csv   "$REFERENCE_CSV"   \
    --device          "$DEVICE"          \
    --seed            "$SEED"

echo ""
echo "[OK] Run completado: $OUTPUT"
