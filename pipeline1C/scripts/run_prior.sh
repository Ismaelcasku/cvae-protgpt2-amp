#!/bin/bash
# =============================================================================
# Pipeline 1C — Generación desde prior ~ N(0,1)
# Modelo: cvae_v4_gate_run3_freebits01 (epoch 20)
#         val_lm=5.62  val_delta_shuffle=1.50  val_delta_zero=+0.57
#
# Uso:
#   bash run_prior.sh                    # parametros por defecto
#   bash run_prior.sh ecoli 500          # bacteria especifica, 500 seqs
#   bash run_prior.sh all 200 0.8        # all + temperatura 0.8
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/fullvae_generate.py"

# ── Artifacts ────────────────────────────────────────────────────────────────
ARTIFACTS_DIR="$HOME/bhome/ML/pipeline1B/runs/cvae_v4_gate_run3_freebits01/artifacts"

# ── Salida ───────────────────────────────────────────────────────────────────
RUN_NAME="prior_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$HOME/bhome/ML/pipeline1C/runs/$RUN_NAME"
OUTPUT="$OUT_DIR/results.csv"

# ── Parámetros (sobreescribibles por argv) ────────────────────────────────────
BACTERIA="${1:-all}"          # ecoli | kpneumoniae | paeruginosa | all
N="${2:-200}"                 # secuencias únicas por bacteria
TEMPERATURE="${3:-0.9}"
SIGMA="1.0"                   # prior = N(0, sigma)
TOP_P="0.95"
MAX_NEW_TOKENS="60"
MIN_LEN="8"
MAX_LEN="60"
BATCH_SIZE="32"
SEED="42"
DEVICE="cuda"

# ── Referencia para novelty (ajusta la ruta a tu CSV de training) ─────────────
REFERENCE_CSV="$HOME/bhome/ML/data/train.csv"

# ── Ejecución ────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "  Pipeline 1C  |  mode=prior"
echo "  artifacts  : $ARTIFACTS_DIR"
echo "  bacteria   : $BACTERIA  |  n=$N"
echo "  sigma      : $SIGMA  |  temp=$TEMPERATURE"
echo "  output     : $OUTPUT"
echo "=============================================="

python "$SCRIPT" \
    --artifacts_dir   "$ARTIFACTS_DIR"   \
    --output          "$OUTPUT"          \
    --bacteria        "$BACTERIA"        \
    --mode            prior              \
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
