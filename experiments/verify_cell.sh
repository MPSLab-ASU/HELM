#!/usr/bin/env bash
# =============================================================================
# verify_cell.sh - fast single-cell Table II verification for artifact
# evaluators: run ONE model x backend latency cell at the paper's operating
# point (padded 128-token input, 128 output tokens) with a reduced request
# count, then compare it against the canonical paper results.
#
# n=3 by default: per-token decode variation is below 1% (paper, Table III
# note), so 3 requests are ample for the +/-25% reproduction tolerance while
# cutting wall-clock ~4x vs the full 10-request protocol.
#
# Usage:
#   bash experiments/verify_cell.sh <hf-model-id> <backend> <gpu-name> <table-model>
#
#   bash experiments/verify_cell.sh Qwen/Qwen3-14B helm       "RTX 3090" Qwen3-14B
#   bash experiments/verify_cell.sh Qwen/Qwen3-14B accelerate "RTX 3090" Qwen3-14B
#   DTYPE=bfloat16 bash experiments/verify_cell.sh google/gemma-2-27b-it helm "RTX 3090" Gemma-2-27B
#
# backend: helm | helm-router | accelerate | deepspeed | vllm
# gpu-name / table-model: the Table II row to compare against
#           ("RTX 4060" | "RTX 3090" | "L40S"; model as printed in Table II).
#
# Env overrides:
#   GPU_ID=<n>      which GPU to benchmark on (index from `nvidia-smi -L`);
#                   required on multi-GPU hosts unless CUDA_VISIBLE_DEVICES is set
#   NUM_REQUESTS=3  OUTPUT_LEN=128  NUM_WARMUP=1  CPU_THREADS=8
#   DTYPE=float16   TOLERANCE=0.25  OUT_DIR=<dir>
#   PAPER_BENCH=experiments/paper_bench.py   (test hook)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_ID="${1:?usage: verify_cell.sh <hf-model-id> <backend> <gpu-name> <table-model>}"
BACKEND="${2:?missing backend (helm|helm-router|accelerate|deepspeed|vllm)}"
GPU_NAME="${3:?missing Table II GPU name (e.g. \"RTX 3090\")}"
TABLE_MODEL="${4:?missing Table II model name (e.g. Qwen3-14B)}"

NUM_REQUESTS="${NUM_REQUESTS:-3}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_WARMUP="${NUM_WARMUP:-1}"
CPU_THREADS="${CPU_THREADS:-8}"
DTYPE="${DTYPE:-float16}"
TOLERANCE="${TOLERANCE:-0.25}"
PAPER_BENCH="${PAPER_BENCH:-experiments/paper_bench.py}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
OUT_DIR="${OUT_DIR:-experiments/results/verify_cell_$(date +%Y%m%d_%H%M%S)/${MODEL_ID//\//-}_${BACKEND}}"

# ── GPU selection ────────────────────────────────────────────────────────────
# The paper's setup is strictly single-GPU; on multi-GPU hosts an unpinned run
# lets baselines (Accelerate device_map=auto) spread across devices and
# invalidates the feasibility/OOM comparison. Require an explicit choice.
if [[ -n "${GPU_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && (( $(nvidia-smi -L | grep -c '^GPU ' || true) > 1 )); then
        echo "[verify_cell] multiple GPUs detected - pick one with GPU_ID=<index>:"
        nvidia-smi -L
        echo "[verify_cell] e.g.: GPU_ID=0 bash experiments/verify_cell.sh $MODEL_ID $BACKEND \"$GPU_NAME\" $TABLE_MODEL"
        exit 2
    fi
    sel="${CUDA_VISIBLE_DEVICES:-0}"
    sel="${sel%%,*}"
    gpu_line=""
    [[ "$sel" =~ ^[0-9]+$ ]] && gpu_line="$(nvidia-smi -L | sed -n "$(( sel + 1 ))p")"
    echo "[verify_cell] benchmarking on: ${gpu_line:-GPU $sel}"
    case "$gpu_line" in
        *"$GPU_NAME"*) ;;
        *) echo "[verify_cell] note: selected GPU does not look like the requested Table II row '$GPU_NAME';"
           echo "               absolute numbers are only comparable on a matching GPU class." ;;
    esac
fi

mkdir -p "$OUT_DIR"
echo "[verify_cell] $MODEL_ID / $BACKEND -> $OUT_DIR (n=$NUM_REQUESTS, out=$OUTPUT_LEN, in=128 padded)"

"$PYTHON" -u "$PAPER_BENCH" \
    --model "$MODEL_ID" \
    --backends "$BACKEND" \
    --dtype "$DTYPE" \
    --input-len 128 --pad-to-input-len \
    --output-lens "$OUTPUT_LEN" \
    --num-requests "$NUM_REQUESTS" --num-warmup "$NUM_WARMUP" \
    --cpu-threads "$CPU_THREADS" \
    --skip-feasibility --skip-max-decode --throughput-concurrency 0 \
    --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"

echo
echo "[verify_cell] comparing against Table II ($GPU_NAME / $TABLE_MODEL / $BACKEND, +/-$(awk "BEGIN{print $TOLERANCE*100}")%)"
"$PYTHON" results/validate_results.py \
    --check-run "$OUT_DIR/paper_results.json" \
    --gpu "$GPU_NAME" --model "$TABLE_MODEL" \
    --backend "$BACKEND" --output-len "$OUTPUT_LEN" \
    --tolerance "$TOLERANCE"

echo
echo "[verify_cell] plot this cell against the paper (add more --run pairs as you verify backends):"
echo "  python results/plot_comparison.py --gpu \"$GPU_NAME\" --model $TABLE_MODEL --run $BACKEND=$OUT_DIR/paper_results.json"
