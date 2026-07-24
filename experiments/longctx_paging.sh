#!/usr/bin/env bash
# =============================================================================
# experiments/longctx_paging.sh
# =============================================================================
# Long-context decode with KV paging active (paper Fig. 6).
#
# Sweeps prompt length on a configuration whose KV cache genuinely outgrows the
# GPU budget, so CPU KV paging activates (the paper's Fig. 6 run is
# Mistral-Nemo-12B; `bash reproduce.sh fig6` pins that model). The script
# enables paper_bench's --pad-to-input-len so each --input-len point is a real
# prompt length, not merely a truncation cap. Actual input tokens and KV paging
# counters are recorded; each run reports decode throughput AND eviction/
# prefetch counts, i.e. throughput vs. context in the paging regime and how
# eviction/prefetch frequency scales with context.
#
# Env overrides:
#   MODEL=NousResearch/Llama-2-13b-hf   Paging-active model (weights > GPU headroom)
#   CONTEXTS="1024 2048 4096 8192 16384"  Prompt lengths to sweep
#   OUTPUT_LEN=64                       Decode tokens (enough to measure tok/s in regime)
#   NUM_REQUESTS=5                      Timed requests per point
#   NUM_WARMUP=1                        Warmup requests
#   CPU_THREADS=8                       HELM CPU GEMV threads
#   CONDA_ENV=<name>                    Conda env (default: current Python)
#   CUDA_VISIBLE_DEVICES=0              GPU id
#   TIMEOUT=14400                       Per-bench wall-clock timeout (s)
#   MIN_FREE_GB=30                      Hard-fail if free RAM below this (13B fp16 + KV)
#   SKIP_RAM_CHECK=0                    Set 1 to bypass the RAM preflight
#   ALLOW_DOWNLOAD=1                    1 = allow HF downloads; 0 = offline
#   OUT_ROOT=<dir>                      Output base dir
#   QUICK=0                            1 = smoke: contexts "512 2048", 2 reqs, out=16
#   SYNTHETIC_LONG_PROMPT=1            Generate a prompt longer than max(CONTEXTS)
#                                       so --input-len is a real context length,
#                                       not merely a truncation cap.
#
# --pad-to-input-len is passed below so --input-len actually produces an
# input_len-token prefill. Without it, paper_bench only *truncates* a ~20-token
# default prompt, so every context runs the same tiny prompt: flat tok/s, flat
# peak GPU, and prefetched=evicted=0 (KV never grows -> no paging).
#
# Optional: HELM_GPU_KV_WATERMARK_MB=<MB> forces the GPU KV residency budget.
# Set it below the context KV footprint to *guarantee* eviction/prefetch and
# measure the PCIe KV-streaming crossover (fixed-budget W, sweep context > W).
#
# KV offload stays ON (default). Do NOT pass --no-kv-offload here: paging is the point.
#
# Examples:
#   QUICK=1 bash experiments/longctx_paging.sh
#   bash experiments/longctx_paging.sh
#   python experiments/longctx_paging_summarize.py "$OUT_ROOT"
# =============================================================================

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-NousResearch/Llama-2-13b-hf}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
NUM_WARMUP="${NUM_WARMUP:-1}"
CPU_THREADS="${CPU_THREADS:-8}"
CONDA_ENV="${CONDA_ENV:-}"
TIMEOUT="${TIMEOUT:-14400}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
SKIP_RAM_CHECK="${SKIP_RAM_CHECK:-0}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
QUICK="${QUICK:-0}"
SYNTHETIC_LONG_PROMPT="${SYNTHETIC_LONG_PROMPT:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HELM_CONT_CAPACITY="${HELM_CONT_CAPACITY:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
if [[ "$ALLOW_DOWNLOAD" == "0" ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

if [[ "$QUICK" == "1" ]]; then
    CONTEXTS="${CONTEXTS:-512 2048}"
    NUM_REQUESTS=2
    NUM_WARMUP=1
    OUTPUT_LEN=16
else
    CONTEXTS="${CONTEXTS:-1024 2048 4096 8192 16384}"
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/results/longctx_paging_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/STATUS.tsv"
printf "context\tactual_input_tokens\tstatus\tplan\ttok_s\tttft_ms\tpeak_gpu_mb\tpeak_cpu_mb\tevict_calls\tpages_evicted\tbytes_evicted\tprefetch_calls\tpages_prefetched\tbytes_prefetched\n" > "$STATUS"

log()  { echo "[longctx $(date +%H:%M:%S)] $*"; }
warn() { echo "[longctx WARN $(date +%H:%M:%S)] $*" >&2; }
die()  { echo "[longctx FATAL $(date +%H:%M:%S)] $*" >&2; exit 1; }

# Python env: the active Python (uv .venv, venv, Docker), or a conda env when
# CONDA_ENV is set explicitly.
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        log "conda env: $CONDA_ENV"
    else
        warn "could not activate conda env '$CONDA_ENV'; using current Python ($(command -v python))"
    fi
else
    log "using current Python: $(command -v python || echo 'none on PATH')"
fi

if [[ "$SKIP_RAM_CHECK" != "1" ]]; then
    FREE_GB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"
    if [[ -n "${FREE_GB:-}" && "$FREE_GB" -lt "$MIN_FREE_GB" ]]; then
        die "free RAM ${FREE_GB} GB < MIN_FREE_GB ${MIN_FREE_GB} GB. Set SKIP_RAM_CHECK=1 to override."
    fi
fi

BASE_PROMPT_ARGS=()
BASE_PROMPT_NOTE="default short prompt"
if [[ "$SYNTHETIC_LONG_PROMPT" == "1" ]]; then
    BASE_PROMPT_ARGS=(--pad-to-input-len)
    BASE_PROMPT_NOTE="paper_bench --pad-to-input-len"
fi

cat > "$OUT_ROOT/CONFIG.txt" <<EOF
=== Long-context KV paging sweep ===
Started:    $(date)
Model:      $MODEL  (KV offload ON; paging is the point)
Contexts:   $CONTEXTS
Output len: $OUTPUT_LEN   Requests: $NUM_REQUESTS (warmup $NUM_WARMUP)
Prompt:     $BASE_PROMPT_NOTE
Cont KV:    HELM_CONT_CAPACITY=$HELM_CONT_CAPACITY
KV config:  HELM_KV_PAGE_SIZE=${HELM_KV_PAGE_SIZE:-auto}  HELM_GPU_KV_WATERMARK_MB=${HELM_GPU_KV_WATERMARK_MB:-auto}
GPU:
$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)")
RAM:
$(free -h 2>/dev/null | head -2)
EOF
cat "$OUT_ROOT/CONFIG.txt"

extract_point() {
    local results_json="$1" outlen="$2"
    python - "$results_json" "$outlen" <<'PY' 2>/dev/null
import json, sys
path, outlen = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
    s = d["latency_sweep"]["helm"][str(outlen)]
    print(f'{s.get("stage_plan","?")}\t{s.get("input_tokens_p50","?")}\t'
          f'{s.get("decode_tok_per_s_mean","?")}\t{s.get("ttft_p50","?")}\t'
          f'{s.get("peak_gpu_mb_mean","?")}\t{s.get("peak_cpu_mb_mean","?")}\t'
          f'{s.get("kv_evict_calls","?")}\t{s.get("kv_pages_evicted","?")}\t'
          f'{s.get("kv_bytes_evicted","?")}\t{s.get("kv_prefetch_calls","?")}\t'
          f'{s.get("kv_pages_prefetched","?")}\t{s.get("kv_bytes_prefetched","?")}')
except Exception:
    print("?\t?\t?\t?\t?\t?\t?\t?\t?\t?\t?\t?")
PY
}

overall_rc=0
for ctx in $CONTEXTS; do
    sub="$OUT_ROOT/ctx_${ctx}"
    if [[ -f "$sub/.done" ]]; then
        log "ctx=$ctx already done — skip"; continue
    fi
    mkdir -p "$sub"
    log "═══ context=$ctx ═══"
    # This sweep only needs decode tok/s + KV paging counters vs context length:
    # latency sweep alone. --ablations / --throughput-concurrency trigger
    # in-process reloads that ate the 2h wall on the original run (died mid
    # batch=2 reload after only ctx=1024 was complete). Dropping them collapses
    # each outer iteration to one load + sweep.
    if timeout --kill-after=30 "$TIMEOUT" python -u experiments/paper_bench.py \
        --model "$MODEL" \
        "${BASE_PROMPT_ARGS[@]}" \
        --backends helm \
        --dtype float16 \
        --input-len "$ctx" \
        --output-lens "$OUTPUT_LEN" \
        --num-requests "$NUM_REQUESTS" \
        --num-warmup "$NUM_WARMUP" \
        --cpu-threads "$CPU_THREADS" \
        --skip-feasibility --skip-max-decode --no-lm-eval \
        --throughput-concurrency 0 \
        --timeout "$TIMEOUT" \
        --output-dir "$sub" \
        2>&1 | tee "$sub/run.log"
    then
        point="$(extract_point "$sub/paper_results.json" "$OUTPUT_LEN")"
        IFS=$'\t' read -r plan actual_input toks ttft gpu cpu ec ev be pc pf bp <<< "$point"
        touch "$sub/.done"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$ctx" "$actual_input" "success" "$plan" "$toks" "$ttft" "$gpu" "$cpu" \
            "$ec" "$ev" "$be" "$pc" "$pf" "$bp" >> "$STATUS"
        log "✓ ctx=$ctx  actual_input=$actual_input  plan=[$plan]  tok/s=$toks  ttft_ms=$ttft  gpu_mb=$gpu cpu_mb=$cpu  evict_calls=$ec pages_evicted=$ev bytes_evicted=$be  prefetch_calls=$pc pages_prefetched=$pf bytes_prefetched=$bp"
        if [[ "$pf" == "0" || "$pf" == "?" ]]; then
            warn "  prefetched=$pf at ctx=$ctx: KV may still fit on GPU (no paging). Increase ctx."
        fi
    else
        rc=$?
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ctx" "-" "failed(rc=$rc)" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" >> "$STATUS"
        warn "✗ ctx=$ctx failed rc=$rc (see $sub/run.log)"
        overall_rc=1
    fi
    sync 2>/dev/null || true; sleep 3
done

log "═════════════════════════════════════════════"
log "DONE. Output root: $OUT_ROOT"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
log "Summarize: python experiments/longctx_paging_summarize.py \"$OUT_ROOT\""
exit $overall_rc
