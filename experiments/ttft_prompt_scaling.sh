#!/usr/bin/env bash
# =============================================================================
# experiments/ttft_prompt_scaling.sh
# =============================================================================
# TTFT vs. prompt length on a large CPU partition (paper Fig. 4).
#
# Sweeps input length for Qwen3-32B on the large CPU partition the auto
# planner selects on a 24 GB GPU (stage0@cpu(43u), stage1@cuda(23u)) and
# records TTFT p50/p95 per input length. Accelerate is optionally run on the
# same config so the HELM-vs-Accelerate TTFT tradeoff is shown at that
# partition.
#
# TTFT is expected to GROW with prompt length: prefill is O(prompt) work and a
# heavy CPU partition makes each token expensive. --pad-to-input-len below
# pads every prompt to exactly the requested length so the input axis is real.
#
# Hardware: Qwen3-32B fp16 weights are ~64 GB. The cpu(43) partition keeps 43
# transformer layers in CPU RAM, so this needs the high-RAM host used for
# rtx_3090_helm_32b_ablations.json (AMD EPYC + RTX 3090), not a consumer
# machine. The script fails fast if free RAM looks insufficient.
#
# Env overrides (all optional):
#   MODEL=Qwen/Qwen3-32B           Target model
#   BACKENDS="helm accelerate"     Backends to sweep
#   INPUT_LENS="128 512 1024 2048 4096"   Prompt lengths to sweep
#   OUTPUT_LEN=8                   Decode tokens per request (small: isolates TTFT)
#   NUM_REQUESTS=5                 Timed requests per point (p50/p95)
#   NUM_WARMUP=1                   Warmup requests (excluded from stats)
#   CPU_THREADS=8                  HELM CPU GEMV threads
#   CONDA_ENV=<name>               Conda env to activate (default: current Python)
#   CUDA_VISIBLE_DEVICES=0         GPU id (0 = RTX 3090)
#   TIMEOUT=14400                  Per-bench wall-clock timeout (s); 4096-token
#                                  prefill on a 43-layer CPU stage is slow.
#   MIN_FREE_GB=70                 Hard-fail if free RAM is below this
#   SKIP_RAM_CHECK=0               Set 1 to bypass the RAM preflight
#   ALLOW_DOWNLOAD=1               1 = allow HF downloads; 0 = offline (HF_HUB_OFFLINE=1)
#   OUT_ROOT=<dir>                 Output base dir
#   QUICK=0                        1 = smoke: 2 input lens, 2 reqs, out=4
#   EXPECT_CPU_UNITS=43            Expected CPU units in the auto plan (warn on drift)
#
# Examples:
#   QUICK=1 bash experiments/ttft_prompt_scaling.sh                 # smoke first
#   bash experiments/ttft_prompt_scaling.sh                         # helm + accelerate
#   BACKENDS=helm bash experiments/ttft_prompt_scaling.sh           # helm only
#
# After it finishes:
#   python experiments/ttft_prompt_scaling_summarize.py "$OUT_ROOT"
# emits a TTFT-vs-input-length table (per backend, with the selected stage plan).
# =============================================================================

set -uo pipefail   # NOT -e: a single point failing must not abort the sweep

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-32B}"
BACKENDS="${BACKENDS:-helm accelerate}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
NUM_WARMUP="${NUM_WARMUP:-1}"
CPU_THREADS="${CPU_THREADS:-8}"
CONDA_ENV="${CONDA_ENV:-}"
TIMEOUT="${TIMEOUT:-14400}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
SKIP_RAM_CHECK="${SKIP_RAM_CHECK:-0}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
QUICK="${QUICK:-0}"
# CPU_LAYERS: if set (e.g. 43), pin HELM to a MANUAL cpu(N)+cuda(rest) split via
# paper_bench --cpu-layers, instead of relying on the auto-planner. This is what
# lets a >24 GB GPU (e.g. A6000 49 GB, which would auto-pick a lighter split)
# reproduce the paper's 24 GB-forced cpu(43) partition.
CPU_LAYERS="${CPU_LAYERS:-}"
EXPECT_CPU_UNITS="${EXPECT_CPU_UNITS:-${CPU_LAYERS:-43}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
if [[ "$ALLOW_DOWNLOAD" == "0" ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

if [[ "$QUICK" == "1" ]]; then
    INPUT_LENS="${INPUT_LENS:-128 1024}"
    NUM_REQUESTS=2
    NUM_WARMUP=1
    OUTPUT_LEN=4
else
    INPUT_LENS="${INPUT_LENS:-128 512 1024 2048 4096}"
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/results/ttft_prompt_scaling_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/STATUS.tsv"
printf "backend\tinput_len\tstatus\tstage_plan\tttft_p50_ms\n" > "$STATUS"

log()  { echo "[ttft $(date +%H:%M:%S)] $*"; }
warn() { echo "[ttft WARN $(date +%H:%M:%S)] $*" >&2; }
die()  { echo "[ttft FATAL $(date +%H:%M:%S)] $*" >&2; exit 1; }

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

# ── Preflight: explicit failure, no silent fallback ──────────────────────────
if [[ "$SKIP_RAM_CHECK" != "1" ]]; then
    FREE_GB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"
    if [[ -n "${FREE_GB:-}" && "$FREE_GB" -lt "$MIN_FREE_GB" ]]; then
        die "free RAM ${FREE_GB} GB < MIN_FREE_GB ${MIN_FREE_GB} GB. Qwen3-32B fp16 + cpu(43) needs a high-RAM host. Set SKIP_RAM_CHECK=1 to override."
    fi
    log "RAM preflight OK (free ~${FREE_GB:-?} GB)"
fi

cat > "$OUT_ROOT/CONFIG.txt" <<EOF
=== TTFT-vs-input sweep (large CPU partition) ===
Started:      $(date)
Model:        $MODEL
Backends:     $BACKENDS
Input lens:   $INPUT_LENS
Output len:   $OUTPUT_LEN
Num requests: $NUM_REQUESTS (warmup $NUM_WARMUP)
CPU threads:  $CPU_THREADS
Conda env:    $CONDA_ENV
CUDA dev:     $CUDA_VISIBLE_DEVICES
Offline:      HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
Per-bench TO: ${TIMEOUT}s
CPU_LAYERS:   ${CPU_LAYERS:-<auto>}
Expect plan:  stage0@cpu(${EXPECT_CPU_UNITS}u)
GPU:
$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)")
RAM:
$(free -h 2>/dev/null | head -2)
EOF
cat "$OUT_ROOT/CONFIG.txt"

# Pull ttft_p50 + stage_plan out of one paper_results.json (stdlib python).
extract_point() {
    local results_json="$1" backend="$2" outlen="$3"
    python - "$results_json" "$backend" "$outlen" <<'PY' 2>/dev/null
import json, sys
path, backend, outlen = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
    s = d["latency_sweep"][backend][str(outlen)]
    print(f'{s.get("stage_plan","?")}\t{s.get("ttft_p50","?")}')
except Exception:
    print("?\t?")
PY
}

# Manual-partition args. Built as an array (not an unquoted ${var:+...}, whose
# embedded quotes do not survive word-splitting reliably) so an empty CPU_LAYERS
# expands to nothing and a set one passes "--cpu-layers 43" cleanly to paper_bench.
HELM_PART_ARGS=()
[[ -n "$CPU_LAYERS" ]] && HELM_PART_ARGS=(--cpu-layers "$CPU_LAYERS")

# ── Sweep ────────────────────────────────────────────────────────────────────
overall_rc=0
for backend in $BACKENDS; do
    for in_len in $INPUT_LENS; do
        sub="$OUT_ROOT/$backend/in${in_len}"
        if [[ -f "$sub/.done" ]]; then
            log "$backend/in$in_len already done — skip"
            continue
        fi
        mkdir -p "$sub"
        log "═══ $backend  input_len=$in_len ═══"
        # The study only needs TTFT vs input_len at the cpu(43) partition: latency
        # sweep alone. --ablations and --throughput-concurrency are not needed,
        # and they each trigger a fresh model reload inside the
        # same process — on the original (with ablations) the 2h wall expired
        # mid-ablation after only the input=128/helm point was complete. Dropping
        # them collapses each outer iteration from ~4 reloads (~7m+sweep) down
        # to one load + sweep (~5m). --throughput-concurrency 0 is the explicit
        # skip sentinel paper_bench understands.
        if timeout --kill-after=30 "$TIMEOUT" python -u experiments/paper_bench.py \
            --model "$MODEL" \
            --backends "$backend" \
            --dtype float16 \
            --input-len "$in_len" \
            --pad-to-input-len \
            --output-lens "$OUTPUT_LEN" \
            --num-requests "$NUM_REQUESTS" \
            --num-warmup "$NUM_WARMUP" \
            --cpu-threads "$CPU_THREADS" \
            ${HELM_PART_ARGS[@]+"${HELM_PART_ARGS[@]}"} \
            --skip-feasibility --skip-max-decode --no-lm-eval \
            --throughput-concurrency 0 \
            --timeout "$TIMEOUT" \
            --output-dir "$sub" \
            2>&1 | tee "$sub/run.log"
        then
            point="$(extract_point "$sub/paper_results.json" "$backend" "$OUTPUT_LEN")"
            plan="${point%%$'\t'*}"; ttft="${point##*$'\t'}"
            touch "$sub/.done"
            printf "%s\t%s\t%s\t%s\t%s\n" "$backend" "$in_len" "success" "$plan" "$ttft" >> "$STATUS"
            log "✓ $backend/in$in_len  plan=[$plan]  ttft_p50=${ttft}ms"
            if [[ "$backend" == "helm" && "$plan" != *"cpu(${EXPECT_CPU_UNITS}"* ]]; then
                warn "  plan is NOT cpu(${EXPECT_CPU_UNITS}u): [$plan]. the study needs the LARGE partition; check planner/hardware."
            fi
        else
            rc=$?
            printf "%s\t%s\t%s\t%s\t%s\n" "$backend" "$in_len" "failed(rc=$rc)" "-" "-" >> "$STATUS"
            warn "✗ $backend/in$in_len failed rc=$rc (see $sub/run.log)"
            overall_rc=1
        fi
        sync 2>/dev/null || true; sleep 3
    done
done

log "═════════════════════════════════════════════"
log "DONE. Output root: $OUT_ROOT"
log "Status table:"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
log ""
log "Summarize with:"
log "  python experiments/ttft_prompt_scaling_summarize.py \"$OUT_ROOT\""
exit $overall_rc
