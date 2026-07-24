#!/usr/bin/env bash
# =============================================================================
# run_llama13b_helm.sh - focused HELM benchmark for a LLaMA-family 13B model
# (plus a Gemma-2-9B control) on an RTX 3090-class GPU. Its Llama-2-13B run is
# the HELM side of the paper's llama.cpp comparison (Table III).
#
# Sections (select with SECTIONS, space/comma separated):
#   LLAMA13_HELM  NousResearch/Llama-2-13b-hf   (fp16, overflow regime)
#   GEMMA_HELM    google/gemma-2-9b-it          (bf16, in-VRAM control)
#
# The script defaults to offline model loading (HF_HUB_OFFLINE=1), performs a
# GPU-headroom preflight, runs experiments/paper_bench.py for each requested
# section, validates paper_results.json afterwards (zero successful requests
# is a *failure*, never silently ignored), and appends one line per section to
# $OUT_ROOT/STATUS.tsv:  <section>\t<success|failed>\t<exit_code>\t<detail>
#
# There is NO fallback between sections: a failed LLaMA-13B run is reported as
# failed even if the Gemma control succeeds.
#
# Env overrides:
#   SECTIONS               sections to run           (default: LLAMA13_HELM)
#   OUT_ROOT               output root               (default: experiments/results/llama13b_helm_<ts>)
#   OUTPUT_LENS            output lengths to sweep   (default: "128")
#   QUICK=1                1 request / 0 warmup for smoke runs
#   CONDA_BASE             conda install root        (default: $HOME/miniconda3)
#   CONDA_ENV              conda env name            (default: none = current Python)
#   LLAMA13_MIN_GPU_FREE_MB  required free VRAM      (default: 20000)
#   CPU_THREADS            HELM CPU threads          (default: 16)
#
# Usage:
#   QUICK=1 SECTIONS=LLAMA13_HELM bash experiments/run_llama13b_helm.sh
#
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SECTIONS="${SECTIONS:-LLAMA13_HELM}"
OUT_ROOT="${OUT_ROOT:-experiments/results/llama13b_helm_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_LENS="${OUTPUT_LENS:-128}"
QUICK="${QUICK:-0}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-}"
LLAMA13_MIN_GPU_FREE_MB="${LLAMA13_MIN_GPU_FREE_MB:-20000}"
CPU_THREADS="${CPU_THREADS:-16}"

# Offline by default so a benchmark run never silently downloads different
# weights. Pre-download the model once, or set HF_HUB_OFFLINE=0.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/STATUS.tsv"
[[ -f "$STATUS" ]] || printf "section\tstatus\texit_code\tdetail\n" > "$STATUS"

log() { printf '[llama13b] %s\n' "$*"; }

record() { # section status exit_code detail
    printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> "$STATUS"
}

# ── Environment activation (only when CONDA_ENV is set; else current Python) ─
if [[ -n "$CONDA_ENV" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || true
fi
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

# ── Preflight: GPU present with enough free VRAM (the 3090 must be idle) ────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found - a CUDA GPU is required."
    record "preflight" "failed" "1" "nvidia-smi not found"
    exit 1
fi
# Respect CUDA_VISIBLE_DEVICES (nvidia-smi ignores it); otherwise take the
# freest GPU rather than blindly reading GPU 0.
GPU_QUERY=(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_QUERY+=(-i "${CUDA_VISIBLE_DEVICES%%,*}")
GPU_FREE_MB="$("${GPU_QUERY[@]}" | sort -nr | head -1 | tr -dc '0-9')"
if [[ -z "$GPU_FREE_MB" || "$GPU_FREE_MB" -lt "$LLAMA13_MIN_GPU_FREE_MB" ]]; then
    log "FATAL: only ${GPU_FREE_MB:-0} MiB GPU memory free (< ${LLAMA13_MIN_GPU_FREE_MB} MiB). Is the GPU occupied?"
    record "preflight" "failed" "1" "gpu_free_mb=${GPU_FREE_MB:-0} < ${LLAMA13_MIN_GPU_FREE_MB}"
    exit 1
fi
log "GPU headroom OK: ${GPU_FREE_MB} MiB free"
command -v free >/dev/null 2>&1 && log "host memory: $(free -h | grep -i '^Mem' || true)"

# ── Per-section config ───────────────────────────────────────────────────────
section_model() {
    case "$1" in
        LLAMA13_HELM) echo "NousResearch/Llama-2-13b-hf" ;;
        GEMMA_HELM)   echo "google/gemma-2-9b-it" ;;
        *)            echo "" ;;
    esac
}
section_dtype() {
    case "$1" in
        GEMMA_HELM) echo "bfloat16" ;;  # Gemma-2 is numerically unsafe in fp16
        *)          echo "float16" ;;
    esac
}

NUM_REQUESTS=10; NUM_WARMUP=1
[[ "$QUICK" == "1" ]] && { NUM_REQUESTS=1; NUM_WARMUP=0; }

FAILURES=0

for section in $(echo "$SECTIONS" | tr ',' ' '); do
    model_id="$(section_model "$section")"
    if [[ -z "$model_id" ]]; then
        log "unknown section '$section' - skipping"
        record "$section" "failed" "1" "unknown section"
        FAILURES=1
        continue
    fi

    # Offline cache preflight: fail explicitly instead of downloading.
    # Same cache-resolution order as run_paper_experiments.sh; HF_HUB_CACHE
    # already points at the hub/ level, HF_HOME one level above it.
    if [[ -n "${HF_HUB_CACHE:-}" ]]; then
        cache_dir="$HF_HUB_CACHE/models--${model_id//\//--}"
    else
        hf_home="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}}"
        cache_dir="$hf_home/hub/models--${model_id//\//--}"
    fi
    if [[ "$HF_HUB_OFFLINE" == "1" && ! -d "$cache_dir" ]]; then
        log "$section: model not in local HF cache: $cache_dir"
        record "$section" "failed" "1" "model not in local HF cache: $model_id"
        FAILURES=1
        continue
    fi

    sec_dir="$OUT_ROOT/$section"
    mkdir -p "$sec_dir"
    log "$section: model=$model_id dtype=$(section_dtype "$section") output_lens=[$OUTPUT_LENS] -> $sec_dir"

    rc=0
    # shellcheck disable=SC2086
    "$PYTHON" -u experiments/paper_bench.py \
        --model "$model_id" \
        --backends helm \
        --dtype "$(section_dtype "$section")" \
        --input-len 128 --pad-to-input-len \
        --output-lens $OUTPUT_LENS \
        --num-requests "$NUM_REQUESTS" --num-warmup "$NUM_WARMUP" \
        --cpu-threads "$CPU_THREADS" \
        --skip-feasibility --skip-max-decode --throughput-concurrency 0 \
        --output-dir "$sec_dir" \
        > "$sec_dir/run.log" 2>&1 || rc=$?

    results_json="$sec_dir/paper_results.json"
    if [[ ! -f "$results_json" ]]; then
        log "$section: paper_bench.py exited rc=$rc without paper_results.json"
        record "$section" "failed" "${rc:-1}" "paper_results.json not produced (rc=$rc)"
        FAILURES=1
        continue
    fi

    # Explicit validation: every requested output length must have at least
    # one successful request; zero successes is a failed section, no fallback.
    detail=""
    vrc=0
    # shellcheck disable=SC2086
    detail="$("$PYTHON" - "$results_json" $OUTPUT_LENS <<'PY'
import json
import sys

path, lens = sys.argv[1], sys.argv[2:]
data = json.load(open(path))
helm = (data.get("latency_sweep") or {}).get("helm") or {}
problems = []
for length in lens:
    entry = helm.get(str(length)) or {}
    n_req = entry.get("n_requests", 0)
    n_ok = entry.get("n_success", 0)
    if not entry or not n_ok:
        problems.append(f"output={length}: {n_ok}/{n_req} successful requests")
if problems:
    print("; ".join(problems))
    sys.exit(1)
print("all requested output lengths succeeded")
PY
)" || vrc=$?

    if [[ "$vrc" -ne 0 || "$rc" -ne 0 ]]; then
        log "$section: FAILED ($detail)"
        record "$section" "failed" "$(( rc > 0 ? rc : vrc ))" "$detail"
        FAILURES=1
    else
        log "$section: success ($detail)"
        record "$section" "success" "0" "$sec_dir"
    fi
done

log "done - status:"
cat "$STATUS"
exit "$FAILURES"
