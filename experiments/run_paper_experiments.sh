#!/usr/bin/env bash
# =============================================================================
# experiments/run_paper_experiments.sh
# =============================================================================
# Self-contained paper experiment runner.  No arguments required.
#
# Runs every combination of:
#   Model   × Backend × Section
#
# Models tested (in order of increasing size):
#   Qwen/Qwen3-4B
#   Qwen/Qwen3-8B
#   Qwen/Qwen3-14B
#   Qwen/Qwen3-32B
#
# Backends (auto-detected — a baseline is skipped with a warning when its
# package is not installed; see research/README.md for the baseline env):
#   helm        HELM heterogeneous inference (always run)
#   accelerate  HuggingFace Accelerate       (always run)
#   deepspeed   DeepSpeed ZeRO-Inference     (if `deepspeed` is importable)
#   vllm        vLLM                         (GPU only, if `vllm` is importable)
#
# Sections (each is an isolated subprocess — memory fully freed between them):
#   E  Feasibility probe   — does the model load? peak GPU/CPU memory at load
#   F  Max decode length   — longest output before OOM (HELM's KV-offload wins)
#   A  Latency sweep       — TTFT, decode tok/s, p50/p95/p99 × output lengths
#   B  Throughput          — req/s and aggregate tok/s at concurrency 1/2/4/8
#   C  HELM ablations      — batch_size, ctx_len, no_avx, no_kv_offload, threads
#   D  LM quality          — MMLU, HellaSwag, ARC-Easy (once per model)
#
# Crash recovery:
#   Each section writes a <section>.done sentinel when it completes.
#   Re-running the script skips already-completed sections automatically.
#
# Env overrides (optional):
#   QUICK=1                        Fewer requests and shorter sequences (smoke-test)
#   NO_LM_EVAL=1                   Skip Section D
#   SECTIONS=E,F,A                 Only run these sections (default: E,A,B,C; F on request)
#   OUT_ROOT=<dir>                 Override base output directory
#   DTYPE=bfloat16                 Override dtype (default: float16)
#   CPU_THREADS=16                 Override CPU thread count
#   BACKENDS_OVERRIDE=helm         Run a single backend (helm/accelerate/deepspeed/vllm)
#   MODELS_OVERRIDE=Qwen/Qwen3-4B  Run a single model
#   PYTHON=<interpreter>           Python to use (default: python on PATH)
#   DELETE_MODEL_CACHE=1           Delete each model's Hugging Face cache after
#                                  its experiments finish (saves disk on small
#                                  machines; default 0 = keep the cache)
# =============================================================================

set -euo pipefail

# ── Single-instance lock ───────────────────────────────────────────────────────
# Prevents two benchmark runs from competing for the GPU simultaneously.
LOCKFILE="/tmp/helm_paper_bench.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "ERROR: another benchmark run is already in progress (lock: $LOCKFILE)."
    echo "       Kill it first or wait for it to finish."
    echo "       Running PIDs: $(lsof -t "$LOCKFILE" 2>/dev/null | tr '\n' ' ')"
    exit 1
fi
# Lock is held for the lifetime of this process; auto-released on exit.

# ── Models ────────────────────────────────────────────────────────────────────

MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

# ── Tunable parameters ────────────────────────────────────────────────────────

PYTHON="${PYTHON:-python}"
QUICK="${QUICK:-0}"
NO_LM_EVAL="${NO_LM_EVAL:-1}"   # lm-eval disabled by default; set NO_LM_EVAL=0 to enable
SECTIONS="${SECTIONS:-E,A,B,C}"   # add F (max decode length, Fig. 5) explicitly: slow, per-candidate OOM probing
DTYPE="${DTYPE:-float16}"
CPU_THREADS="${CPU_THREADS:-8}"
BACKENDS_OVERRIDE="${BACKENDS_OVERRIDE:-}"   # e.g. "helm,accelerate" to skip vllm/deepspeed
MODELS_OVERRIDE="${MODELS_OVERRIDE:-}"       # e.g. "Qwen/Qwen3-4B,Qwen/Qwen3-8B"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-experiments/results}"
OUT_DIR="$OUT_ROOT/$TIMESTAMP"

INPUT_LEN="${INPUT_LEN:-64}"      # Table II / committed paper runs used INPUT_LEN=128 PAD_INPUT=1
PAD_INPUT="${PAD_INPUT:-0}"       # 1 -> pass --pad-to-input-len (pad prompt to exactly INPUT_LEN)
TIMEOUT="${TIMEOUT:-3600}"          # per-request timeout for sections A/B
MAX_DECODE_TIMEOUT="${MAX_DECODE_TIMEOUT:-3600}"  # timeout for each max-decode candidate in section F

if [[ "$QUICK" == "1" ]]; then
    OUTPUT_LENS="128"
    NUM_REQUESTS=5
    THROUGHPUT_CONCURRENCY="1 2"
    LM_EVAL_LIMIT=50
    MAX_DECODE_CANDIDATES="128 512 1024 2048"
else
    OUTPUT_LENS="128"
    NUM_REQUESTS=10
    THROUGHPUT_CONCURRENCY="1 2 4 8"
    LM_EVAL_LIMIT=200
    MAX_DECODE_CANDIDATES="128 512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 26624 27648 28672 29696 30720 31744 32768"
fi
# Explicit overrides beat the QUICK/full presets (useful for fast verification):
NUM_REQUESTS="${NUM_REQUESTS_OVERRIDE:-$NUM_REQUESTS}"
OUTPUT_LENS="${OUTPUT_LENS_OVERRIDE:-$OUTPUT_LENS}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"  # warmup requests before each timed section

LM_EVAL_TASKS="mmlu hellaswag arc_easy"

# ── Colours ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ts()       { date +%H:%M:%S; }
log()      { echo -e "${CYAN}[$(ts)]${RESET} $*"; }
log_ok()   { echo -e "${GREEN}[$(ts)] ✓${RESET} $*"; }
log_warn() { echo -e "${YELLOW}[$(ts)] ⚠${RESET} $*"; }
log_err()  { echo -e "${RED}[$(ts)] ✗${RESET} $*"; }
header()   { echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"; \
             echo -e "${BOLD}${CYAN}║  $*${RESET}"; \
             echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RESET}\n"; }

# ── Section guard ─────────────────────────────────────────────────────────────

should_run_section() {
    local sec="$1"
    [[ ",$SECTIONS," == *",$sec,"* ]]
}

# ── Done-file checkpointing ───────────────────────────────────────────────────
# Each completed section writes a zero-byte sentinel file.
# If the script is re-run, that section is skipped.

is_done() {
    local dir="$1" sec="$2"
    [[ -f "$dir/.${sec}.done" ]]
}

mark_done() {
    local dir="$1" sec="$2"
    touch "$dir/.${sec}.done"
    # Also append to the global progress log
    echo "{\"ts\":\"$(date -Iseconds)\",\"dir\":\"$dir\",\"section\":\"$sec\"}" \
        >> "$OUT_DIR/progress.jsonl"
}

# ── Memory flush ──────────────────────────────────────────────────────────────
# Called after every subprocess to ensure the OS reclaims GPU/CPU memory
# before the next experiment starts.

flush_memory() {
    log "  Flushing memory …"
    sync
    # Drop page cache if we have permission (no-op if not root, no sudo prompt)
    if [[ -w /proc/sys/vm/drop_caches ]]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi

    # Wait for the GPU VRAM to drop below a threshold before proceeding.
    # Each section runs paper_bench.py as a subprocess; when that process exits
    # the CUDA context is destroyed and VRAM is released by the driver.
    # We poll nvidia-smi for up to 30 s so the next experiment starts clean.
    if command -v nvidia-smi &>/dev/null; then
        local gpu_free_mb threshold_mb=500 waited=0
        local gpu_idx="${CUDA_VISIBLE_DEVICES:-0}"
        gpu_idx="${gpu_idx%%,*}"
        [[ "$gpu_idx" =~ ^[0-9]+$ ]] || gpu_idx=0
        while true; do
            # nvidia-smi reports memory.used; compare against total to get free
            local used total
            # Query the benchmarked GPU (first CUDA_VISIBLE_DEVICES entry).
            # awk reads all lines instead of closing the pipe early: `head -1`
            # SIGPIPEs nvidia-smi on multi-GPU hosts, which pipefail + set -e
            # turn into a fatal exit 141.
            used=$(nvidia-smi -i "$gpu_idx" --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {gsub(/ /, ""); print}' || true)
            total=$(nvidia-smi -i "$gpu_idx" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {gsub(/ /, ""); print}' || true)
            if [[ -n "$used" && -n "$total" ]]; then
                gpu_free_mb=$(( total - used ))
                if (( gpu_free_mb >= threshold_mb )); then
                    log "  GPU free: ${gpu_free_mb} MB — proceeding"
                    break
                fi
                log_warn "  Waiting for GPU VRAM to free (used=${used}MB / ${total}MB) …"
            fi
            if (( waited >= 30 )); then
                log_warn "  GPU did not fully free after 30 s — proceeding anyway"
                break
            fi
            sleep 3
            (( waited += 3 ))
        done
    else
        sleep 3   # No nvidia-smi: fall back to fixed delay
    fi
}

# ── Model cache eviction ──────────────────────────────────────────────────────
# HuggingFace caches checkpoints under:
#   $HF_HOME/hub/models--<org>--<name>/   (default HF_HOME = ~/.cache/huggingface)
# With DELETE_MODEL_CACHE=1 we delete this after all experiments for a model
# finish so that only one model's weights occupy disk at a time.

delete_model_cache() {
    local model="$1"   # e.g. Qwen/Qwen3-8B
    local hf_home="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}}"
    # Convert "Org/Name" → "models--Org--Name"
    local cache_name="models--${model//\/\//-}"
    cache_name="models--${model/\//--}"
    local cache_dir="$hf_home/hub/$cache_name"

    if [[ -d "$cache_dir" ]]; then
        local size
        size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "?")
        # Snapshot free space before deletion so we can confirm it actually freed
        local hub_dir="$hf_home/hub"
        mkdir -p "$hub_dir"
        local free_before
        free_before=$(df "$hub_dir" --output=avail -BM 2>/dev/null | tail -1 | tr -d 'M ' || echo "0")

        log "  Deleting HF cache for $model  ($size)  → $cache_dir"
        rm -rf "$cache_dir"
        sync   # flush filesystem buffers so the kernel updates block counts

        # Poll df until free space actually increases (confirmation-based, not time-based)
        local waited=0
        while true; do
            local free_after
            free_after=$(df "$hub_dir" --output=avail -BM 2>/dev/null | tail -1 | tr -d 'M ' || echo "0")
            if (( free_after > free_before )); then
                log_ok "  Cache deleted — disk freed  (${free_before}MB → ${free_after}MB free)"
                break
            fi
            if (( waited >= 60 )); then
                log_warn "  Disk free space did not increase after 60 s — proceeding anyway"
                break
            fi
            sleep 2
            (( waited += 2 ))
        done
    else
        log "  No HF cache found at $cache_dir (already clean)"
    fi
    flush_memory
}

# ── Python runner helper ──────────────────────────────────────────────────────
# Runs paper_bench.py in a fresh subprocess with the given extra args.
# Returns non-zero on failure but never aborts the outer loop.

run_bench() {
    local label="$1"; shift
    log "  Running: $label"
    if "$PYTHON" experiments/paper_bench.py "$@" 2>&1; then
        return 0
    else
        log_warn "  $label exited with non-zero status (results may be partial)"
        return 1
    fi
}

# ── Common flags shared by every section ──────────────────────────────────────

common_flags() {
    local model="$1" backend="$2" out_dir="$3"
    echo --model "$model" \
         --backends "$backend" \
         --dtype "$DTYPE" \
         --input-len "$INPUT_LEN" \
         $([[ "$PAD_INPUT" == "1" ]] && echo --pad-to-input-len) \
         --cpu-threads "$CPU_THREADS" \
         --output-dir "$out_dir"
}

# ── Per-section runners ───────────────────────────────────────────────────────

run_section_E() {
    local model="$1" backend="$2" out="$3"
    mkdir -p "$out"
    run_bench "E[feasibility] $model/$backend" \
        $(common_flags "$model" "$backend" "$out") \
        --output-lens 64 \
        --num-requests 1 \
        --skip-latency-sweep \
        --skip-max-decode \
        --no-lm-eval \
        --throughput-concurrency 0 \
        2>&1 | tee "$out/run.log"
}

run_section_F() {
    local model="$1" backend="$2" out="$3"
    mkdir -p "$out"
    run_bench "F[max-decode] $model/$backend" \
        $(common_flags "$model" "$backend" "$out") \
        --output-lens 64 \
        --num-requests 1 \
        --skip-feasibility \
        --skip-latency-sweep \
        --no-lm-eval \
        --max-decode-candidates $MAX_DECODE_CANDIDATES \
        --throughput-concurrency 0 \
        --timeout "$MAX_DECODE_TIMEOUT" \
        2>&1 | tee "$out/run.log"
}

run_section_A() {
    local model="$1" backend="$2" out="$3"
    mkdir -p "$out"
    run_bench "A[latency] $model/$backend" \
        $(common_flags "$model" "$backend" "$out") \
        --output-lens $OUTPUT_LENS \
        --num-requests "$NUM_REQUESTS" \
        --num-warmup "$WARMUP_REQUESTS" \
        --skip-feasibility \
        --skip-max-decode \
        --no-lm-eval \
        --throughput-concurrency 0 \
        --timeout "$TIMEOUT" \
        2>&1 | tee "$out/run.log"
}

run_section_B() {
    local model="$1" backend="$2" out="$3"
    mkdir -p "$out"
    run_bench "B[throughput] $model/$backend" \
        $(common_flags "$model" "$backend" "$out") \
        --output-lens 128 \
        --num-requests "$NUM_REQUESTS" \
        --skip-feasibility \
        --skip-max-decode \
        --no-lm-eval \
        --throughput-concurrency $THROUGHPUT_CONCURRENCY \
        --timeout "$TIMEOUT" \
        2>&1 | tee "$out/run.log"
}

run_section_C() {
    # Ablations — HELM only
    local model="$1" out="$2"
    mkdir -p "$out"
    run_bench "C[ablations] $model/helm" \
        $(common_flags "$model" "helm" "$out") \
        --output-lens 128 \
        --num-requests "$NUM_REQUESTS" \
        --skip-feasibility \
        --skip-max-decode \
        --no-lm-eval \
        --throughput-concurrency 0 \
        --ablations batch_size context_length no_avx no_kv_offload cpu_threads \
        --timeout "$TIMEOUT" \
        2>&1 | tee "$out/run.log"
}

run_section_D() {
    # LM quality — once per model, using the active backend
    local model="$1" backend="$2" out="$3"
    mkdir -p "$out"
    run_bench "D[lm-eval] $model/$backend" \
        $(common_flags "$model" "$backend" "$out") \
        --output-lens 64 \
        --num-requests 1 \
        --skip-feasibility \
        --skip-max-decode \
        --skip-latency-sweep \
        --throughput-concurrency 0 \
        --lm-eval-tasks $LM_EVAL_TASKS \
        --lm-eval-limit "$LM_EVAL_LIMIT" \
        2>&1 | tee "$out/run.log"
}

# ── Hardware detection ────────────────────────────────────────────────────────

detect_hardware() {
    log "Detecting hardware …"
    "$PYTHON" - <<'PYEOF' | tee "$OUT_DIR/hardware.json"
import json, torch
info = {"cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    info["gpu_count"]           = torch.cuda.device_count()
    info["gpu_name"]            = torch.cuda.get_device_name(0)
    info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
try:
    import psutil, platform
    info["cpu_count"]    = psutil.cpu_count(logical=False)
    info["ram_total_gb"] = psutil.virtual_memory().total / 1024**3
    info["platform"]     = platform.platform()
except: pass
try:
    import cpuinfo
    info["cpu_brand"] = cpuinfo.get_cpu_info().get("brand_raw","?")
except: pass
print(json.dumps(info, indent=2))
PYEOF

    HAS_GPU=$("$PYTHON" -c \
        "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo "0")
    GPU_GB=$("$PYTHON" -c \
        "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.total_memory/1024**3:.0f}')" \
        2>/dev/null || echo "0")
    RAM_GB=$("$PYTHON" -c \
        "import psutil; print(f'{psutil.virtual_memory().total/1024**3:.0f}')" \
        2>/dev/null || echo "0")

    export HAS_GPU GPU_GB RAM_GB
    log "GPU: ${HAS_GPU} (${GPU_GB}GB VRAM)  |  RAM: ${RAM_GB}GB"
}

select_backends() {
    if [[ -n "$BACKENDS_OVERRIDE" ]]; then
        # User-specified override: comma-separated list, e.g. "helm,accelerate"
        IFS=',' read -ra BACKENDS <<< "$BACKENDS_OVERRIDE"
        log "Backends (override): ${BACKENDS[*]}"
    else
        # Auto-detect: always run helm + accelerate; add the optional research
        # baselines only when their packages are importable (they are not part
        # of the native install), and vLLM only when a CUDA GPU is present.
        BACKENDS=()
        if [[ "$HAS_GPU" == "1" ]]; then
            if "$PYTHON" -c "import vllm" &>/dev/null; then
                BACKENDS+=("vllm")
            else
                log_warn "vllm not installed — skipping the vLLM baseline (see research/README.md)"
            fi
        fi
        BACKENDS+=("helm" "accelerate")
        if "$PYTHON" -c "import deepspeed" &>/dev/null; then
            BACKENDS+=("deepspeed")
        else
            log_warn "deepspeed not installed — skipping the DeepSpeed baseline (see research/README.md)"
        fi
        log "Backends (auto): ${BACKENDS[*]}"
    fi
    export BACKENDS
}

# ── Per-model summary ─────────────────────────────────────────────────────────

generate_model_summary() {
    local model="$1" model_slug="$2" model_dir="$3"
    log "Generating summary for $model …"
    "$PYTHON" - "$model_dir" "$model" > "$model_dir/model_summary.md" 2>/dev/null || true
}

# ── Overall summary ───────────────────────────────────────────────────────────

generate_overall_summary() {
    log "Generating overall SUMMARY.md …"
    "$PYTHON" - "$OUT_DIR" "${MODELS[@]}" > "$OUT_DIR/SUMMARY.md" <<'PYEOF' 2>/dev/null || true
import json, sys, os
from pathlib import Path

out_dir = sys.argv[1]
models  = sys.argv[2:]

lines = [
    "# HELM Paper Experiment Results",
    "",
    f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
]

hw_file = os.path.join(out_dir, "hardware.json")
if os.path.exists(hw_file):
    hw = json.load(open(hw_file))
    lines += [
        "## Hardware",
        f"- CPU : {hw.get('cpu_brand','?')} ({hw.get('cpu_count','?')} cores)",
        f"- RAM : {hw.get('ram_total_gb',0):.0f} GB",
        f"- GPU : {hw.get('gpu_name','none')} ({hw.get('gpu_memory_total_gb',0):.0f} GB)"
          if hw.get("cuda_available") else "- GPU : none",
        "",
    ]

# ── Feasibility table ────────────────────────────────────────────────────────
lines += ["## Memory Feasibility", ""]
all_backends = ["vllm", "helm", "accelerate", "deepspeed"]
header = "| model | " + " | ".join(all_backends) + " |"
sep    = "|" + "---|" * (len(all_backends) + 1)
lines += [header, sep]

for model in models:
    slug = model.replace("/", "-")
    row  = f"| {model.split('/')[-1]} |"
    for b in all_backends:
        fpath = os.path.join(out_dir, slug, b, "E_feasibility", "paper_results.json")
        if not os.path.exists(fpath):
            row += " — |"
            continue
        data = json.load(open(fpath))
        fd   = data.get("feasibility", {}).get(b, {})
        if fd.get("fits_in_memory"):
            row += f" ✓ {fd.get('peak_gpu_mb',0):.0f}MB |"
        else:
            row += f" ✗ {fd.get('status','?')} |"
    lines.append(row)
lines.append("")

# ── Max decode length table ───────────────────────────────────────────────────
lines += ["## Max Decode Length (tokens)", ""]
lines += [header, sep]
for model in models:
    slug = model.replace("/", "-")
    row  = f"| {model.split('/')[-1]} |"
    for b in all_backends:
        fpath = os.path.join(out_dir, slug, b, "F_max_decode", "paper_results.json")
        if not os.path.exists(fpath):
            row += " — |"
            continue
        data = json.load(open(fpath))
        md   = data.get("max_decode_length", {}).get(b, {})
        mx   = md.get("max_output_len", 0)
        oom  = md.get("oom_at")
        row += f" {mx:,}{' (oom@'+str(oom)+')' if oom else ''} |"
    lines.append(row)
lines.append("")

# ── Decode tok/s at output_len=128 ───────────────────────────────────────────
lines += ["## Decode Throughput at output_len=128 (tok/s mean)", ""]
lines += [header, sep]
for model in models:
    slug = model.replace("/", "-")
    row  = f"| {model.split('/')[-1]} |"
    for b in all_backends:
        fpath = os.path.join(out_dir, slug, b, "A_latency", "paper_results.json")
        if not os.path.exists(fpath):
            row += " — |"
            continue
        data  = json.load(open(fpath))
        sweep = data.get("latency_sweep", {}).get(b, {})
        entry = sweep.get("128", {})
        tps   = entry.get("decode_tok_per_s_mean", 0)
        row  += f" {tps:.1f} |" if tps else " err |"
    lines.append(row)
lines.append("")

# ── TTFT p50 at output_len=128 ────────────────────────────────────────────────
lines += ["## TTFT p50 at output_len=128 (ms)", ""]
lines += [header, sep]
for model in models:
    slug = model.replace("/", "-")
    row  = f"| {model.split('/')[-1]} |"
    for b in all_backends:
        fpath = os.path.join(out_dir, slug, b, "A_latency", "paper_results.json")
        if not os.path.exists(fpath):
            row += " — |"
            continue
        data  = json.load(open(fpath))
        sweep = data.get("latency_sweep", {}).get(b, {})
        entry = sweep.get("128", {})
        ttft  = entry.get("ttft_p50", 0)
        row  += f" {ttft:.1f} |" if ttft else " err |"
    lines.append(row)
lines.append("")

# ── HELM compiler metrics per model ──────────────────────────────────────────
lines += ["## HELM Compiler Metrics", ""]
lines += ["| model | compile_s | stage_plan | cost_err_pct |", "|---|---|---|---|"]
for model in models:
    slug  = model.replace("/", "-")
    fpath = os.path.join(out_dir, slug, "helm", "A_latency", "paper_results.json")
    if not os.path.exists(fpath):
        continue
    data  = json.load(open(fpath))
    sweep = data.get("latency_sweep", {}).get("helm", {})
    first = next((v for v in sweep.values()
                  if isinstance(v, dict) and v.get("compile_time_s", 0) > 0), None)
    if first:
        ct   = first["compile_time_s"]
        plan = first.get("stage_plan", "?")
        pred = first.get("cost_model_decode_ms", 0)
        meas = first.get("decode_lat_mean", 0)
        err  = abs(pred - meas) / meas * 100 if meas > 0 else 0
        lines.append(f"| {model.split('/')[-1]} | {ct:.1f}s | `{plan}` | {err:.1f}% |")
lines.append("")

lines += ["---", "*Generated by run_paper_experiments.sh*"]
print("\n".join(lines))
PYEOF
}

# ── Step 0: Environment setup ─────────────────────────────────────────────────

header "Step 0: Environment Setup"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/progress.jsonl"

log "Repo   : $REPO_ROOT"
log "Output : $OUT_DIR"
log "Quick  : $QUICK"
log "Sections: $SECTIONS"

# Apply MODELS_OVERRIDE if set
if [[ -n "$MODELS_OVERRIDE" ]]; then
    IFS=',' read -ra MODELS <<< "$MODELS_OVERRIDE"
    log "Models (override): ${MODELS[*]}"
fi

# Never install or sync packages from here: use the environment you prepared
# (`uv sync`, pip, Docker, or the research env from research/README.md).
"$PYTHON" -c "import helm, torch" || {
    log_err "HELM is not installed for $PYTHON. Activate your environment (e.g. source .venv/bin/activate) or set PYTHON=<interpreter>."
    exit 1
}
log "Using the active Python environment without installing or syncing packages."

# ── Step 1: Hardware detection ─────────────────────────────────────────────

header "Step 1: Hardware Detection"
detect_hardware
select_backends

# ── Step 2: Main experiment loops ────────────────────────────────────────────

header "Step 2: Experiments  (${#MODELS[@]} models × ${#BACKENDS[@]} backends)"

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL//\//-}"
    MODEL_DIR="$OUT_DIR/$MODEL_SLUG"
    mkdir -p "$MODEL_DIR"

    header "Model: $MODEL"

    # ── Disk space pre-flight check ───────────────────────────────────────────
    # Estimate model size from parameter count in the name (e.g. 4B, 8B, 32B).
    # Each parameter ≈ 2 bytes (float16). Require at least that much free disk
    # before allowing the download to start, so we never crash mid-download.
    _param_b=$(echo "$MODEL" | grep -oP '\d+(?=B)' | tail -1 || echo "0")
    _required_mb=$(( _param_b * 2 * 1024 ))   # params × 2 bytes → MB
    _hf_hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    mkdir -p "$_hf_hub"
    _free_mb=$(df "$_hf_hub" --output=avail -BM 2>/dev/null | tail -1 | tr -d 'M ' || echo "999999")
    if (( _required_mb > 0 && _free_mb < _required_mb )); then
        log_warn "  Skipping $MODEL — not enough disk space"
        log_warn "  Need ~${_required_mb}MB, only ${_free_mb}MB free on $( df "$_hf_hub" --output=target 2>/dev/null | tail -1 )"
        continue
    fi
    log "  Disk pre-flight: ~${_free_mb}MB free, need ~${_required_mb}MB — OK"

    # ── Per-backend sections (E F A B) ────────────────────────────────────────
    for BACKEND in "${BACKENDS[@]}"; do
        BDIR="$MODEL_DIR/$BACKEND"
        mkdir -p "$BDIR"

        log "── Backend: $BACKEND ──"

        # ── Section E: Feasibility ──────────────────────────────────────────
        if should_run_section "E"; then
            EDIR="$BDIR/E_feasibility"
            if is_done "$BDIR" "E"; then
                log "  [E] already done — skipping"
            else
                log "  [E] Feasibility probe …"
                run_section_E "$MODEL" "$BACKEND" "$EDIR" \
                    && mark_done "$BDIR" "E" \
                    || log_warn "  [E] failed — continuing"
                flush_memory
            fi
        fi

        # ── Section F: Max decode length (off by default — request with SECTIONS=...,F) ──
        if should_run_section "F"; then
            FDIR="$BDIR/F_max_decode"
            if is_done "$BDIR" "F"; then
                log "  [F] already done — skipping"
            else
                log "  [F] Max decode length probe …"
                run_section_F "$MODEL" "$BACKEND" "$FDIR" \
                    && mark_done "$BDIR" "F" \
                    || log_warn "  [F] failed — continuing"
                flush_memory
            fi
        fi

        # ── Section A: Latency sweep ─────────────────────────────────────────
        if should_run_section "A"; then
            ADIR="$BDIR/A_latency"
            if is_done "$BDIR" "A"; then
                log "  [A] already done — skipping"
            else
                log "  [A] Latency sweep (output_lens=$OUTPUT_LENS, N=$NUM_REQUESTS) …"
                run_section_A "$MODEL" "$BACKEND" "$ADIR" \
                    && mark_done "$BDIR" "A" \
                    || log_warn "  [A] failed — continuing"
                flush_memory
            fi
        fi

        # ── Section B: Throughput ────────────────────────────────────────────
        if should_run_section "B"; then
            BSDIR="$BDIR/B_throughput"
            if is_done "$BDIR" "B"; then
                log "  [B] already done — skipping"
            else
                log "  [B] Throughput sweep (concurrency=$THROUGHPUT_CONCURRENCY) …"
                run_section_B "$MODEL" "$BACKEND" "$BSDIR" \
                    && mark_done "$BDIR" "B" \
                    || log_warn "  [B] failed — continuing"
                flush_memory
            fi
        fi

    done   # end backends loop

    # ── Section C: HELM ablations (once per model, helm backend) ─────────────
    if should_run_section "C" && [[ " ${BACKENDS[*]} " == *" helm "* ]]; then
        CDIR="$MODEL_DIR/helm/C_ablations"
        CCHK="$MODEL_DIR/helm"
        mkdir -p "$CCHK"
        if is_done "$CCHK" "C"; then
            log "  [C] HELM ablations already done — skipping"
        else
            log "  [C] HELM ablations …"
            run_section_C "$MODEL" "$CDIR" \
                && mark_done "$CCHK" "C" \
                || log_warn "  [C] failed — continuing"
            flush_memory
        fi
    fi

    # ── Section D: LM quality benchmarks (once per model) ────────────────────
    if should_run_section "D" && [[ "$NO_LM_EVAL" != "1" ]]; then
        DDIR="$MODEL_DIR/D_lm_eval"
        DCHK="$MODEL_DIR"
        if is_done "$DCHK" "D"; then
            log "  [D] LM eval already done — skipping"
        else
            if "$PYTHON" -c "import lm_eval" &>/dev/null; then
                log "  [D] LM quality benchmarks (tasks=$LM_EVAL_TASKS) …"
                run_section_D "$MODEL" "${BACKENDS[0]}" "$DDIR" \
                    && mark_done "$DCHK" "D" \
                    || log_warn "  [D] failed — continuing"
                flush_memory
            else
                log_warn "  [D] lm-eval not installed — skipping (see research/README.md)"
            fi
        fi
    fi

    # ── Per-model summary ─────────────────────────────────────────────────────
    generate_model_summary "$MODEL" "$MODEL_SLUG" "$MODEL_DIR"
    log_ok "Model $MODEL complete → $MODEL_DIR"

    # ── Optionally evict this model's weights before downloading the next ─────
    if [[ "${DELETE_MODEL_CACHE:-0}" == "1" ]]; then
        delete_model_cache "$MODEL"
    fi

done   # end models loop

# ── Step 3: Overall summary ───────────────────────────────────────────────────

header "Step 3: Generating Summary"
generate_overall_summary
log_ok "Summary → $OUT_DIR/SUMMARY.md"
echo ""
cat "$OUT_DIR/SUMMARY.md"

# ── Final report ──────────────────────────────────────────────────────────────

echo ""
header "All Experiments Complete"
log "Results: $OUT_DIR"
log ""
log "Directory structure:"
log "  hardware.json            — machine specs"
log "  progress.jsonl           — append-only completion log"
log "  SUMMARY.md               — paper-ready tables"
for MODEL in "${MODELS[@]}"; do
    slug="${MODEL//\//-}"
    log "  $slug/"
    for BACKEND in "${BACKENDS[@]}"; do
        log "    $BACKEND/"
        log "      E_feasibility/  F_max_decode/  A_latency/  B_throughput/"
    done
    log "    helm/C_ablations/"
    log "    D_lm_eval/"
done
