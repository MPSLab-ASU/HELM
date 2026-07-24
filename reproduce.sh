#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — one-command reproduction driver for the HELM paper results.
#
#   bash reproduce.sh <target>       # run experiments AND render the
#                                    # corresponding table/figure comparison
#   bash reproduce.sh                # print target list + time estimates
#
# Each experiment target (table2/table3/fig4/fig5/fig6) runs its measurements
# and then automatically invokes results/make_figures.py on the run's output
# directory, so the paper-vs-measured PNG + markdown land in <run_root>/figures/.
#
# Works natively (uv or pip environment) and inside the Docker images built
# from the repository Dockerfile. Per-claim details, expected outputs and
# tolerances: REPRODUCING.md.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
    cat <<'EOF'
HELM artifact reproduction driver.

bash reproduce.sh <target>

Targets:
  test     CPU-only functional check (tests + canonical self-check)
  smoke    small end-to-end GPU run + correctness check
  table2   Table II experiments  -> renders Table II paper-vs-measured
  table3   Table III experiments -> renders Table III comparison
  fig4     Fig. 4 experiments    -> renders Fig. 4 comparison
  fig5     Fig. 5 experiments    -> renders Fig. 5 comparison
  fig6     Fig. 6 experiments    -> renders Fig. 6 comparison
  all      test + smoke + every table/figure target in sequence

Every experiment target finishes by rendering its paper-vs-measured
comparison (PNG + markdown companion in <run_root>/figures/) from the run it
just produced, via results/make_figures.py.

Details (what each target runs -> approx. time):

  test     pytest tests/ -q (GPU-gated tests skip on CPU-only hosts,
           and the AVX2 kernel tests skip on non-AVX2/ARM hosts), then
           python results/validate_results.py (canonical-results
           self-check). No GPU, no model downloads.                  ~5 min

  smoke    Qwen3-4B end-to-end partitioned run (auto plan, KV offload),
           then correctness vs. the HuggingFace baseline
           (experiments/verify_e2e.py, Qwen3-0.6B). Downloads ~9 GB
           of weights on first use.                                 ~30 min

  table2   Table II decode-throughput/TTFT grid:
           experiments/run_paper_experiments.sh (sections E,A,B,C;
           INPUT_LEN=128 PAD_INPUT=1, the paper's operating point).
           vLLM/DeepSpeed baselines run only if installed
           (research/README.md); otherwise they are skipped.
           Resumable: rerun with the OUT_ROOT it prints.  hours per GPU class

  table3   llama.cpp comparison (needs llama-cpp-python and a
           Llama-2-13B GGUF; set GGUF=<path>, GGUF_TAG=fp16|q4_k_m):
           experiments/llama_cpp_bench.py --n-gpu-layers sweep. ~30-60 min

  fig4     TTFT vs. prompt length, Qwen3-32B (needs ~70 GB free RAM):
           experiments/ttft_prompt_scaling.sh + summarizer.             ~1-2 h

  fig5     Max-context extension (slow OOM probes):
           SECTIONS=F experiments/run_paper_experiments.sh.            hours

  fig6     Long-context KV paging, Mistral-Nemo-12B:
           experiments/longctx_paging.sh + summarizer.       ~1-2 h

  all      test, smoke, then table2, fig4, fig5, fig6, table3 —
           rendering after each. A failing target does not abort the
           sequence; GPU targets are SKIPPED when no CUDA device is
           visible and table3 is SKIPPED when GGUF is unset. Ends
           with a per-target PASS/FAIL/SKIPPED summary.

Utilities:

  figures [run_root]
           Re-render only (no experiments): all tables/figures
           (Table II/III, Fig. 4-6) as PNG + markdown from
           results/paper/ into results/figures/ (needs matplotlib,
           no GPU). With a run_root, overlay the measured outputs a
           previous target wrote under it, paper vs measured, into
           <run_root>/figures/.                                      seconds

  validate <run> "<GPU>" <Model>
           Compare a fresh run against its Table II cell:
           results/validate_results.py --check-run.                  seconds
           <run>   a paper_results.json or a directory containing one
           <GPU>   "RTX 4060" | "RTX 3090" | "L40S"
           <Model> e.g. Qwen3-8B   (TOLERANCE=0.25 to override)

Notes:
  - GPU targets fail fast if no CUDA device is visible. In Docker add
    --gpus all and mount the model cache:
      -v $HOME/.cache/huggingface:/root/.cache/huggingface
  - QUICK=1 shrinks table2/fig4/fig6 to smoke size; each driven script
    documents further env overrides in its header.
  - OUT_ROOT=<dir> pins a target's output directory (table2/fig5: pass
    the same dir again to resume a partial grid). Under 'all' each
    target always gets its own fresh directory.
  - Experiment drivers use the same Python this script selects (the
    project .venv from `uv sync`, else the active interpreter); nothing
    is installed behind your back. Set PYTHON=<interpreter> to override.
EOF
}

die() { echo "reproduce.sh: ERROR: $*" >&2; exit 1; }

# Python with the helm package *installed*: an explicit $PYTHON, then the
# project venv ($ROOT/.venv, what `uv sync` creates), then the active
# interpreter (Docker image, activated venv/conda), else create .venv from
# uv.lock with `uv sync --frozen`. The import probe runs from / — from the repo
# root ANY interpreter can "import helm" via the local source directory, which
# would silently select an unrelated python without the pinned dependencies.
PY=""
has_helm() {
    (cd / && "$1" -c "import helm") >/dev/null 2>&1
}
find_python() {
    [[ -n "$PY" ]] && return 0
    local cand
    if [[ -n "${PYTHON:-}" ]] && has_helm "$PYTHON"; then
        PY="$(command -v "$PYTHON")"
    elif [[ -x "$ROOT/.venv/bin/python" ]] && has_helm "$ROOT/.venv/bin/python"; then
        PY="$ROOT/.venv/bin/python"
    else
        for cand in python python3; do
            if command -v "$cand" >/dev/null 2>&1 && has_helm "$cand"; then
                PY="$(command -v "$cand")"
                break
            fi
        done
    fi
    if [[ -z "$PY" ]] && command -v uv >/dev/null 2>&1 && [[ -f uv.lock ]]; then
        echo "reproduce.sh: no environment with 'helm' found; running 'uv sync --frozen'" >&2
        uv sync --frozen >&2 && has_helm "$ROOT/.venv/bin/python" && PY="$ROOT/.venv/bin/python"
    fi
    [[ -n "$PY" ]] || return 1
    # The experiment drivers call `python` from PATH; make it this interpreter.
    PATH="$(dirname "$PY"):$PATH"
    export PATH
    export PYTHON="$PY"
    return 0
}

pick_python() {
    find_python || die "no Python environment with the 'helm' package found.
  Install one of:   uv sync                  (native, pinned lockfile)
                    pip install -e '.[dev]'
  or run inside the Docker image (see README.md 'Reproducing the paper')."
}

# Non-fatal CUDA probe (used by 'all' to decide SKIPPED vs run).
have_cuda() {
    find_python || return 1
    if $PY -c "import torch" >/dev/null 2>&1; then
        $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    else
        command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
    fi
}

need_cuda() {
    pick_python
    have_cuda || die "target '$1' needs a CUDA GPU (torch.cuda.is_available() is False
  or no NVIDIA device found).
  In Docker: run with --gpus all (NVIDIA container toolkit required) and a
  cu130 image (docker build -t helm .), not the CPU image.
  CPU-only functional check: bash reproduce.sh test"
}

# Python with matplotlib for rendering (make_figures.py needs neither the
# helm package nor torch); fall back to the uv env.
FIGPY=""
pick_figpy() {
    [[ -n "$FIGPY" ]] && return 0
    local cand
    for cand in "${PYTHON:-}" "$ROOT/.venv/bin/python" python python3; do
        [[ -n "$cand" ]] || continue
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import matplotlib" >/dev/null 2>&1; then
            FIGPY="$cand"
            return 0
        fi
    done
    return 1
}

# render_target <target> <run_root>: overlay the measured run onto the
# paper's table/figure and print where the rendered comparison landed.
render_target() {
    local target="$1" run_root="$2" base
    case "$target" in
        table2) base="table2_decode_throughput" ;;
        table3) base="table3_llamacpp" ;;
        fig4)   base="fig4_ttft_prompt_scaling" ;;
        fig5)   base="fig5_max_output_tokens" ;;
        fig6)   base="fig6_paging_throughput" ;;
        *)      die "render_target: unknown target '$target'" ;;
    esac
    pick_figpy || die "rendering '$target' needs matplotlib.
  Install it with:  pip install -e '.[dev]'   or   uv sync
  The measurements are safe in $run_root — re-render any time with:
    python results/make_figures.py $target --run-root '$run_root'"
    echo ""
    # Measured renders go next to the run, never over the committed
    # paper-reference renditions in results/figures/.
    local fig_dir="$run_root/figures"
    echo "== render: python results/make_figures.py $target --run-root $run_root --out-dir $fig_dir =="
    $FIGPY results/make_figures.py "$target" --run-root "$run_root" --out-dir "$fig_dir"
    echo ""
    echo "Rendered paper-vs-measured comparison for $target:"
    echo "  $fig_dir/$base.png"
    echo "  $fig_dir/$base.md"
    echo "(measured run kept in: $run_root)"
}

stage_test() {
    pick_python
    echo "== [1/2] test suite: pytest tests/ -q (CPU-only, ~5 min) =="
    $PY -m pytest tests/ -q
    echo "== [2/2] canonical-results self-check: results/validate_results.py =="
    $PY results/validate_results.py
    echo "OK: functional check passed."
}

stage_smoke() {
    need_cuda smoke
    pick_python
    echo "== [1/2] end-to-end partitioned run: Qwen3-4B, auto plan, KV offload =="
    $PY -m helm.cli --model Qwen/Qwen3-4B --mode execute_stagewise \
        --compiler-plan auto --max-new-tokens 64 --kv-offload
    echo "== [2/2] correctness vs. HuggingFace baseline (Qwen3-0.6B) =="
    $PY experiments/verify_e2e.py --model Qwen/Qwen3-0.6B
    echo "OK: smoke run + correctness check passed."
}

stage_table2() {
    need_cuda table2
    # Dedicated per-invocation root: run_paper_experiments.sh nests its own
    # <timestamp>/ under OUT_ROOT, and make_figures.py discovers
    # **/paper_results.json + hardware.json recursively under --run-root.
    local run_root="${OUT_ROOT:-experiments/results/table2_$(date +%Y%m%d_%H%M%S)}"
    echo "table2: measured outputs -> $run_root"
    echo "        (resume a partial grid later with OUT_ROOT=$run_root bash reproduce.sh table2)"
    OUT_ROOT="$run_root" INPUT_LEN="${INPUT_LEN:-128}" PAD_INPUT="${PAD_INPUT:-1}" \
        bash experiments/run_paper_experiments.sh
    render_target table2 "$run_root"
    echo "Per-cell check: bash reproduce.sh validate '$run_root' \"<GPU>\" <Model>"
}

stage_fig4() {
    need_cuda fig4
    # ttft_prompt_scaling.sh writes <OUT_ROOT>/<backend>/in<len>/paper_results.json,
    # exactly the layout make_figures.py fig4 discovery expects.
    export OUT_ROOT="${OUT_ROOT:-experiments/results/ttft_prompt_scaling_$(date +%Y%m%d_%H%M%S)}"
    bash experiments/ttft_prompt_scaling.sh
    pick_python
    $PY experiments/ttft_prompt_scaling_summarize.py "$OUT_ROOT"
    render_target fig4 "$OUT_ROOT"
}

stage_fig5() {
    need_cuda fig5
    # Same driver as table2 (section F only); dedicated root for discovery.
    local run_root="${OUT_ROOT:-experiments/results/fig5_$(date +%Y%m%d_%H%M%S)}"
    echo "fig5: measured outputs -> $run_root"
    echo "      (resume later with OUT_ROOT=$run_root bash reproduce.sh fig5)"
    OUT_ROOT="$run_root" SECTIONS=F INPUT_LEN="${INPUT_LEN:-128}" PAD_INPUT="${PAD_INPUT:-1}" \
        bash experiments/run_paper_experiments.sh
    render_target fig5 "$run_root"
}

stage_fig6() {
    need_cuda fig6
    # The paper's Fig. 6 run is Mistral-Nemo-12B (results/paper/fig6_paging_throughput.json);
    # the script's own default is the Llama-2-13B paging config, so pin the model here.
    # longctx_paging.sh writes <OUT_ROOT>/ctx_<len>/paper_results.json,
    # exactly the layout make_figures.py fig6 discovery expects.
    export MODEL="${MODEL:-mistralai/Mistral-Nemo-Instruct-2407}"
    export OUT_ROOT="${OUT_ROOT:-experiments/results/longctx_paging_$(date +%Y%m%d_%H%M%S)}"
    bash experiments/longctx_paging.sh
    pick_python
    $PY experiments/longctx_paging_summarize.py "$OUT_ROOT"
    render_target fig6 "$OUT_ROOT"
}

stage_table3() {
    need_cuda table3
    pick_python
    $PY -c "import llama_cpp" >/dev/null 2>&1 \
        || die "target 'table3' needs llama-cpp-python (not part of the pinned deps).
  Install a CUDA build, e.g.:
    CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
    [[ -n "${GGUF:-}" ]] || die "target 'table3' needs GGUF=<path to a Llama-2-13B .gguf>.
  Example:
    huggingface-cli download TheBloke/Llama-2-13B-GGUF llama-2-13b.Q4_K_M.gguf \\
        --local-dir experiments/gguf_local
    GGUF=experiments/gguf_local/llama-2-13b.Q4_K_M.gguf GGUF_TAG=q4_k_m \\
        bash reproduce.sh table3"
    [[ -f "$GGUF" ]] || die "GGUF file not found: $GGUF"
    # llama_cpp_bench.py writes <out>/results.json, which make_figures.py
    # table3 discovery picks up recursively under --run-root.
    local out="${OUT_ROOT:-experiments/results/table3_llamacpp_$(date +%Y%m%d_%H%M%S)}"
    $PY experiments/llama_cpp_bench.py --local-gguf "$GGUF" \
        --local-tag "${GGUF_TAG:-q4_k_m}" --n-gpu-layers -1 0 20 30 34 \
        --output-dir "$out"
    echo "llama.cpp side done: $out/results.json"
    echo "HELM side of Table III (optional; its paper_results.json lands under"
    echo "  experiments/results/ — re-render with 'bash reproduce.sh figures <dir>'):"
    echo "  HF_HUB_OFFLINE=0 bash experiments/run_llama13b_helm.sh"
    render_target table3 "$out"
}

stage_all() {
    local -a targets=(test smoke table2 fig4 fig5 fig6 table3)
    local -a summary=()
    local t rc cuda_ok=0 fails=0
    if have_cuda; then cuda_ok=1; fi
    for t in "${targets[@]}"; do
        if [[ "$t" != "test" && "$cuda_ok" -eq 0 ]]; then
            echo ""
            echo "=== all: $t SKIPPED (no CUDA device visible) ==="
            summary+=("$t: SKIPPED (no CUDA device)")
            continue
        fi
        if [[ "$t" == "table3" && -z "${GGUF:-}" ]]; then
            echo ""
            echo "=== all: table3 SKIPPED (GGUF not set; see 'bash reproduce.sh' usage) ==="
            summary+=("table3: SKIPPED (GGUF not set)")
            continue
        fi
        echo ""
        echo "=== all: target '$t' ==="
        # Subshell so one failing target cannot abort the sequence; fresh
        # OUT_ROOT per target so outputs never mix.
        set +e
        ( set -e; unset OUT_ROOT; "stage_$t" )
        rc=$?
        set -e
        if [[ "$rc" -eq 0 ]]; then
            summary+=("$t: PASS")
        else
            summary+=("$t: FAIL (exit $rc)")
            fails=$((fails + 1))
        fi
    done
    echo ""
    echo "=== all: per-target summary ==="
    printf '  %s\n' "${summary[@]}"
    if [[ "$fails" -gt 0 ]]; then
        echo "all: $fails target(s) failed."
        exit 1
    fi
    echo "all: every executed target passed."
}

stage_figures() {
    local run_root="${1:-}"
    pick_figpy || die "'figures' needs matplotlib.
  Install it with:  pip install -e '.[dev]'   or   uv sync"
    if [[ -n "$run_root" ]]; then
        [[ -d "$run_root" ]] || die "figures: run root is not a directory: $run_root"
        $FIGPY results/make_figures.py all --run-root "$run_root" --out-dir "$run_root/figures"
        echo "PNGs + markdown companions (paper vs measured): $run_root/figures/"
    else
        $FIGPY results/make_figures.py all
        echo "PNGs + markdown companions (paper reference): results/figures/"
    fi
}

stage_validate() {
    local run="${1:-}" gpu="${2:-}" model="${3:-}"
    [[ -n "$run" && -n "$gpu" && -n "$model" ]] \
        || die "usage: bash reproduce.sh validate <run_dir_or_paper_results.json> \"<GPU>\" <Model>
  e.g.: bash reproduce.sh validate \\
            'experiments/results/<ts>/Qwen-Qwen3-8B/helm/A_latency' 'RTX 4060' Qwen3-8B"
    local json="$run"
    if [[ -d "$run" ]]; then
        local -a found=()
        local f
        while IFS= read -r f; do found+=("$f"); done \
            < <(find "$run" -name paper_results.json | sort)
        (( ${#found[@]} )) || die "no paper_results.json found under $run"
        if (( ${#found[@]} > 1 )); then
            # Multiple runs under this directory: keep only the ones whose
            # path mentions <Model> so the right cell gets compared.
            local -a matched=()
            for f in "${found[@]}"; do
                case "$(tr '[:upper:]' '[:lower:]' <<<"$f")" in
                    *"$(tr '[:upper:]' '[:lower:]' <<<"$model")"*) matched+=("$f") ;;
                esac
            done
            (( ${#matched[@]} )) && found=("${matched[@]}")
        fi
        (( ${#found[@]} == 1 )) \
            || die "ambiguous run directory: ${#found[@]} paper_results.json files match under $run:
$(printf '  %s\n' "${found[@]}")  pass the per-model run directory (or the .json file) instead."
        json="${found[0]}"
    fi
    [[ -f "$json" ]] || die "file not found: $json"
    pick_python
    # shellcheck disable=SC2086
    $PY results/validate_results.py --check-run "$json" --gpu "$gpu" --model "$model" \
        ${TOLERANCE:+--tolerance "$TOLERANCE"}
}

main() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        usage
        exit 0
    fi
    shift || true
    case "$target" in
        test)             stage_test ;;
        smoke)            stage_smoke ;;
        table2)           stage_table2 ;;
        table3)           stage_table3 ;;
        fig4)             stage_fig4 ;;
        fig5)             stage_fig5 ;;
        fig6)             stage_fig6 ;;
        all)              stage_all ;;
        figures)          stage_figures "$@" ;;
        validate)         stage_validate "$@" ;;
        -h|--help|help)   usage ;;
        *)                usage; die "unknown target '$target'" ;;
    esac
}

main "$@"
