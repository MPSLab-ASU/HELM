# How to Read This Codebase

This guide is for anyone new to HELM who wants to understand what it does and where to start reading. It assumes you know Python and PyTorch basics but have not touched this codebase before.

---

## What is HELM in one paragraph

HELM runs LLMs that are **too large for a single GPU** on consumer hardware (one GPU + CPU RAM). It does this by:
1. Tracing the model's forward pass with PyTorch FX
2. Microbenchmarking your actual hardware (GPU FLOPS, memory bandwidth, PCIe speed)
3. Searching all possible CPU/GPU layer split points under a roofline cost model
4. Compiling the model into two isolated subgraphs: one runs on CPU, one on GPU
5. Running them in sequence — only one activation tensor crosses PCIe per token, never weights

If the model fits in GPU VRAM, use vLLM instead — HELM pays off only when the model exceeds VRAM.

---

## Before you read any code, read these two files

| File | Why |
|---|---|
| [README.md](README.md) | The 10-step pipeline diagram + performance numbers. Gives you the mental model before you see implementation. |
| [docs/CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md) | Deep-dive on every module. Use as reference once you have the mental model. |

---

## Reading order — from concept to implementation

Follow this sequence. Each step builds on the previous.

### Step 1 — Understand the IR (5 min)
**File:** [helm/compiler/IR/graph.py](helm/compiler/IR/graph.py)

This is the typed intermediate representation that everything else operates on. A `HelmNode` wraps an FX node and adds cost metadata (FLOP count, param bytes, activation bytes). A `HelmGraph` is a list of `HelmNode`s with edges. Read the dataclasses — no logic to trace yet, just shapes.

### Step 2 — Understand how the model is traced (10 min)
**File:** [helm/compiler/importers/decode_tracer.py](helm/compiler/importers/decode_tracer.py)

HELM traces the **decode step** (seq_len=1) rather than prefill. This file builds the wrapper that holds the KV cache as a constant inside the traced graph. The trace is captured once; the same compiled graph runs both prefill (prompt) and decode (token-by-token).

### Step 3 — Understand how semantic meaning is added (10 min)
**File:** [helm/compiler/importers/fx_importer.py](helm/compiler/importers/fx_importer.py)

Raw FX nodes have no semantic labels. This file classifies each node: is it a projection weight? An attention op? An embedding? It lifts the raw FX graph into the typed `HelmGraph` IR from Step 1.

### Step 4 — Understand cost analysis (10 min)
**File:** [helm/compiler/analysis/hybrid_analyzer.py](helm/compiler/analysis/hybrid_analyzer.py)

Walks the `HelmGraph` and annotates every node with `param_bytes`, `activation_bytes`, and `kv_bytes_per_token`. These numbers feed the roofline model.

### Step 5 — Understand coarse grouping (5 min)
**File:** [helm/compiler/partition/partition_units.py](helm/compiler/partition/partition_units.py)

Groups IR nodes into coarse `PartitionUnit`s: one embedding unit, L transformer-block units, one LM-head unit. The split point search operates over these units, not individual nodes.

### Step 6 — Understand the cost model (10 min)
**File:** [helm/compiler/optimization/cost_model.py](helm/compiler/optimization/cost_model.py)

The roofline model. Given a device profile (FLOPS, bandwidth, PCIe throughput), estimates decode latency for a given CPU/GPU partition. Separate rooflines for projection (memory-bound GEMV) vs attention (compute-bound at long context).

### Step 7 — Understand strategy selection (5 min)
**File:** [helm/compiler/optimization/strategy_selector.py](helm/compiler/optimization/strategy_selector.py)

Iterates all O(L) feasible split points, scores each with the cost model, returns the best. Runs in under 1 ms.

### Step 8 — Understand how the FX graph is split (10 min)
**File:** [helm/compiler/lowering/stage_fx_builder.py](helm/compiler/lowering/stage_fx_builder.py)

Takes the original FX graph and the chosen split point, produces two independent `GraphModule`s. Each is a standalone PyTorch model — no shared state, no hooks.

### Step 9 — Read the compiler orchestrator (10 min)
**File:** [helm/compiler/compiler.py](helm/compiler/compiler.py)

Calls all of the above in sequence. The entry point is `compile_graph()`. The output is a `CompiledHelmArtifact` containing the stage subgraphs and the partition plan.

### Step 10 — Understand how stages execute (10 min)
**File:** [helm/runtime/executor.py](helm/runtime/executor.py)

`StageRuntimeExecutor` runs a list of `Stage`s in sequence. It calls each stage's `GraphModule`, transfers the output activation tensor to the next stage's device, and returns logits.

### Step 11 — Understand the generation loop (10 min)
**File:** [helm/runtime/pipeline_runtime.py](helm/runtime/pipeline_runtime.py)

`PipelineRuntime.generate()` is the top-level API. It runs prefill (full prompt in one pass), then decode (one token per iteration), managing attention masks and the KV cache.

### Step 12 — Understand KV offloading (15 min, only if you need long context)
Files in order:
- [helm/runtime/kv_allocator.py](helm/runtime/kv_allocator.py) — page pool management
- [helm/runtime/kv_cache.py](helm/runtime/kv_cache.py) — paged KV storage + streaming attention
- [helm/runtime/kv_offload.py](helm/runtime/kv_offload.py) — patches attention `forward` at the class level to intercept KV read/write

---

## How to run it quickly

```bash
# Install (requires uv)
uv sync

# Inspect partition plan without executing (safe, no generation)
uv run helm --model Qwen/Qwen3-8B --mode plan --compiler-plan auto --print-plan

# Full generation with auto-partition
uv run helm --model Qwen/Qwen3-8B --mode execute_stagewise --compiler-plan auto --max-new-tokens 64

# With KV offloading for long context
uv run helm --model Qwen/Qwen3-8B --mode execute_stagewise --compiler-plan auto --max-new-tokens 64 --kv-offload
```

See `helm/_pipeline.py` (the module behind the `helm` CLI; `experiments/dev_pipeline.py` is a compatibility entrypoint) for a complete programmatic example, and `experiments/verify_e2e.py` for the canonical end-to-end correctness check.

---

## Key concepts and where they live

| Concept | File | What to look for |
|---|---|---|
| Traced decode wrapper | `decode_tracer.py` | `_Wrapper` class and `DecodeTracer.trace()` |
| IR node / graph | `IR/graph.py` | `HelmNode`, `HelmGraph` dataclasses |
| Cost metadata per node | `hybrid_analyzer.py` | `HybridAnalyzer.run()` return value |
| Device microbenchmark | `device_profiler.py` | `profile_devices()`, cached in `.helm_profile.json` |
| Roofline score | `cost_model.py` | `HelmCostModel.estimate_plan()` |
| Split point search | `strategy_selector.py` | `StrategySelector.select()` |
| FX graph splitting | `stage_fx_builder.py` | `StageFXBuilder.build()` |
| Stage execution | `executor.py` | `StageRuntimeExecutor.run()` |
| Generation loop | `pipeline_runtime.py` | `PipelineRuntime.generate()` |
| KV page eviction | `kv_cache.py` | `KVCacheManager.evict_pages()` |
| Attention patching | `kv_offload.py` | `KVOffloadManager.patch_module_roots()` |

---

## Where to look for known issues

See the "Known Limitations" section of [docs/CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md).

---

## Tests as living documentation

If any module's behavior is unclear, read its test first:

| Module | Test file |
|---|---|
| IR graph | `tests/unit/compiler/test_ir_graph.py` |
| FX importer | `tests/unit/compiler/test_fx_importer.py` |
| Strategy selector | `tests/unit/compiler/test_strategy_selector.py` |
| KV cache | `tests/unit/runtime/test_kv_cache.py` |
| KV offload | `tests/unit/runtime/test_kv_offload.py` |
| Pipeline runtime | `tests/unit/runtime/test_pipeline_runtime.py` |
| Full compiler pipeline | `tests/integration/test_compiler_pipeline.py` |
| Full runtime pipeline | `tests/integration/test_runtime_pipeline.py` |

Run all tests with:
```bash
uv run pytest tests/ -x -q
```
