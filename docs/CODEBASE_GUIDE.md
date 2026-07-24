# HELM Codebase Guide

**HELM (Heterogeneous Execution for Large Models)** is a compiler + runtime that runs LLMs too large for a single GPU by partitioning them across CPU RAM and GPU VRAM. This guide explains every file, class, and key function, and how they connect.

---

## Table of Contents

1. [What HELM Does — The Big Picture](#1-what-helm-does)
2. [Repository Layout](#2-repository-layout)
3. [Reading Order — How to Explore the Code](#3-reading-order)
4. [Compiler Pipeline](#4-compiler-pipeline)
   - [Entry Point: `compiler.py`](#41-entry-point-compilerpy)
   - [FX Tracing: `decode_tracer.py`](#42-fx-tracing-decode_tracerpy)
   - [Graph IR: `graph.py`](#43-graph-ir-graphpy)
   - [FX Importer: `fx_importer.py`](#44-fx-importer-fx_importerpy)
   - [Cost Analysis: `hybrid_analyzer.py`](#45-cost-analysis-hybrid_analyzerpy)
   - [Partition Units: `partition_units.py`](#46-partition-units-partition_unitspy)
   - [Device Profiler: `device_profiler.py`](#47-device-profiler-device_profilerpy)
   - [Cost Model: `cost_model.py`](#48-cost-model-cost_modelpy)
   - [Strategy Selector: `strategy_selector.py`](#49-strategy-selector-strategy_selectorpy)
   - [Partition Plan: `partition_plan.py`](#410-partition-plan-partition_planpy)
   - [Stage Builder: `stage_fx_builder.py`](#411-stage-builder-stage_fx_builderpy)
5. [Runtime Execution](#5-runtime-execution)
   - [Pipeline Runtime: `pipeline_runtime.py`](#51-pipeline-runtime-pipeline_runtimepy)
   - [Stage Executor: `executor.py`](#52-stage-executor-executorpy)
   - [KV Allocator: `kv_allocator.py`](#53-kv-allocator-kv_allocatorpy)
   - [KV Cache: `kv_cache.py`](#54-kv-cache-kv_cachepy)
   - [KV Offload: `kv_offload.py`](#55-kv-offload-kv_offloadpy)
   - [Tensor Transfer: `tensor_transfer.py`](#56-tensor-transfer-tensor_transferpy)
   - [Stage: `stage.py`](#57-stage-stagepy)
6. [CLI: `cli.py`](#6-cli-clipy)
7. [Experiments & Benchmarks](#7-experiments--benchmarks)
8. [Tests](#8-tests)
9. [Key Algorithms Explained](#9-key-algorithms-explained)
10. [Data Flow Diagram](#10-data-flow-diagram)
11. [Known Limitations](#11-known-limitations)

---

## 1. What HELM Does

Given a model like Qwen3-14B (28 GB weights) on a machine with 24 GB GPU VRAM:

1. **Traces** the model's decode step as a single FX graph
2. **Analyzes** each node's compute and memory cost
3. **Profiles** your actual hardware (not hardcoded constants)
4. **Searches** all O(L) possible CPU/GPU split points
5. **Picks** the split that minimises decode latency under a roofline model
6. **Compiles** the FX graph into two standalone `GraphModule`s: one for CPU, one for GPU
7. **Executes** them in sequence: CPU stage runs first, passes activations to GPU stage
8. Optionally **offloads KV cache** pages to CPU RAM to extend context length

**Key insight:** weights never move during decode. Only one small activation tensor (hidden_states) crosses the PCIe bus per token, per stage boundary. This is fundamentally different from DeepSpeed (streams weights every layer) or Accelerate (no cost model, no isolation).

---

## 2. Repository Layout

```
HELM/
├── helm/                        # All production code
│   ├── cli.py                   # CLI entry point
│   ├── __init__.py
│   ├── compiler/                # Compilation pipeline
│   │   ├── compiler.py          # Main orchestrator
│   │   ├── IR/
│   │   │   └── graph.py         # HelmGraph typed IR
│   │   ├── importers/
│   │   │   ├── decode_tracer.py # FX tracing with KV cache
│   │   │   ├── fx_importer.py   # Semantic annotation
│   │   │   ├── attention_capture.py  # Q/K/V capture during trace
│   │   │   └── patch_fx.py      # FX workarounds
│   │   ├── analysis/
│   │   │   └── hybrid_analyzer.py    # Shape + cost analysis
│   │   ├── partition/
│   │   │   ├── partition_units.py    # Group nodes into coarse units
│   │   │   └── partition_plan.py     # Plan data structures
│   │   ├── optimization/
│   │   │   ├── device_profiler.py    # Hardware microbenchmarks
│   │   │   ├── cost_model.py         # Roofline cost estimation
│   │   │   └── strategy_selector.py  # Exhaustive plan search
│   │   ├── lowering/
│   │   │   └── stage_fx_builder.py   # Fragment FX → per-device GraphModules
│   │   └── scheduling/
│   │       └── scheduler.py          # Execution schedule generation
│   ├── runtime/
│   │   ├── pipeline_runtime.py  # High-level prefill + decode loop
│   │   ├── executor.py          # Per-stage execution engine
│   │   ├── kv_cache.py          # Paged KV storage + streaming attention
│   │   ├── kv_allocator.py      # Page pool management
│   │   ├── kv_offload.py        # Class-level attention patching
│   │   ├── tensor_transfer.py   # Cross-device tensor movement
│   │   └── stage.py             # Stage dataclass
│   ├── kernels/
│   │   └── __init__.py          # Optional AVX2 C++ extension loader
│   └── _pipeline.py             # Quick iterative testing / diagnostic pipeline (packaged, ships in the wheel)
├── experiments/
│   ├── dev_pipeline.py          # Compatibility shim — re-exports helm/_pipeline.py
│   ├── paper_bench.py           # Full benchmark suite (vLLM/Accelerate/DeepSpeed/HELM)
│   ├── profile_pipeline.py      # Per-stage profiling
├── tests/
│   ├── conftest.py              # Shared fixtures (TinyTransformer, cost model, etc.)
│   ├── unit/
│   │   ├── compiler/            # Unit tests per compiler module
│   │   └── runtime/             # Unit tests per runtime module
│   └── integration/
│       ├── test_compiler_pipeline.py
│       └── test_runtime_pipeline.py
├── README.md                    # Usage & results
└── pyproject.toml               # Dependencies
```

---

## 3. Reading Order

If you're new to the codebase, read files in this order:

1. **`README.md`** — overview + performance numbers
2. **`helm/compiler/IR/graph.py`** — understand the IR (HelmNode, HelmEdge, HelmGraph) before anything else
3. **`helm/compiler/importers/decode_tracer.py`** — how a model gets traced into an FX graph
4. **`helm/compiler/importers/fx_importer.py`** — how semantic meaning is layered on top
5. **`helm/compiler/analysis/hybrid_analyzer.py`** — how costs are estimated
6. **`helm/compiler/partition/partition_units.py`** — coarse grouping for planning
7. **`helm/compiler/optimization/cost_model.py`** — the roofline model
8. **`helm/compiler/optimization/strategy_selector.py`** — how the best plan is chosen
9. **`helm/compiler/lowering/stage_fx_builder.py`** — how the FX graph is split
10. **`helm/compiler/compiler.py`** — the orchestrator that calls all of the above
11. **`helm/runtime/executor.py`** — how stages are executed
12. **`helm/runtime/pipeline_runtime.py`** — the full generation loop
13. **`helm/runtime/kv_cache.py`** → **`kv_allocator.py`** → **`kv_offload.py`** — KV offloading

---

## 4. Compiler Pipeline

### 4.1 Entry Point: `compiler.py`

**Path:** `helm/compiler/compiler.py`
**Role:** Orchestrates the entire compilation from raw model to `CompiledHelmArtifact`.

#### Key dataclasses

| Class | Purpose |
|---|---|
| `HelmCompileOptions` | All compile knobs: mode, objective, plan_mode (auto/manual), cpu/gpu layer ranges, kv_offload flag, dtype |
| `HelmIR` | Serializable IR snapshot: schema_version, graph_kind, nodes, units, plan, schedule, hardware, metadata |
| `CompiledHelmArtifact` | Final output: compiled GraphModule, cost analysis, partition plan, stage graphs |

#### Key functions

| Function | What it does |
|---|---|
| `parse_layer_spec(spec)` | Converts `"0:7"` → `(0, 7)`. Used for manual CPU/GPU layer range flags. |
| `_build_helm_graph(gm)` | Wraps an FX `GraphModule` into a `HelmGraph` IR. Calls `FXImporter` to add semantic metadata. |
| `_run_analysis(gm, helm_graph, model, example_inputs)` | Runs `HybridAnalyzer` to populate each node's shape/cost fields. |
| `_run_static_fallback_analysis()` | Conservative cost estimates when shape propagation fails (e.g. dynamic shapes). |
| `compile_graph()` | **Main entry point.** Calls tracer → importer → analyzer → partition builder → device profiler → strategy selector → stage lowering. Returns `CompiledHelmArtifact`. |
| `helm_backend(gm, example_inputs)` | `torch.compile` integration hook. Calls `compile_graph()`. **Warning:** silently swallows exceptions — see BUG-1. |

#### Flow inside `compile_graph()`:

```
1. DecodeTracer.trace(model)          → FX GraphModule
2. _build_helm_graph(gm)              → HelmGraph (with semantic tags)
3. _run_analysis(...)                 → HelmGraph (with cost metadata)
4. PartitionUnitBuilder.build()       → List[PartitionUnit]
5. profile_devices(...)               → DeviceProfile (CPU) + DeviceProfile (GPU)
6. StrategySelector.select(...)       → (PartitionPlan, PlanCost)
7. StageFXBuilder.build(gm, plan)     → List[Stage]  (per-device GraphModules)
8. Return CompiledHelmArtifact
```

---

### 4.2 FX Tracing: `decode_tracer.py`

**Path:** `helm/compiler/importers/decode_tracer.py`
**Role:** Produces a single FX graph that captures one decode step (seq_len=1), which is reused at runtime for both prefill and decode.

#### Why this is hard

PyTorch FX works by symbolic tracing through Python code. Two things break vanilla tracing for transformer decode:
1. **`DynamicCache.update()`** calls `list.append()` — Python containers can't be symbolically traced.
2. Separate decoder-layer submodules should stay as atomic leaves (not expanded) to allow clean per-layer partitioning.

#### Key inner classes

| Class | Purpose |
|---|---|
| `DecodeTracer._HelmTracer(fx.Tracer)` | Custom FX Tracer. Marks `EmbeddingLayer`, `RotaryEmbedding`, `lm_head`, and `DecoderLayer` submodules as **leaves** — FX records a `call_module` node instead of tracing inside them. |
| `DecodeTracer._Wrapper(nn.Module)` | Wraps the model. Registers `forward_pre_hooks` on each `DecoderLayer` that inject a real `DynamicCache` object at runtime (bypassing FX's inability to trace `list.append`). |

#### Key methods

| Method | What it does |
|---|---|
| `trace(model)` | Main entry. Builds `_Wrapper`, runs `_HelmTracer.trace()`, returns `GraphModule`. |
| `_detect_kv_kwarg()` | Checks whether the model uses `past_key_value` or `past_key_values` (architecture difference). |
| `_detect_pos_embed()` | Detects if Qwen-style `position_embeddings` is a separate argument. |

#### How KV injection works at runtime

During tracing, `_HelmTracer` records `call_module(decoder_layer_N, ...)` as a leaf. The `_Wrapper`'s pre-hook fires at runtime, injects the real `DynamicCache` into the call, and the layer's actual `forward()` (not the traced version) calls `DynamicCache.update()` correctly.

---

### 4.3 Graph IR: `graph.py`

**Path:** `helm/compiler/IR/graph.py`
**Role:** Typed semantic intermediate representation. Every FX node becomes a `HelmNode`; every tensor flow becomes a `HelmEdge`.

#### HelmEdge

```python
class HelmEdge:
    src_id: int
    dst_id: int
    tensor_shape: Optional[List[int]]
    tensor_bytes: int
    is_residual: bool              # skip connection
    crosses_layer_boundary: bool   # used to find stage split points
```

#### HelmNode — key fields

| Field | Type | Purpose |
|---|---|---|
| `id` | int | Unique integer identifier |
| `name` | str | Human-readable name (e.g. `"N42"`) |
| `fx_node` | `fx.Node` | Back-reference to original FX node |
| `op_type` | str | `call_module`, `call_function`, `placeholder`, `output` |
| `layer_id` | Optional[int] | Which transformer layer (0..L-1) this node belongs to |
| `is_attention`, `is_mlp`, `is_norm`, `is_embedding`, `is_output_head` | bool | Semantic classification |
| `flops_prefill`, `flops_decode` | int | FLOP counts for two regimes |
| `activation_bytes` | int | Output tensor size in bytes |
| `param_bytes` | int | Weight bytes in this node's module |
| `kv_bytes_per_token` | int | KV cache growth per generated token |

#### HelmGraph

```python
class HelmGraph:
    nodes: List[HelmNode]
    edges: List[HelmEdge]
    fx_to_helm: Dict[fx.Node, HelmNode]        # lookup by FX node
    helm_id_to_node: Dict[int, HelmNode]        # lookup by integer node id
    layer_to_node_ids: Dict[int, List[int]]     # all node ids in layer N
    input_node_ids: List[int]
    output_node_ids: List[int]
```

Key method: `_build_from_fx(gm)` — iterates the FX graph, creates one `HelmNode` per node, builds `HelmEdge` for every use.

---

### 4.4 FX Importer: `fx_importer.py`

**Path:** `helm/compiler/importers/fx_importer.py`
**Role:** Takes a raw `HelmGraph` (just structure, no semantics) and annotates nodes with layer IDs and semantic tags.

#### Key methods

| Method | What it does |
|---|---|
| `_assign_module_paths()` | For each `call_module` node, records the full dotted module path (e.g. `model.layers.5.self_attn.q_proj`). |
| `_extract_layer_id_from_path(path)` | Regex-searches for patterns like `layers.5`, `h.3`, `blocks.7` in the module path. Returns the integer layer index. |
| `_assign_semantic_tags()` | String-matches module paths against keywords: `attn`/`attention` → `is_attention=True`; `mlp`/`ffn` → `is_mlp=True`; `norm`/`ln` → `is_norm=True`; etc. |
| `_populate_graph_indexes()` | Builds `layer_to_node_ids` and `block_to_node_ids` for O(1) lookups. |

**Supported architectures:** Any model using `layers.N`, `h.N`, or `blocks.N` naming (covers LLaMA, Qwen, GPT, Mistral, Falcon, etc.).

---

### 4.5 Cost Analysis: `hybrid_analyzer.py`

**Path:** `helm/compiler/analysis/hybrid_analyzer.py`
**Role:** Populates every `HelmNode` with actual shape information and estimated compute/memory costs.

#### HybridAnalysisSummary

Returned from analysis. Contains: `num_nodes`, `num_nodes_with_shapes`, `total_activation_bytes`, `total_param_bytes`, `total_flops_prefill`, `total_flops_decode`, `total_kv_bytes_per_token`.

#### HybridAnalyzer — analysis phases

| Phase | Method | What happens |
|---|---|---|
| 1. Shape Propagation | `_propagate_shapes(gm, example_inputs)` | Runs the FX graph once with real inputs; records each node's actual output `tensor.shape`. |
| 2. Activation Annotation | `_annotate_node_shapes_and_activations()` | For each node with known shapes: compute `activation_bytes = product(shape) * dtype_size`. |
| 3. Module Cost Annotation | `_annotate_module_costs()` | For `call_module` nodes: sum all `param.numel() * dtype_size` for weight bytes; estimate attention KV growth; estimate FLOPs. |
| 4. Block Aggregation | (internal) | Sum node costs within each transformer block for partition-level cost model. |

#### How FLOP estimation works

- **Prefill (GEMM regime, seq_len=S):** `2 * S * in_features * out_features` per linear layer
- **Decode (GEMV regime, seq_len=1):** `2 * in_features * out_features` per linear layer
- **Attention prefill:** `4 * S^2 * num_heads * head_dim`
- **KV per token:** `2 * num_kv_heads * head_dim * dtype_size * num_layers`

Fallback: if shape propagation fails, uses static heuristics based on module type.

---

### 4.6 Partition Units: `partition_units.py`

**Path:** `helm/compiler/partition/partition_units.py`
**Role:** Groups the O(nodes) FX nodes into just O(L+2) coarse units for O(L) plan search.

#### PartitionUnit

```python
@dataclass
class PartitionUnit:
    unit_id: int
    unit_type: str          # "embedding", "transformer_block", "output"
    layer_start: Optional[int]
    layer_end: Optional[int]
    node_ids: List[int]
    # Aggregated costs:
    flops_prefill: float
    flops_decode: float
    param_bytes: int
    activation_bytes: int
    kv_bytes_per_token: int
    contains_attention: bool
    contains_mlp: bool
    contains_norm: bool
    contains_kv_projection: bool
```

#### PartitionUnitBuilder

| Method | What it creates |
|---|---|
| `_build_embedding_unit()` | One unit for all `is_embedding=True` nodes |
| `_build_layer_units()` | One unit per transformer block (all nodes with same `layer_id`) |
| `_build_output_unit()` | One unit for all `is_output_head=True` nodes (lm_head) |
| `_aggregate_cost(node_ids)` | Sums all node-level costs into the unit |

**Why coarsen?** The strategy selector must enumerate all split points. With O(nodes) granularity that's expensive; with O(L) units it's trivial (< 1 ms for 32-layer model).

---

### 4.7 Device Profiler: `device_profiler.py`

**Path:** `helm/compiler/optimization/device_profiler.py`
**Role:** Measures actual hardware capabilities — not estimates, actual numbers.

#### Benchmarks run

| Benchmark | What's measured | How |
|---|---|---|
| CPU stream bandwidth | DRAM read+write (GB/s) | 64 MB buffer copy, timed over N iterations |
| CPU prefill FLOPs | Effective TFLOPS in GEMM regime | `(128, H) × (H, H)` matmul |
| CPU decode FLOPs | Effective TFLOPS in GEMV regime | `(1, H) × (H, H)` matmul |
| GPU stream bandwidth | DRAM bandwidth (GB/s) | Same buffer copy on GPU |
| GPU prefill FLOPs | GPU GEMM TFLOPS | Same matmul on GPU |
| GPU decode FLOPs | GPU GEMV TFLOPS | Same matmul on GPU |
| PCIe H2D | CPU→GPU bytes/s | `torch.pinned_memory` → GPU copy |
| PCIe D2H | GPU→CPU bytes/s | GPU → `torch.pinned_memory` copy |

#### Timing helpers

- `_time_cpu(fn)`: Adaptive iteration count (runs more iterations for fast functions to get stable timing).
- `_time_gpu(fn)`: Uses `torch.cuda.synchronize()` before/after for accurate GPU timing.

**Caching:** Results are stored in a module-level dict keyed by `(hidden_size, dtype)`. A second `compile()` call on the same machine reuses cached measurements.

---

### 4.8 Cost Model: `cost_model.py`

**Path:** `helm/compiler/optimization/cost_model.py`
**Role:** Given a `PartitionPlan` and actual device profiles, estimates exact latency for each stage.

#### Key dataclasses

| Class | Key fields |
|---|---|
| `ModelConfig` | `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_kv_heads`, `head_dim`, `dtype_size`. Created via `from_hf_config(model.config)`. |
| `DeviceProfile` | `peak_flops_prefill`, `peak_flops_decode`, `mem_bandwidth`, `memory_capacity`, `efficiency_compute`, `efficiency_memory`, `l3_size_bytes`, `l3_bandwidth` |
| `LinkProfile` | `src`, `dst`, `bandwidth_bytes_per_s`, `latency_s` — models the PCIe bus |
| `WorkloadSpec` | `batch_size`, `prefill_seq_len`, `decode_context_len`, `decode_tokens`, `dtype_size` |
| `StageCost` | Per-stage breakdown: `param_bytes`, `activation_bytes`, `kv_bytes`, `prefill_latency_s`, `decode_token_latency_s`, `feasible`, `decode_regime` |
| `PlanCost` | Aggregated: `feasible`, `prefill_latency_s`, `decode_token_latency_s`, `throughput_tokens_per_s`, `max_stage_memory_bytes` |

#### HelmCostModel.estimate_plan()

```
For each stage:
  1. Compute projection cost:
     proj_compute_s = proj_flops / (peak_flops * eff_compute)
     proj_memory_s  = param_bytes / (mem_bw * eff_memory)
     stage_compute_s = max(proj_compute_s, proj_memory_s)   # roofline

  2. Compute attention cost (separate because KV memory access differs):
     attn_memory_s = kv_bytes / (effective_kv_bandwidth)
     stage_attn_s  = max(attn_compute_s, attn_memory_s)

  3. Communication cost (activation transfer across PCIe):
     comm_s = activation_bytes / link.bandwidth_bytes_per_s

  4. Stage total = stage_compute_s + stage_attn_s + comm_s

  5. Feasibility = stage memory_bytes <= device.memory_capacity

Plan total = sum of all stage totals
```

**Roofline insight:** Memory bandwidth is usually the bottleneck for decode (GEMV is memory-bound). Compute is bottleneck for prefill (GEMM). The cost model handles both regimes correctly with separate FLOP measurements.

---

### 4.9 Strategy Selector: `strategy_selector.py`

**Path:** `helm/compiler/optimization/strategy_selector.py`
**Role:** Finds the best partition plan by exhaustive search over all feasible splits.

#### StrategySelectorConfig

```python
@dataclass
class StrategySelectorConfig:
    objective: str           # "decode_latency" | "prefill_latency" | "total_latency" | "throughput"
    max_stages: int          # usually 2 (CPU + GPU)
    allow_cpu: bool
    allow_multi_gpu: bool
    kv_offload: bool
    kv_reserve_tokens: int   # tokens to keep hot in GPU KV cache
```

#### StrategySelector.select() algorithm

```
Plans evaluated (in order):

1. All-GPU (1 stage): all units on GPU
   → Score with cost_model.estimate_plan()

2. For k = 1 to N-1 (N = number of units):
   CPU stage: units[0:k]
   GPU stage: units[k:N]
   → Score both stages; plan score = max(stage scores) [critical path]

3. All-CPU (1 stage): all units on CPU
   → Score

4. Return the plan with minimum objective score
```

**Why not greedy?** Greedy would keep adding to GPU until it's full. But the roofline model may show that a GPU stage of 12 layers is memory-bandwidth-bound while a split at 8+4 allows both stages to stay in their fast regions.

**Time complexity:** O(L) plan evaluations × O(1) cost estimate = O(L) total. Runs in < 1 ms for Qwen3-32B.

---

### 4.10 Partition Plan: `partition_plan.py`

**Path:** `helm/compiler/partition/partition_plan.py`

#### Key dataclasses

| Class | Purpose |
|---|---|
| `StageSpec` | One stage: `stage_id`, `device_id` (`"cpu"` or `"cuda:0"`), `units[]`, `layer_start`, `layer_end` |
| `PartitionPlan` | Full plan: `stages[]` of `StageSpec` objects |
| `ExecutionSchedule` | Execution config: `mode`, `num_stages`, `microbatches` |

---

### 4.11 Stage Builder: `stage_fx_builder.py`

**Path:** `helm/compiler/lowering/stage_fx_builder.py`
**Role:** Takes the full FX `GraphModule` + `PartitionPlan`, fragments it into per-device standalone `GraphModule`s.

#### StageFXBuilder — key methods

| Method | What it does |
|---|---|
| `_compute_stages()` | Maps each `HelmNode` → `stage_id` based on which `PartitionUnit` it belongs to. |
| `_build_stage(stage_id)` | Extracts the subgraph for one stage: collects all nodes assigned to that stage, ensures all dependencies are satisfied. |
| `build()` | Returns `List[Stage]` with per-device `GraphModule`s. |

#### Node assignment algorithm

```
1. Direct assignment: node_id → stage via unit_to_stage lookup
2. Backward pass: unassigned nodes inherit stage from their consumers
   (e.g., weight `get_attr` nodes follow their user)
3. Forward pass: unassigned nodes inherit stage from their producers
   (e.g., intermediate ops follow their input)
4. Final fallback: assign to stage 0
```

**Output:** Each `Stage` object contains `stage_id`, `device` (`"cpu"` or `"cuda:0"`), and a `module: GraphModule` that is a complete standalone executable subgraph.

---

## 5. Runtime Execution

### 5.1 Pipeline Runtime: `pipeline_runtime.py`

**Path:** `helm/runtime/pipeline_runtime.py`
**Role:** High-level generation loop. Runs prefill then autoregressive decode.

#### Key insight about same-graph reuse

The decode tracer captures a single decode step (seq_len=1). At prefill time, HELM runs the **same** compiled stages but with a full-length input and a different attention mask. This avoids maintaining separate prefill-compiled and decode-compiled models.

#### PipelineRuntime — key methods

| Method | What it does |
|---|---|
| `_build_causal_mask(seq_len)` | Returns `(S, S)` lower-triangular mask for prefill (all positions attend to prior positions). |
| `_build_decode_mask(total_len)` | Returns `(1, total_len)` all-ones mask for decode (new token attends to all prior). |
| `prefill(input_ids)` | Runs `StageRuntimeExecutor` with full prompt; populates `DynamicCache`; returns logits. |
| `decode_step(input_ids, step_position)` | Runs executor with single token; returns next-token logits. |
| `_reset_decode_cache()` | Clears `DynamicCache` before generation. Fast path: direct wrapper reference. Slow path: traverses `forward_pre_hooks`. |
| `generate(input_ids, max_new_tokens)` | Full loop: prefill → sample first token → decode_step × max_new_tokens. Returns all generated token IDs. |

---

### 5.2 Stage Executor: `executor.py`

**Path:** `helm/runtime/executor.py`
**Role:** Manages stage initialization (weight placement, prewarming) and per-token stage execution.

#### The weight placement problem

After loading a model with `device_map='auto'`, PyTorch might place early layers on GPU. HELM may want a different assignment (e.g., layers 0-7 on CPU, 8-31 on GPU). Naively moving everything first would spike peak memory.

#### StageRuntimeExecutor — initialization

| Method | What it does |
|---|---|
| `_break_cross_stage_ties()` | Detects tied weights (same `data_ptr()` across stage boundaries, e.g., `embed_tokens.weight == lm_head.weight` in Qwen3). Clones them so each stage owns independent tensors. This is required before targeted device moves. |
| `_prewarm_stage_devices()` | **Interleaved swap strategy:** Move CPU-stage weights to CPU first (freeing any GPU VRAM they occupy), then move GPU-stage weights CPU→CUDA. Peak memory = max(original, one submodule), not full model. |
| `_patch_cpu_linears()` | Applies AVX2-optimized linear ops from `helm/kernels/` to all CPU-stage linear layers. |

#### StageRuntimeExecutor.run(inputs)

```
1. Execute Stage 0 (CPU):
   stage_0.module(inputs) → cpu_activations

2. Transfer activations:
   move_tensor(cpu_activations, "cuda:0")

3. Execute Stage 1 (GPU):
   stage_1.module(gpu_activations) → gpu_logits

4. Return logits (on GPU)
```

Cache injection happens transparently via the `forward_pre_hooks` registered during tracing.

---

### 5.3 KV Allocator: `kv_allocator.py`

**Path:** `helm/runtime/kv_allocator.py`
**Role:** Page pool manager. Pre-allocates page tensors and recycles them.

#### KVAllocator — key methods

| Method | What it does |
|---|---|
| `_create_page(device)` | Allocates one page tensor: shape `(1, num_kv_heads, page_size, head_dim)`. |
| `reserve_batch(batch_size, device)` | Pre-allocates N pages at once to avoid repeated allocation overhead. |
| `allocate(device)` | Pops a page from the free pool. If pool is empty, calls `reserve_batch` first. |
| `free(page)` | Returns a page to the free pool for reuse. Resets `used_tokens = 0`. |

**Pool strategy:**
- GPU: 8-page reserve batches (VRAM is scarce)
- CPU: 16-page reserve batches (RAM is abundant)

---

### 5.4 KV Cache: `kv_cache.py`

**Path:** `helm/runtime/kv_cache.py`
**Role:** Manages paged KV storage with GPU/CPU residency tracking and streaming attention.

#### KVPage

```python
@dataclass
class KVPage:
    page_id: str
    layer_id: int
    start_token: int      # position of first token in this page
    used_tokens: int      # how many tokens are stored
    capacity_tokens: int  # page_size
    k_tensor: Tensor      # (1, num_kv_heads, capacity, head_dim)
    v_tensor: Tensor
    device: str           # "cuda:0" or "cpu"
    state: str            # "FREE", "GPU", "CPU"
```

#### LayerKVCache

One per transformer layer. Holds a list of `KVPage`s and the tail page (the currently-being-filled page).

#### KVCacheManager — key methods

| Method | What it does |
|---|---|
| `initialize_layer_caches(num_layers)` | Creates empty `LayerKVCache` for each layer. |
| `append_prefill(layer_id, K, V)` | Stores K, V for all prefill tokens. Splits into page-sized chunks. |
| `append_decode(layer_id, K_new, V_new)` | Appends one token's K, V to the tail page. If tail is full, allocates new page. |
| `_enforce_residency_policy()` | If total GPU bytes > `gpu_high_watermark_bytes`, evict oldest pages to CPU. FIFO by `start_token`, protecting tail page. |
| `perform_streaming_attention(layer_id, Q, scale, mask)` | Decode-time attention without materializing full context on GPU. Fetches pages page-by-page via `_kv_copy_stream`, computes online softmax. |

#### Online softmax streaming attention

```
running_num = 0        # numerator accumulator
running_denom = 0      # denominator accumulator
running_max = -inf     # for numerical stability

For each page (oldest first):
    H2D copy: page.k_tensor, page.v_tensor → GPU (async via _kv_copy_stream)
    Compute: scores = Q @ K.T / scale
    Update running max, rescale running_num/denom, accumulate
    Free GPU copy

output = running_num / running_denom
```

This is the [Flash-Attention style online softmax](https://arxiv.org/abs/2205.14135) but applied page-by-page instead of block-by-block. Memory usage = O(page_size) instead of O(context_length).

---

### 5.5 KV Offload: `kv_offload.py`

**Path:** `helm/runtime/kv_offload.py`
**Role:** Patches transformer attention modules at the class level to use `KVCacheManager` instead of `DynamicCache`.

#### KVOffloadConfig

```python
@dataclass
class KVOffloadConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int = 64
    dtype: torch.dtype = torch.float16
    gpu_watermark_bytes: Optional[int] = None  # default: ~512 tokens per layer
    cont_capacity: int = 0  # optional all-GPU contiguous fast-path budget

    @staticmethod
    def from_model(model, ...) -> 'KVOffloadConfig':  # extracts from model.config
```

`cont_capacity` enables an optional fast path: when set above `0` and the full sequence fits within this token budget, HELM uses in-place writes + a single SDPA call instead of paged attention. The default is `0` so paged KV and the GPU residency watermark are active during offload runs.

#### Architecture-specific forward factories

| Function | Target architecture |
|---|---|
| `_make_qwen2_forward(kvcms)` | Qwen2 attention modules |
| `_make_qwen3_forward(kvcms)` | Qwen3 attention modules |
| `_make_llama_forward(kvcms)` | LLaMA attention modules |
| `_make_gemma2_forward(kvcms)` | Gemma-2 attention modules |

Each factory returns a patched `forward(hidden_states, ..., **kwargs)` that:
1. Computes Q, K, V via the layer's existing projections
2. Calls `manager.append_prefill()` or `manager.append_decode()` instead of `DynamicCache.update()`
3. For decode (`q_len == 1`): calls `manager.perform_streaming_attention()` instead of standard attention

**Why class-level patching?** Applies to all instances, including those inside compiled `GraphModule`s, without needing to modify the FX graph.

---

### 5.6 Tensor Transfer: `tensor_transfer.py`

**Path:** `helm/runtime/tensor_transfer.py`

#### move_tensor(tensor, device)

Recursively moves tensors in nested structures (tuples, lists, dicts) to the target device. Skips tensors already on the target device (normalized string comparison). Placeholder for future async PCIe/NVLink overlap.

---

### 5.7 Stage: `stage.py`

**Path:** `helm/runtime/stage.py`

```python
@dataclass
class Stage:
    stage_id: int
    device: str           # "cpu" or "cuda:0"
    module: GraphModule   # standalone FX subgraph
```

Minimal dataclass. Created by `StageFXBuilder.build()`, consumed by `StageRuntimeExecutor`.

---

## 6. CLI: `cli.py`

**Path:** `helm/cli.py`

#### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--model` | (required) | HuggingFace model name or path |
| `--prompt` | `'Explain what a compiler does.'` | Input text |
| `--max-input-tokens` | `64` | Max prompt tokens |
| `--max-new-tokens` | `8` | Tokens to generate |
| `--dtype` | `float16` | `float16`, `bfloat16`, or `float32` |
| `--compiler-plan` | `auto` | `auto` (profile + search) or `manual` |
| `--compiler-cpu-layers` | `None` | e.g. `"0:7"` for manual plan |
| `--compiler-gpu-layers` | `None` | e.g. `"8:31"` for manual plan |
| `--print-plan` | `False` | Dump partition plan JSON to stdout |
| `--kv-offload` | `False` | Enable paged KV cache offloading |
| `--cpu-threads` | `6` | CPU thread count for PyTorch ops |
| `--backend` | `auto` | `auto` (route all-GPU plans to vLLM when installed), `router` (require vLLM for all-GPU plans), or `helm` (always native) |
| `--mode` | `plan` | `generate`, `baseline`, `import`, `units`, `plan`, `lower`, `execute_stagewise`, `dry_run` |

#### `--mode` values explained

| Mode | What runs |
|---|---|
| `generate` | Public inference API path: builds a `HelmInferenceConfig` (honoring `--backend`) and runs `HelmInference.generate()` |
| `baseline` | Pure HuggingFace forward, no HELM |
| `import` | Trace + build IR only |
| `units` | Trace + IR + cost analysis + partition units |
| `plan` | Everything above + device profile + strategy selection |
| `lower` | Everything above + stage FX building |
| `execute_stagewise` | Full compile + one forward pass |
| `dry_run` | Validates arguments and writes metrics.json, returns before the model loads (cheapest smoke mode) |

Every mode except `generate` is dispatched to `helm/_pipeline.py`'s diagnostic/benchmark pipeline. `--backend` only applies to `--mode generate`; it selects whether `HelmInference` routes all-GPU plans to vLLM automatically when it is installed (`auto`, the default), requires vLLM for them (`router`), or always uses HELM's native CPU+GPU executor (`helm`). Plans with a CPU stage always run on HELM.

---

## 7. Experiments & Benchmarks

### `helm/_pipeline.py`

**Path:** `helm/_pipeline.py`

The implementation behind quick iterative development testing: loads a model, runs a baseline forward, then compiles with HELM and compares outputs and timing. It lives under `helm/` and ships in the installed wheel — it does not require the unpackaged `experiments/` tree. `experiments/dev_pipeline.py` is a thin compatibility shim that re-exports this module (`sys.modules[__name__] = _pipeline`) so existing research scripts that `import experiments.dev_pipeline` keep working.

### `experiments/paper_bench.py`

Full benchmark harness. Compares HELM against vLLM, HuggingFace Accelerate, and DeepSpeed.

#### Sections run

| Section | What it measures |
|---|---|
| **E — Feasibility** | Does the model load? Peak GPU/CPU memory at load time. |
| **F — Max decode length** | Longest output before OOM. |
| **A — Latency sweep** | TTFT (time-to-first-token) and decode tok/s; p50/p95/p99 across 10 requests. |
| **B — Throughput** | Aggregate tok/s at batch sizes 1, 2, 4, 8. |
| **C — Ablations** | Vary: batch_size, context_length, AVX on/off, KV offload on/off, CPU threads. |
| **D — LM quality** | MMLU, HellaSwag, ARC-Easy via `lm-evaluation-harness`. |

**Sentinel system:** Each section writes a `.done` file on completion. Re-running after a crash automatically skips completed sections.

### `experiments/profile_pipeline.py`

Per-stage profiling. Measures per-layer latency, memory, and activation sizes to validate cost model predictions.

### Figure / table drivers

| Script | Paper element |
|---|---|
| `experiments/run_paper_experiments.sh` | Table II, Fig. 5, ablations (wraps `paper_bench.py`) |
| `experiments/ttft_prompt_scaling.sh` + `_summarize.py` | Fig. 4 — TTFT vs. prompt length |
| `experiments/longctx_paging.sh` + `_summarize.py` | Fig. 6 — long-context KV paging |
| `experiments/llama_cpp_bench.py`, `experiments/run_llama13b_helm.sh` | Table III — llama.cpp comparison |
| `experiments/compile_breakdown.py` | §VI-D — compile-time breakdown |
| `experiments/plan_curve.py` | Cost-model partition curve (decode vs. TTFT per split) |
| `experiments/verify_e2e.py`, `experiments/verify_cell.sh` | Correctness check; single Table II cell verification |

`reproduce.sh` at the repository root wraps these per table/figure.

---

## 8. Tests

### `tests/conftest.py` — shared fixtures

| Fixture | What it provides |
|---|---|
| `simple_mlp` | Minimal MLP nn.Module for testing FX tracing |
| `tiny_transformer` | Small transformer (TinyLayer submodules) for partition testing |
| `tiny_transformer_tied` | Same but with tied `embed_tokens.weight == lm_head.weight` |
| `simple_gm` | FX-traced `simple_mlp` (all ops expanded) |
| `tiny_gm_with_leaves` | FX-traced `tiny_transformer` with `TinyLayer` as leaves |
| `cpu_profile` | `DeviceProfile`: 1 TFLOPS, 50 GB/s bandwidth, 32 GB capacity |
| `gpu_profile` | `DeviceProfile`: 100 TFLOPS, 500 GB/s bandwidth, 8 GB capacity |
| `pcie_link` | `LinkProfile`: 16 GB/s, 100 µs latency |
| `cost_model` | `HelmCostModel` built from above profiles |
| `workload` | `WorkloadSpec`: batch=1, prefill=128, decode_context=256, decode_tokens=64 |
| `kv_allocator` | `KVAllocator`: 4 layers, 2 heads, 16 dim, page_size=8 |

### Unit tests (key ones)

| Test file | What it tests |
|---|---|
| `test_ir_graph.py` | HelmNode/HelmEdge construction, semantic fields, defaults |
| `test_fx_importer.py` | Layer ID extraction, semantic tagging from module paths |
| `test_partition_units.py` | Unit building, cost aggregation |
| `test_cost_model.py` | Roofline estimate, feasibility check |
| `test_strategy_selector.py` | Plan enumeration, objective scoring |
| `test_kv_cache.py` | Page append, prefill/decode, eviction |
| `test_kv_allocator.py` | Pool allocation/freeing |
| `test_batch_kv_offload.py` | Multi-batch KV managers with shared allocator |
| `test_streaming_attention.py` | Online softmax correctness |
| `test_tensor_transfer.py` | Recursive device movement |

### Integration tests

| Test file | What it validates |
|---|---|
| `test_compiler_pipeline.py` | End-to-end compile: model → trace → analyze → partition → lower → stage graphs |
| `test_runtime_pipeline.py` | End-to-end execute: prefill + decode loop with real stage execution |

---

## 9. Key Algorithms Explained

### Roofline Cost Model

```
latency = max(compute_time, memory_time)

compute_time = FLOPs / (peak_FLOPS × efficiency)
memory_time  = bytes_accessed / (DRAM_bandwidth × efficiency)

If compute_time > memory_time: "compute-bound"
If memory_time > compute_time: "memory-bound" ← decode is always here
```

For decode: GEMV operations read the full weight matrix but do tiny compute. Every modern GPU is memory-bound at batch=1 decode. The cost model correctly captures this.

### Exhaustive O(L) Split Search

For a model with L transformer blocks:
- Unit count = L + 2 (embedding + L blocks + output)
- Split positions = L + 1 possible CPU/GPU boundaries
- Each evaluation = O(1) cost model call
- Total = O(L) → < 1 ms for L=64

### Paged KV Cache

Problem: KV cache for 32K tokens × 32 layers × 8 KV heads × 128 head_dim = 8 GB. This OOMs on a 24 GB card that already holds 14B weights.

Solution:
1. Divide KV storage into `page_size=64` token pages (default)
2. Keep only the most recent N pages on GPU (controlled by `gpu_watermark_bytes`)
3. Evict oldest pages to CPU pinned memory (no copy overhead since pinned)
4. At decode attention: fetch pages one-by-one from CPU → GPU async, compute online softmax

This extends effective context by ~8× on RTX 3090 with Qwen3-4B.

### Interleaved Device Swap (Weight Prewarming)

Problem: Naively moving 14B weights requires peak memory = full model size × 2.

Solution:
```
While stages not fully placed:
    Move GPU-stage chunk: CPU → GPU    (frees CPU RAM)
    Move CPU-stage chunk: GPU → CPU    (frees GPU VRAM)
```
Peak memory = max(initial, one chunk) ≈ 1/N of full model. Critical on 16 GB machines.

---

## 10. Data Flow Diagram

```
User calls: helm compile --model Qwen/Qwen3-14B --kv-offload
                               │
                    ┌──────────▼──────────┐
                    │     cli.py          │
                    │  parse args →       │
                    │  HelmCompileOptions │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │    compiler.py: compile_graph()  │
                    └──┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  DecodeTracer   │  → FX GraphModule (seq_len=1 decode step)
              └────────┬────────┘    DecoderLayer = leaf nodes
                       │
              ┌────────▼────────┐
              │   FXImporter    │  → HelmGraph (layer IDs, semantic tags)
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ HybridAnalyzer  │  → per-node: shapes, FLOPs, param bytes, KV bytes
              └────────┬────────┘
                       │
              ┌────────▼─────────────┐
              │ PartitionUnitBuilder │  → O(L+2) units with aggregated costs
              └────────┬─────────────┘
                       │
              ┌────────▼────────┐
              │  DeviceProfiler  │  → actual CPU/GPU FLOPS, bandwidth, PCIe speed
              └────────┬────────┘
                       │
              ┌────────▼─────────────┐
              │  HelmCostModel +     │
              │  StrategySelector    │  → best PartitionPlan (e.g. layers 0:7=CPU, 8:31=GPU)
              └────────┬─────────────┘
                       │
              ┌────────▼────────┐
              │ StageFXBuilder  │  → List[Stage]: Stage0(cpu, GraphModule), Stage1(gpu, GraphModule)
              └────────┬────────┘
                       │
                    CompiledHelmArtifact
                               │
                    ┌──────────▼──────────────────────┐
                    │   StageRuntimeExecutor           │
                    │  - break tied weights            │
                    │  - interleaved device prewarm    │
                    │  - patch CPU linears (AVX2)      │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │     PipelineRuntime              │
                    │  prefill(prompt) →               │
                    │  generate(max_new_tokens=512) →  │
                    │  return token ids                │
                    └──────────────────────────────────┘

  If --kv-offload:
  ┌──────────────────────────────────────────────────────┐
  │  KVOffloadConfig.from_model()                        │
  │  _make_qwen3_forward() patches attention class       │
  │  KVAllocator manages GPU/CPU page pools              │
  │  KVCacheManager: append, evict, streaming attention  │
  └──────────────────────────────────────────────────────┘
```

---

## 11. Known Limitations

| Area | Limitation | Location |
|---|---|---|
| KV offload | CPU KV pages are allocated pageable rather than pinned, limiting how much `non_blocking=True` H2D copies can overlap with compute | `helm/runtime/kv_allocator.py` |
| Partitioning | The planner searches contiguous CPU→GPU splits of a single GPU; multi-GPU plans are out of scope | `helm/compiler/optimization/strategy_selector.py` |
| Validation | Focused unit tests cover attention-mask handling, CUDA stream lifetimes and paging invariants; large-model and multi-GPU end-to-end runs are not part of CI | `tests/` |
