"""
HELM compiler pipeline.

This module provides a canonical compile path that can be reused by
benchmarks, tests, and future API/runtime integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import logging
import os

import psutil

logger = logging.getLogger(__name__)
import torch
import torch.fx

from .IR.graph import HelmGraph
from .analysis.hybrid_analyzer import HybridAnalyzer
from .importers.fx_importer import FXImporter
from .lowering.stage_fx_builder import StageFXBuilder
from .optimization.cost_model import DeviceProfile, HelmCostModel, LinkProfile, ModelConfig, WorkloadSpec
from .optimization.device_profiler import profile_devices
from .optimization.strategy_selector import StrategySelector, StrategySelectorConfig
from .partition.partition_plan import PartitionPlan, StageSpec
from .partition.partition_units import PartitionUnit, PartitionUnitBuilder
from .scheduling.scheduler import Scheduler


@dataclass
class HelmCompileOptions:
    mode: str = "both"  # prefill / decode / both
    objective: str = "decode_latency"
    plan_mode: str = "auto"  # auto / manual
    allow_cpu: bool = True
    allow_gpu: bool = True
    allow_multi_gpu: bool = False
    max_stages: int = 2
    cpu_layers: Optional[str] = None
    gpu_layers: Optional[str] = None
    lower_stages: bool = False
    graph_kind: str = "prefill"
    model_name: str = ""
    workload: Dict[str, int] = field(default_factory=dict)
    kv_offload: bool = False
    kv_reserve_tokens: int = 0  # reserve GPU KV headroom for N tokens when kv_offload=True
    gpu_memory_reserve_mb: Optional[int] = None  # per-compile override; env var still supported
    allow_static_analysis_fallback: bool = False
    debug: bool = False


@dataclass
class HelmIR:
    schema_version: str
    graph_kind: str
    model_name: str
    nodes: List[Dict[str, Any]]
    units: List[Dict[str, Any]]
    plan: Dict[str, Any]
    schedule: Dict[str, Any]
    hardware: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_kind": self.graph_kind,
            "model_name": self.model_name,
            "nodes": self.nodes,
            "units": self.units,
            "plan": self.plan,
            "schedule": self.schedule,
            "hardware": self.hardware,
            "metadata": self.metadata,
        }


@dataclass
class CompiledHelmArtifact:
    gm: torch.fx.GraphModule
    helm_graph: HelmGraph
    analysis_summary: Dict[str, Any]
    partition_units: List[PartitionUnit]
    partition_plan: PartitionPlan
    plan_data: Dict[str, Any]
    plan_cost: Any  # PlanCost from cost model, or None for manual/override plans
    schedule: Dict[str, Any]
    stage_graphs: List[Any]
    helmir: HelmIR
    metadata: Dict[str, Any]


def parse_layer_spec(spec: Optional[str]) -> set[int]:
    """Parse a layer spec into a set of layer indices.

    Accepts a single index ("43"), an inclusive range ("0:42"), or a
    comma-separated list of either ("0:5,20:25" or "0:1,3,5:6"). The multi-range
    form expresses non-contiguous (interleaved) placements, used to study how
    expanding the partition design space beyond one contiguous split affects
    performance.
    """
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        text = part.strip()
        if not text:
            continue
        if ":" in text:
            start_s, end_s = text.split(":", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range '{part}' in spec '{spec}'")
            out.update(range(start, end + 1))
        else:
            out.add(int(text))
    return out


def _build_helm_graph(gm: torch.fx.GraphModule) -> HelmGraph:
    graph = HelmGraph(gm.graph)
    importer = FXImporter(gm, graph)
    importer.run()
    return graph


def _run_analysis(
    gm: torch.fx.GraphModule,
    helm_graph: HelmGraph,
    model: torch.nn.Module,
    example_inputs: Any,
) -> Tuple[HybridAnalyzer, Dict[str, Any]]:
    analyzer = HybridAnalyzer(gm=gm, helm_graph=helm_graph, model=model)
    summary = analyzer.run(example_inputs)
    summary_data = {
        "num_nodes": summary.num_nodes,
        "num_nodes_with_shapes": summary.num_nodes_with_shapes,
        "total_activation_bytes": summary.total_activation_bytes,
        "total_param_bytes": summary.total_param_bytes,
        "total_flops_prefill": summary.total_flops_prefill,
        "total_flops_decode": summary.total_flops_decode,
        "total_kv_bytes_per_token": summary.total_kv_bytes_per_token,
    }
    return analyzer, summary_data


def _run_static_fallback_analysis(
    gm: torch.fx.GraphModule,
    helm_graph: HelmGraph,
    model: torch.nn.Module,
    workload: Dict[str, int],
) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
    inter_size = int(getattr(cfg, "intermediate_size", 0) or 0)
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads) or 0)

    bsz = int(workload.get("batch_size", 1) or 1)
    seq_len = int(workload.get("prefill_seq_len", 64) or 64)
    dtype_size = int(workload.get("dtype_size", 2) or 2)

    kv_head_dim = hidden_size // num_heads if num_heads > 0 else 0
    kv_bytes_per_token = int(2 * max(num_kv_heads, 1) * kv_head_dim * dtype_size) if hidden_size > 0 else 0

    for node in helm_graph.nodes:
        node.input_shapes = []
        node.output_shapes = []
        node.activation_bytes = 0
        node.bytes_read = 0
        node.bytes_written = 0
        node.kv_bytes_per_token = 0
        node.flops_prefill = 0
        node.flops_decode = 0
        node.param_bytes = 0

        if node.op_type == "call_module":
            submodule = gm.get_submodule(node.target)
            node.param_bytes = sum(p.numel() * p.element_size() for p in submodule.parameters())

            target = str(node.target)
            if (".layers." in target or ".h." in target or ".blocks." in target) and hidden_size > 0 and inter_size > 0:
                # Prefill FLOPs — must match hybrid_analyzer._estimate_block_costs exactly.
                # attn_linear: 4 projections (Q,K,V,O), factor 2 FLOPs/MAC → 4*B*S*H*H
                # attn_scores: QK^T + AV, each B*S*S*H MACs → 4*B*S*S*H FLOPs
                # mlp: 3 projections (gate,up,down), factor 2 FLOPs/MAC → 6*B*S*H*I
                attn_linear = 4 * bsz * seq_len * hidden_size * hidden_size
                attn_scores = 4 * bsz * seq_len * seq_len * hidden_size
                mlp = 6 * bsz * seq_len * hidden_size * inter_size
                node.flops_prefill = int(attn_linear + attn_scores + mlp)

                # Decode FLOPs — seq=1 token, same formulas with S=1.
                attn_linear_d = 4 * bsz * hidden_size * hidden_size
                attn_scores_d = 4 * bsz * seq_len * hidden_size  # context window S
                mlp_d = 6 * bsz * hidden_size * inter_size
                node.flops_decode = int(attn_linear_d + attn_scores_d + mlp_d)
                node.kv_bytes_per_token = kv_bytes_per_token
            elif "lm_head" in target and hidden_size > 0 and vocab_size > 0:
                node.flops_prefill = int(2 * bsz * seq_len * hidden_size * vocab_size)
                node.flops_decode = int(2 * bsz * hidden_size * vocab_size)

    return {
        "num_nodes": len(helm_graph.nodes),
        "num_nodes_with_shapes": 0,
        "total_activation_bytes": sum(n.activation_bytes for n in helm_graph.nodes),
        "total_param_bytes": sum(n.param_bytes for n in helm_graph.nodes),
        "total_flops_prefill": sum(n.flops_prefill for n in helm_graph.nodes),
        "total_flops_decode": sum(n.flops_decode for n in helm_graph.nodes),
        "total_kv_bytes_per_token": sum(n.kv_bytes_per_token for n in helm_graph.nodes),
        "analysis_mode": "static_fallback",
    }


def _build_partition_units(helm_graph: HelmGraph) -> List[PartitionUnit]:
    builder = PartitionUnitBuilder(helm_graph)
    return builder.build()


def _unit_to_dict(unit: PartitionUnit) -> Dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type,
        "layer_start": unit.layer_start,
        "layer_end": unit.layer_end,
        "node_ids": list(unit.node_ids),
        "flops_prefill": unit.flops_prefill,
        "flops_decode": unit.flops_decode,
        "param_bytes": unit.param_bytes,
        "activation_bytes": unit.activation_bytes,
        "kv_bytes_per_token": unit.kv_bytes_per_token,
        "contains_attention": unit.contains_attention,
        "contains_mlp": unit.contains_mlp,
        "contains_norm": unit.contains_norm,
        "contains_kv_projection": unit.contains_kv_projection,
    }


def _node_to_dict(node: Any) -> Dict[str, Any]:
    return {
        "node_id": node.id,
        "fx_node_name": node.fx_node_name,
        "op_type": node.op_type,
        "target": str(node.target),
        "module_path": node.module_path,
        "layer_id": node.layer_id,
        "input_shapes": node.input_shapes,
        "output_shapes": node.output_shapes,
        "activation_bytes": node.activation_bytes,
        "param_bytes": node.param_bytes,
        "flops_prefill": node.flops_prefill,
        "flops_decode": node.flops_decode,
        "kv_bytes_per_token": node.kv_bytes_per_token,
        "is_attention": node.is_attention,
        "is_mlp": node.is_mlp,
        "is_norm": node.is_norm,
        "is_embedding": node.is_embedding,
        "is_output_head": node.is_output_head,
        "is_kv_projection": node.is_kv_projection,
    }


def _estimate_device_profiles(allow_gpu: bool) -> Tuple[Dict[str, DeviceProfile], Dict[Tuple[str, str], LinkProfile]]:
    devices: Dict[str, DeviceProfile] = {}

    mem = psutil.virtual_memory()
    devices["cpu"] = DeviceProfile(
        device_id="cpu",
        device_type="cpu",
        peak_flops_prefill=2e12,
        peak_flops_decode=2e12,
        mem_bandwidth=50e9,
        memory_capacity=int(mem.total),
    )

    if allow_gpu and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Estimate FP16 TFLOPS from SM count and architecture tier.
        # Reference: A100 (108 SMs, compute 8.0) → ~312 TFLOPS FP16 tensor core.
        # Consumer Ada (compute 8.9) has higher per-SM throughput than Ampere.
        # Per-SM FP16 scaling: ~2.9 TFLOPS/SM for Ada, ~2.4 TFLOPS/SM for Ampere.
        major, minor = props.major, props.minor
        if (major, minor) >= (9, 0):       # Hopper
            tflops_per_sm = 4.0e12
        elif (major, minor) >= (8, 9):     # Ada Lovelace (RTX 40xx)
            tflops_per_sm = 2.9e12
        elif (major, minor) >= (8, 0):     # Ampere (RTX 30xx / A100)
            tflops_per_sm = 2.4e12
        else:                              # Turing / older
            tflops_per_sm = 1.6e12
        estimated_fp16_flops = props.multi_processor_count * tflops_per_sm
        # Memory bandwidth: approximate from VRAM size tier (consumer GDDR6/X).
        # 8 GB tier ≈ 272 GB/s, 16 GB ≈ 384 GB/s, 24+ GB ≈ 900+ GB/s (A100 HBM).
        vram_gb = props.total_memory / 1024 ** 3
        if vram_gb >= 40:
            mem_bw = 2000e9   # HBM2e / HBM3
        elif vram_gb >= 20:
            mem_bw = 960e9    # GDDR6X high-end / HBM2
        elif vram_gb >= 12:
            mem_bw = 504e9    # RTX 4070 Ti class
        else:
            mem_bw = 272e9    # RTX 4060 / 4060 Ti class
        devices["cuda"] = DeviceProfile(
            device_id="cuda",
            device_type="cuda",
            peak_flops_prefill=estimated_fp16_flops,
            peak_flops_decode=estimated_fp16_flops,
            mem_bandwidth=mem_bw,
            memory_capacity=int(props.total_memory),
        )

    links = {
        ("cpu", "cuda"): LinkProfile(src="cpu", dst="cuda", bandwidth_bytes_per_s=16e9, latency_s=10e-6),
        ("cuda", "cpu"): LinkProfile(src="cuda", dst="cpu", bandwidth_bytes_per_s=16e9, latency_s=10e-6),
    }

    return devices, links


def _manual_partition_plan(
    units: Sequence[PartitionUnit],
    cpu_layers: Optional[str],
    gpu_layers: Optional[str],
    model_name: str,
) -> Tuple[PartitionPlan, Dict[str, Any]]:
    cpu_range = parse_layer_spec(cpu_layers)
    gpu_range = parse_layer_spec(gpu_layers)

    if not cpu_range and not gpu_range:
        raise ValueError("manual plan requires at least one of --cpu-layers or --gpu-layers")

    overlap = cpu_range & gpu_range
    if overlap:
        raise ValueError(f"overlapping CPU/GPU layer assignments: {sorted(overlap)}")

    transformer_devices: List[str] = []
    for unit in units:
        if unit.unit_type != "transformer_block" or unit.layer_start is None:
            continue
        if unit.layer_start in cpu_range:
            transformer_devices.append("cpu")
        elif unit.layer_start in gpu_range:
            transformer_devices.append("cuda")
        else:
            transformer_devices.append("cuda")
    output_device = transformer_devices[-1] if transformer_devices else ("cpu" if cpu_range else "cuda")

    assignments: Dict[int, str] = {}
    for unit in units:
        assigned = "cuda"
        if unit.unit_type == "transformer_block" and unit.layer_start is not None:
            if unit.layer_start in cpu_range:
                assigned = "cpu"
            elif unit.layer_start in gpu_range:
                assigned = "cuda"
            else:
                assigned = "cuda"
        elif unit.unit_type == "embedding":
            assigned = "cpu" if cpu_range else "cuda"
        elif unit.unit_type == "output":
            assigned = output_device
        assignments[unit.unit_id] = assigned

    stages: List[StageSpec] = []
    current_device: Optional[str] = None
    current_units: List[PartitionUnit] = []

    for unit in units:
        unit_device = assignments[unit.unit_id]
        if current_device is None:
            current_device = unit_device

        if unit_device != current_device:
            stage_id = len(stages)
            layers = [u.layer_start for u in current_units if u.layer_start is not None]
            stages.append(
                StageSpec(
                    stage_id=stage_id,
                    device_id=current_device,
                    units=list(current_units),
                    layer_start=min(layers) if layers else 0,
                    layer_end=max(layers) if layers else 0,
                )
            )
            current_units = []
            current_device = unit_device

        current_units.append(unit)

    if current_units:
        stage_id = len(stages)
        layers = [u.layer_start for u in current_units if u.layer_start is not None]
        stages.append(
            StageSpec(
                stage_id=stage_id,
                device_id=current_device or "cpu",
                units=list(current_units),
                layer_start=min(layers) if layers else 0,
                layer_end=max(layers) if layers else 0,
            )
        )

    plan = PartitionPlan(stages=stages)

    plan_data = {
        "mode": "manual",
        "model": model_name,
        "assignments": assignments,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "device": stage.device_id,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "unit_ids": [u.unit_id for u in stage.units],
            }
            for stage in stages
        ],
        "summary": {
            "num_units": len(units),
            "num_cpu_units": sum(1 for u in units if assignments[u.unit_id] == "cpu"),
            "num_gpu_units": sum(1 for u in units if assignments[u.unit_id] != "cpu"),
        },
    }
    return plan, plan_data


def _auto_partition_plan(
    units: Sequence[PartitionUnit],
    options: HelmCompileOptions,
    model_name: str,
    hidden_size: int = 4096,
    model_dtype: torch.dtype = torch.float16,
    hf_config=None,
) -> Tuple[PartitionPlan, Dict[str, Any], Any]:
    devices, links = profile_devices(
        hidden_size=hidden_size,
        dtype=model_dtype,
        allow_gpu=options.allow_gpu,
        n_cpu_units=len(units),
        gpu_memory_reserve_mb=options.gpu_memory_reserve_mb,
    )
    for dev_id, prof in devices.items():
        logger.info("[HELM] device_profiler [%s]: decode=%.2f TFLOPS  prefill=%.2f TFLOPS  mem_bw=%.1f GB/s  capacity=%.2f GB",
                    dev_id, prof.peak_flops_decode / 1e12, prof.peak_flops_prefill / 1e12,
                    prof.mem_bandwidth / 1e9, prof.memory_capacity / 1e9)

    available_devices = list(devices.keys())
    if not options.allow_cpu:
        available_devices = [d for d in available_devices if d != "cpu"]
    if not available_devices:
        raise RuntimeError("no devices available for planning")

    dtype_size = 2 if model_dtype in (torch.float16, torch.bfloat16) else 4
    model_config = None
    if hf_config is not None:
        try:
            model_config = ModelConfig.from_hf_config(hf_config, dtype_size=dtype_size)
        except Exception as exc:
            raise RuntimeError(
                f"HELM auto partition failed to build ModelConfig for model "
                f"{model_name or '<unknown>'!r} from {type(hf_config).__name__}: {exc}"
            ) from exc

    # When kv_offload=True and no explicit reserve was requested, default to
    # 512 tokens — matching the KV offload manager's default GPU watermark.
    kv_reserve_tokens = options.kv_reserve_tokens
    if options.kv_offload and kv_reserve_tokens == 0:
        kv_reserve_tokens = 512

    selector = StrategySelector(
        cost_model=HelmCostModel(devices=devices, links=links, model_config=model_config),
        devices=available_devices,
        config=StrategySelectorConfig(
            objective=options.objective,
            max_stages=options.max_stages,
            allow_cpu=options.allow_cpu,
            allow_multi_gpu=options.allow_multi_gpu,
            kv_offload=options.kv_offload,
            kv_reserve_tokens=kv_reserve_tokens,
        ),
    )

    wk = options.workload or {}
    workload = WorkloadSpec(
        batch_size=wk.get("batch_size", 1),
        prefill_seq_len=wk.get("prefill_seq_len", 64),
        decode_context_len=wk.get("decode_context_len", 64),
        decode_tokens=wk.get("decode_tokens", 8),
    )

    total_param_bytes = sum(u.param_bytes for u in units)
    logger.info("[HELM] partition units: %d units, total params=%.2f GB", len(units), total_param_bytes / 1e9)
    for u in units:
        logger.debug("  unit %d (%s layer=%s): param=%.0f MB", u.unit_id, u.unit_type, u.layer_start, u.param_bytes / 1e6)

    plan, best_cost = selector.select(list(units), workload)
    for stage in plan.stages:
        stage_bytes = sum(u.param_bytes for u in stage.units)
        logger.info("[HELM] auto partition: stage %d -> %s  units %s-%s  params=%.2f GB",
                    stage.stage_id, stage.device_id, stage.layer_start, stage.layer_end, stage_bytes / 1e9)

    assignments: Dict[int, str] = {}
    for stage in plan.stages:
        for unit in stage.units:
            assignments[unit.unit_id] = stage.device_id

    plan_data = {
        "mode": "auto",
        "model": model_name,
        "assignments": assignments,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "device": stage.device_id,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "unit_ids": [u.unit_id for u in stage.units],
            }
            for stage in plan.stages
        ],
        "summary": {
            "num_units": len(units),
            "num_cpu_units": sum(1 for u in units if assignments.get(u.unit_id) == "cpu"),
            "num_gpu_units": sum(1 for u in units if assignments.get(u.unit_id) != "cpu"),
        },
    }

    return plan, plan_data, best_cost


def build_partition_plan(
    units: Sequence[PartitionUnit],
    options: HelmCompileOptions,
    hidden_size: int = 4096,
    model_dtype: torch.dtype = torch.float16,
    hf_config=None,
) -> Tuple[PartitionPlan, Dict[str, Any], Any]:
    if options.plan_mode == "manual":
        plan, plan_data = _manual_partition_plan(
            units=units,
            cpu_layers=options.cpu_layers,
            gpu_layers=options.gpu_layers,
            model_name=options.model_name,
        )
        return plan, plan_data, None
    plan, plan_data, best_cost = _auto_partition_plan(
        units=units,
        options=options,
        model_name=options.model_name,
        hidden_size=hidden_size,
        model_dtype=model_dtype,
        hf_config=hf_config,
    )
    return plan, plan_data, best_cost


def build_schedule(plan: PartitionPlan) -> Dict[str, Any]:
    sched = Scheduler().build(plan)
    return {
        "mode": sched.mode,
        "num_stages": sched.num_stages,
        "microbatches": sched.microbatches,
    }


def lower_to_runtime_stages(
    gm: torch.fx.GraphModule,
    helm_graph: HelmGraph,
    units: Sequence[PartitionUnit],
    partition_plan: PartitionPlan,
) -> List[Any]:
    return StageFXBuilder(gm, helm_graph, list(units), partition_plan).build()


def _collect_hardware_meta() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    meta: Dict[str, Any] = {
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "system_ram_total_bytes": int(mem.total),
        "system_ram_available_bytes": int(mem.available),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_sm_count": int(props.multi_processor_count),
            }
        )

    return meta


def validate_helmir(helmir: HelmIR) -> None:
    data = helmir.to_dict()
    required = ["schema_version", "graph_kind", "model_name", "nodes", "units", "plan", "schedule", "hardware", "metadata"]
    for key in required:
        if key not in data:
            raise ValueError(f"helmir missing required key: {key}")

    node_required = [
        "node_id",
        "op_type",
        "target",
        "input_shapes",
        "output_shapes",
        "activation_bytes",
        "param_bytes",
        "flops_prefill",
        "flops_decode",
        "kv_bytes_per_token",
    ]
    for idx, node in enumerate(data["nodes"]):
        for key in node_required:
            if key not in node:
                raise ValueError(f"helmir node[{idx}] missing '{key}'")


def compile_graph(
    gm: torch.fx.GraphModule,
    example_inputs: Any,
    model: torch.nn.Module,
    *,
    tokenizer: Optional[Any] = None,
    options: Optional[HelmCompileOptions] = None,
    partition_plan_override: Optional[PartitionPlan] = None,
    artifacts_dir: Optional[str] = None,
) -> CompiledHelmArtifact:
    opts = options or HelmCompileOptions()

    helm_graph = _build_helm_graph(gm)
    analyzer = None
    try:
        analyzer, analysis_summary = _run_analysis(
            gm=gm,
            helm_graph=helm_graph,
            model=model,
            example_inputs=example_inputs,
        )
        analysis_summary["analysis_mode"] = "dynamic"
    except Exception as exc:
        model_label = opts.model_name or type(model).__name__
        if not opts.allow_static_analysis_fallback:
            raise RuntimeError(
                f"HELM dynamic analysis failed for model {model_label!r}, "
                f"graph_kind={opts.graph_kind!r}: {exc}"
            ) from exc
        try:
            analysis_summary = _run_static_fallback_analysis(
                gm=gm,
                helm_graph=helm_graph,
                model=model,
                workload=opts.workload,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"HELM static analysis fallback failed for model {model_label!r}, "
                f"graph_kind={opts.graph_kind!r} after dynamic analysis failed with: {exc}"
            ) from fallback_exc
        analysis_summary["analysis_error"] = str(exc)

    units = _build_partition_units(helm_graph)
    cfg = getattr(model, "config", None)
    _hidden_size = int(getattr(cfg, "hidden_size", 4096) or 4096)
    _model_dtype = next((p.dtype for p in model.parameters()), torch.float16)
    plan_cost = None
    if partition_plan_override is None:
        plan, plan_data, plan_cost = build_partition_plan(
            units, opts,
            hidden_size=_hidden_size,
            model_dtype=_model_dtype,
            hf_config=cfg,
        )
    else:
        plan = partition_plan_override
        plan_data = {
            "mode": "fixed",
            "model": opts.model_name,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "device": stage.device_id,
                    "layer_start": stage.layer_start,
                    "layer_end": stage.layer_end,
                    "unit_ids": [u.unit_id for u in stage.units],
                }
                for stage in plan.stages
            ],
            "summary": {
                "num_units": len(units),
                "num_cpu_units": sum(1 for stage in plan.stages for _ in stage.units if stage.device_id == "cpu"),
                "num_gpu_units": sum(1 for stage in plan.stages for _ in stage.units if stage.device_id != "cpu"),
            },
            "assignments": {
                u.unit_id: stage.device_id for stage in plan.stages for u in stage.units
            },
        }
    schedule = build_schedule(plan)

    stage_graphs: List[Any] = []
    if opts.lower_stages:
        stage_graphs = lower_to_runtime_stages(gm, helm_graph, units, plan)

    helmir = HelmIR(
        schema_version="0.1",
        graph_kind=opts.graph_kind,
        model_name=opts.model_name,
        nodes=[_node_to_dict(n) for n in helm_graph.nodes],
        units=[_unit_to_dict(u) for u in units],
        plan=plan_data,
        schedule=schedule,
        hardware=_collect_hardware_meta(),
        metadata={
            "analysis": analysis_summary,
            "graph_summary": {
                "num_nodes": len(helm_graph.nodes),
                "num_edges": len(helm_graph.edges),
                "input_node_ids": helm_graph.input_node_ids,
                "output_node_ids": helm_graph.output_node_ids,
            },
        },
    )
    validate_helmir(helmir)

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)

        if analyzer is not None:
            analyzer.export_summary(os.path.join(artifacts_dir, "hybrid_analysis.json"))
        else:
            with open(os.path.join(artifacts_dir, "hybrid_analysis.json"), "w") as f:
                json.dump([_node_to_dict(n) for n in helm_graph.nodes], f, indent=2)

        with open(os.path.join(artifacts_dir, "partition_units.json"), "w") as f:
            json.dump([_unit_to_dict(u) for u in units], f, indent=2)

        with open(os.path.join(artifacts_dir, "partition_plan.json"), "w") as f:
            json.dump(plan_data, f, indent=2)

        with open(os.path.join(artifacts_dir, "schedule.json"), "w") as f:
            json.dump(schedule, f, indent=2)

        with open(os.path.join(artifacts_dir, f"helmir_{opts.graph_kind}.json"), "w") as f:
            json.dump(helmir.to_dict(), f, indent=2)

        summary_path = os.path.join(artifacts_dir, "graph_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"nodes: {len(helm_graph.nodes)}\n")
            f.write(f"edges: {len(helm_graph.edges)}\n")
            f.write(f"inputs: {helm_graph.input_node_ids}\n")
            f.write(f"outputs: {helm_graph.output_node_ids}\n")
            for node in helm_graph.nodes:
                f.write(f"{node}\n")

        if stage_graphs:
            for stage in stage_graphs:
                with open(os.path.join(artifacts_dir, f"stage_{stage.stage_id}_fx.txt"), "w") as f:
                    f.write(f"Stage {stage.stage_id} (Device: {stage.device})\n")
                    f.write(str(stage.module.graph))

    return CompiledHelmArtifact(
        gm=gm,
        helm_graph=helm_graph,
        analysis_summary=analysis_summary,
        partition_units=units,
        partition_plan=plan,
        plan_data=plan_data,
        plan_cost=plan_cost,
        schedule=schedule,
        stage_graphs=stage_graphs,
        helmir=helmir,
        metadata={"options": opts.__dict__.copy()},
    )


def helm_backend(gm: torch.fx.GraphModule, example_inputs, **kwargs):
    """
    Torch.compile backend contract.

    For now we compile metadata/plan and return a runtime wrapper that delegates
    to the original GraphModule. This keeps backend integration stable while
    the staged runtime binding is expanded.
    """

    options = HelmCompileOptions(**kwargs)

    artifact = compile_graph(
        gm=gm,
        example_inputs=example_inputs,
        model=gm,
        options=options,
        artifacts_dir=None,
    )

    def runtime_wrapper(*args, **runtime_kwargs):
        return artifact.gm.forward(*args, **runtime_kwargs)

    return runtime_wrapper
