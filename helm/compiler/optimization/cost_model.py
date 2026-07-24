# helm/compiler/optimization/cost_model.py

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from helm.compiler.partition.partition_plan import PartitionPlan, StageSpec
from helm.compiler.partition.partition_units import PartitionUnit

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

# Fixed overhead reserved for PyTorch / CUDA runtime allocations (buffers,
# workspace, cublas handles, etc.) that are not captured by param/activation
# accounting.  Adjust if profiling shows a different steady-state residual.
_RUNTIME_OVERHEAD_BYTES: int = 200 * 1024 * 1024  # 200 MB

# Batch size at which decode matmuls transition from GEMV (poor SM utilisation)
# to full GEMM throughput.  Linear interpolation between peak_flops_decode and
# peak_flops_prefill uses this as the upper bound.  32 is a typical GPU
# crossover; tune via validate() if your hardware differs.
_GEMM_TRANSITION_BATCH: int = 32


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """
    Architecture constants extracted from model.config.
    Passed to HelmCostModel so it can compute analytical FLOPs and memory
    traffic without relying on the pre-baked estimates stored in PartitionUnit
    (those embed the analysis sequence length and cannot be rescaled reliably).
    """
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int           # = hidden_size // num_attention_heads
    dtype_size: int = 2     # bytes per element (2 = fp16/bf16, 4 = fp32)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"ModelConfig.hidden_size must be > 0, got {self.hidden_size!r}")
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"ModelConfig.num_attention_heads must be > 0, got {self.num_attention_heads!r}"
            )
        if self.head_dim <= 0:
            raise ValueError(f"ModelConfig.head_dim must be > 0, got {self.head_dim!r}")
        if self.dtype_size <= 0:
            raise ValueError(
                f"ModelConfig.dtype_size must be > 0 (e.g. 1=INT8, 2=fp16/bf16, 4=fp32), got {self.dtype_size!r}"
            )

    @staticmethod
    def from_hf_config(cfg, dtype_size: int = 2) -> "ModelConfig":
        num_heads    = int(getattr(cfg, "num_attention_heads", 1))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        hidden_size  = int(getattr(cfg, "hidden_size", 0))
        head_dim     = int(getattr(cfg, "head_dim",
                           hidden_size // num_heads if num_heads > 0 else 0))
        return ModelConfig(
            hidden_size         = hidden_size,
            intermediate_size   = int(getattr(cfg, "intermediate_size", 0)),
            num_attention_heads = num_heads,
            num_kv_heads        = num_kv_heads,
            head_dim            = head_dim,
            dtype_size          = dtype_size,
        )


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    device_type: str            # "cpu" | "cuda"
    peak_flops_prefill: float   # measured effective FLOPS/s (from device_profiler microbenchmark)
    peak_flops_decode: float    # measured effective FLOPS/s at seq=1 (GEMV regime)
    mem_bandwidth: float        # measured DRAM bandwidth (bytes/s, from stream copy benchmark)
    memory_capacity: int        # bytes
    # ── calibration factors ──────────────────────────────────────────────────
    # Multiplicative efficiency factors (0 < eff ≤ 1.0) applied on top of the
    # microbenchmark-measured peak values.  Set below 1.0 if validate() shows
    # the model over-predicts performance for the actual workload shapes.
    efficiency_compute: float = 1.0
    efficiency_memory:  float = 1.0
    # ── CPU L3 cache model ───────────────────────────────────────────────────
    # When set, attention KV accesses that fit in L3 use l3_bandwidth instead
    # of mem_bandwidth.  0 disables L3 modeling (conservative; safe default).
    l3_size_bytes: int   = 0
    l3_bandwidth:  float = 0.0

    def __post_init__(self) -> None:
        if self.device_type not in ("cpu", "cuda"):
            raise ValueError(
                f"DeviceProfile.device_type must be 'cpu' or 'cuda', got {self.device_type!r}"
            )
        if self.efficiency_compute <= 0.0:
            raise ValueError(
                f"DeviceProfile.efficiency_compute must be > 0, got {self.efficiency_compute!r}"
            )
        if self.efficiency_memory <= 0.0:
            raise ValueError(
                f"DeviceProfile.efficiency_memory must be > 0, got {self.efficiency_memory!r}"
            )
        if self.peak_flops_prefill < 0:
            raise ValueError(
                f"DeviceProfile.peak_flops_prefill must be >= 0, got {self.peak_flops_prefill!r}"
            )
        if self.peak_flops_decode < 0:
            raise ValueError(
                f"DeviceProfile.peak_flops_decode must be >= 0, got {self.peak_flops_decode!r}"
            )
        if self.mem_bandwidth < 0:
            raise ValueError(
                f"DeviceProfile.mem_bandwidth must be >= 0, got {self.mem_bandwidth!r}"
            )


@dataclass(frozen=True)
class LinkProfile:
    src: str
    dst: str
    bandwidth_bytes_per_s: float
    latency_s: float

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError(
                f"LinkProfile.bandwidth_bytes_per_s must be > 0, got {self.bandwidth_bytes_per_s!r}"
            )
        if self.latency_s < 0:
            raise ValueError(
                f"LinkProfile.latency_s must be >= 0, got {self.latency_s!r}"
            )


@dataclass(frozen=True)
class WorkloadSpec:
    batch_size: int
    prefill_seq_len: int
    decode_context_len: int
    decode_tokens: int
    kv_quant_ratio: float = 1.0  # 1.0=fp16 KV, 0.5=INT8 KV, 0.25=INT4 KV

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"WorkloadSpec.batch_size must be > 0, got {self.batch_size!r}")
        if self.prefill_seq_len <= 0:
            raise ValueError(
                f"WorkloadSpec.prefill_seq_len must be > 0, got {self.prefill_seq_len!r}"
            )
        if self.decode_context_len < 0:
            raise ValueError(
                f"WorkloadSpec.decode_context_len must be >= 0, got {self.decode_context_len!r}"
            )
        if self.decode_tokens < 0:
            raise ValueError(
                f"WorkloadSpec.decode_tokens must be >= 0, got {self.decode_tokens!r}"
            )
        if not (0.0 < self.kv_quant_ratio <= 1.0):
            raise ValueError(
                f"WorkloadSpec.kv_quant_ratio must be in (0, 1.0], "
                f"got {self.kv_quant_ratio!r}"
            )


@dataclass
class StageCost:
    stage_id: int
    device: str
    param_bytes: int
    activation_bytes: int
    kv_bytes: int
    memory_bytes: int
    prefill_compute_s: float
    prefill_memory_s: float
    prefill_comm_s: float
    prefill_total_s: float
    decode_compute_s: float
    decode_memory_s: float
    decode_comm_s: float
    decode_total_s: float
    feasible: bool
    peak_intermediate_bytes: int = 0   # peak transient memory within one transformer block
    # Whether the decode step is compute- or memory-bandwidth-bound
    decode_regime: str = "memory"   # "memory" | "compute"
    reason: Optional[str] = None


@dataclass
class PlanCost:
    feasible: bool
    stage_costs: List[StageCost] = field(default_factory=list)
    prefill_latency_s: float = 0.0
    decode_token_latency_s: float = 0.0
    total_latency_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    max_stage_memory_bytes: int = 0
    reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

class HelmCostModel:
    """
    Roofline-based cost model for heterogeneous CPU+GPU inference.

    Design principles
    -----------------
    All FLOPs and memory-traffic formulas are derived analytically from
    ModelConfig so they scale correctly with the actual workload (batch size,
    context length) rather than relying on the analysis-time estimates
    embedded in PartitionUnit (which have B and S baked in at analysis time
    and cannot be correctly rescaled for arbitrary workloads).

    param_bytes and kv_bytes_per_token from PartitionUnit are used directly —
    they are workload-independent and measured exactly from the model weights.

    Roofline: stage_time = max(compute_time, memory_time).

    At decode (seq=1) projection layers are always memory-bandwidth-bound on
    both CPU and GPU for current hardware.  At large prefill sequences the GPU
    can flip to compute-bound for projections once arithmetic intensity
    (FLOPs/byte) exceeds the hardware ops-per-byte ratio.

    Requirements
    ------------
    model_config is preferred for exact activation, FLOP, and cross-device
    communication estimates.  When it is unavailable, the model falls back to
    PartitionUnit activation_bytes and stored FLOP counts.
    """

    def __init__(
        self,
        devices: Dict[str, DeviceProfile],
        links: Dict[Tuple[str, str], LinkProfile],
        model_config: Optional[ModelConfig] = None,
    ):
        self.devices      = devices
        self.links        = links
        self.model_config = model_config

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_plan(
        self,
        plan: PartitionPlan,
        workload: WorkloadSpec,
        kv_offload: bool = False,
        kv_reserve_tokens: int = 0,
    ) -> PlanCost:
        """
        Estimate the full cost of a partition plan.

        When kv_offload=True, KV pages are evicted to CPU RAM as context grows.
        kv_reserve_tokens controls how many tokens' worth of KV stays on the
        GPU before eviction begins.  Must be > 0 when kv_offload=True so the
        GPU memory budget accounts for at least the hot working set.

        On the first infeasible stage, returns early.  stage_costs will contain
        only the stages evaluated so far (0 .. infeasible stage inclusive).
        validate() raises on an infeasible plan to avoid partial calibration.
        """
        if not plan.stages:
            raise ValueError("PartitionPlan must contain at least one stage")

        if kv_offload and kv_reserve_tokens == 0:
            raise ValueError(
                "kv_reserve_tokens must be > 0 when kv_offload=True. "
                "Set it to the number of GPU-resident KV tokens kept before eviction "
                "(e.g. prefill_seq_len for a single-request warm cache)."
            )

        plan_cost = PlanCost(feasible=True)
        max_stage_mem   = 0
        total_prefill_s = 0.0
        total_decode_s  = 0.0

        for i, stage in enumerate(plan.stages):
            next_stage = plan.stages[i + 1] if i + 1 < len(plan.stages) else None
            sc = self.estimate_stage(stage, workload, next_stage, kv_offload=kv_offload,
                                     kv_reserve_tokens=kv_reserve_tokens)
            plan_cost.stage_costs.append(sc)

            if not sc.feasible:
                plan_cost.feasible = False
                plan_cost.reason = (
                    f"Stage {stage.stage_id} on {stage.device_id} infeasible: {sc.reason}"
                )
                return plan_cost

            max_stage_mem   = max(max_stage_mem, sc.memory_bytes)
            total_prefill_s += sc.prefill_total_s
            total_decode_s  += sc.decode_total_s

        plan_cost.prefill_latency_s      = total_prefill_s
        plan_cost.decode_token_latency_s = total_decode_s
        plan_cost.total_latency_s = (
            total_prefill_s + workload.decode_tokens * total_decode_s
        )
        # Pipeline throughput: at steady state, one batch of tokens completes
        # every bottleneck-stage cycle, not every full serial pass.
        bottleneck_decode_s = max(sc.decode_total_s for sc in plan_cost.stage_costs)
        if bottleneck_decode_s > 0:
            plan_cost.throughput_tokens_per_s = workload.batch_size / bottleneck_decode_s
        plan_cost.max_stage_memory_bytes = max_stage_mem
        return plan_cost

    def estimate_stage(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        next_stage: Optional[StageSpec] = None,
        kv_offload: bool = False,
        kv_reserve_tokens: int = 0,
    ) -> StageCost:
        if not stage.units:
            raise ValueError(f"Stage {stage.stage_id} must contain at least one unit")

        device = self.devices.get(stage.device_id)
        if device is None:
            raise KeyError(
                f"Device {stage.device_id!r} not found in HelmCostModel.devices. "
                f"Registered devices: {sorted(self.devices.keys())}"
            )

        # Memory sizing
        param_bytes, act_prefill, act_decode, kv_bytes = self._stage_memory(
            stage, workload
        )
        if kv_offload and device.device_type == "cuda":
            # KV pages are evicted to CPU RAM as context grows, so the full kv_bytes
            # budget need not fit on GPU.  Reserve headroom for kv_reserve_tokens
            # tokens, which are always kept on GPU before eviction.
            kv_per_token = sum(u.kv_bytes_per_token for u in stage.units)
            kv_for_budget = int(kv_per_token * workload.batch_size * kv_reserve_tokens * workload.kv_quant_ratio)
        else:
            kv_for_budget = kv_bytes

        peak_intermediate = self._peak_intermediate_bytes(stage, workload, device.device_type)
        margin    = int(0.05 * (param_bytes + act_prefill)) + _RUNTIME_OVERHEAD_BYTES
        total_mem = param_bytes + max(act_prefill, act_decode) + kv_for_budget + peak_intermediate + margin

        feasible = total_mem <= device.memory_capacity
        reason   = (
            None if feasible
            else f"OOM: need {total_mem/1e9:.2f} GB, device has {device.memory_capacity/1e9:.2f} GB"
        )

        pre_comp, pre_mem, pre_base = self._prefill_time(stage, workload, device)
        dec_comp, dec_mem, dec_base = self._decode_time(stage, workload, device)
        pre_comm, dec_comm = self._boundary_comm(stage, next_stage, workload)

        return StageCost(
            stage_id                = stage.stage_id,
            device                  = stage.device_id,
            param_bytes             = param_bytes,
            activation_bytes        = max(act_prefill, act_decode),
            kv_bytes                = kv_bytes,
            peak_intermediate_bytes = peak_intermediate,
            memory_bytes            = total_mem,
            prefill_compute_s= pre_comp,
            prefill_memory_s = pre_mem,
            prefill_comm_s   = pre_comm,
            prefill_total_s  = pre_base + pre_comm,
            decode_compute_s = dec_comp,
            decode_memory_s  = dec_mem,
            decode_comm_s    = dec_comm,
            decode_total_s   = dec_base + dec_comm,
            feasible         = feasible,
            decode_regime    = "compute" if dec_comp > dec_mem else "memory",
            reason           = reason,
        )

    def validate(
        self,
        plan: PartitionPlan,
        workload: WorkloadSpec,
        measured_ms: Dict[int, Dict[str, float]],
        kv_offload: bool = False,
        kv_reserve_tokens: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        """
        Compare cost model predictions against measured per-stage wall times.

        Parameters
        ----------
        measured_ms
            {stage_id: {"decode_ms": float, "prefill_ms": float}}
            Actual measured times, e.g. from executor._PROFILE output.

        Returns
        -------
        Per-stage dict with predicted vs actual times, percent error, and
        suggested efficiency factors that would make the model match the
        measurements.  Pass these as DeviceProfile.efficiency_memory /
        efficiency_compute to calibrate the model for this hardware.

        Note: validate() requires a feasible plan and one measurement entry per
        planned stage.  Missing measurements raise ValueError instead of being
        skipped because partial calibration can hide broken stage accounting.

        How to interpret suggested_efficiency_memory
        --------------------------------------------
        If the stage is memory-bound, the raw memory time (bytes/bw) is
        `sc.decode_memory_s`.  The model predicts `raw / efficiency`.  If
        the prediction is off, the implied efficiency that would match the
        measurement is `raw / actual_s`.  A value < 1.0 means the hardware
        achieves less than the stream-benchmark BW for this workload.
        """
        plan_cost = self.estimate_plan(plan, workload, kv_offload=kv_offload,
                                       kv_reserve_tokens=kv_reserve_tokens)
        if not plan_cost.feasible:
            raise RuntimeError(
                "HelmCostModel.validate cannot calibrate an infeasible plan: "
                f"{plan_cost.reason or 'unknown feasibility failure'}"
            )

        plan_stage_ids = {stage.stage_id for stage in plan.stages}
        measured_stage_ids = set(measured_ms)
        missing_stage_ids = sorted(plan_stage_ids - measured_stage_ids)
        if missing_stage_ids:
            raise ValueError(
                f"HelmCostModel.validate missing measured timings for stage_id(s): "
                f"{missing_stage_ids}"
            )
        unknown_stage_ids = sorted(measured_stage_ids - plan_stage_ids)
        if unknown_stage_ids:
            raise ValueError(
                f"HelmCostModel.validate got measured timings for unknown stage_id(s): "
                f"{unknown_stage_ids}"
            )

        report: Dict[int, Dict[str, float]] = {}
        for sc in plan_cost.stage_costs:
            sid = sc.stage_id
            m = measured_ms[sid]

            pred_dec   = sc.decode_total_s  * 1000
            pred_pre   = sc.prefill_total_s * 1000
            actual_dec = m.get("decode_ms",  0.0)
            actual_pre = m.get("prefill_ms", 0.0)

            err_dec = (
                abs(pred_dec - actual_dec) / actual_dec * 100
                if actual_dec > 0 else float("nan")
            )
            err_pre = (
                abs(pred_pre - actual_pre) / actual_pre * 100
                if actual_pre > 0 else float("nan")
            )

            # Suggested efficiency: raw_memory_s / actual_s (if memory-bound).
            # sc.decode_memory_s is raw (bytes / bw, no efficiency applied).
            sug_eff_dec = None
            if actual_dec > 0 and sc.decode_memory_s > 0 and sc.decode_regime == "memory":
                sug_eff_dec = round(sc.decode_memory_s / (actual_dec / 1000), 3)

            sug_eff_pre = None
            if actual_pre > 0 and sc.prefill_memory_s > 0:
                sug_eff_pre = round(sc.prefill_memory_s / (actual_pre / 1000), 3)

            report[sid] = {
                "predicted_decode_ms":          round(pred_dec, 2),
                "actual_decode_ms":             round(actual_dec, 2),
                "decode_error_pct":             round(err_dec, 1),
                "predicted_prefill_ms":         round(pred_pre, 2),
                "actual_prefill_ms":            round(actual_pre, 2),
                "prefill_error_pct":            round(err_pre, 1),
                "decode_regime":                sc.decode_regime,
                "suggested_efficiency_memory":  sug_eff_dec,
                "suggested_efficiency_memory_prefill": sug_eff_pre,
            }
        return report

    # ── Internal: memory sizing ───────────────────────────────────────────────

    def _stage_memory(
        self, stage: StageSpec, workload: WorkloadSpec
    ) -> Tuple[int, int, int, int]:
        """
        Returns (param_bytes, act_prefill, act_decode, kv_bytes).

        Activation memory = the hidden-state tensor at the stage boundary:
            B × S × H × dtype_size
        This is tight — intermediate tensors (QKV projections, MLP activations)
        are transient and do not accumulate across layers.

        KV memory = all layers in this stage over the full decode context:
            sum(kv_bytes_per_token) × B × context_len
        """
        B   = workload.batch_size
        S   = workload.prefill_seq_len
        ctx = workload.decode_context_len

        param_bytes = sum(u.param_bytes for u in stage.units)
        kv_bytes = int(sum(u.kv_bytes_per_token for u in stage.units) * B * ctx * workload.kv_quant_ratio)

        if self.model_config is None:
            act_prefill = sum(max(u.activation_bytes, 0) for u in stage.units)
            act_decode = (
                max(1, act_prefill // max(S, 1))
                if act_prefill > 0
                else 0
            )
            return param_bytes, act_prefill, act_decode, kv_bytes

        H   = self.model_config.hidden_size
        dsz = self.model_config.dtype_size
        act_prefill = B * S * H * dsz
        act_decode  = B * 1 * H * dsz

        return param_bytes, act_prefill, act_decode, kv_bytes

    def _peak_intermediate_bytes(
        self, stage: StageSpec, workload: WorkloadSpec, device_type: str,
    ) -> int:
        """
        Peak transient memory within one transformer block's forward pass.

        Only one block executes at a time, so the budget needs max(mlp, attn),
        not the sum across all blocks.  Decode intermediates (S=1) are negligible.
        """
        if self.model_config is None:
            return 0
        n_blocks = sum(1 for u in stage.units if u.unit_type == "transformer_block")
        if n_blocks == 0:
            return 0
        B   = workload.batch_size
        S   = workload.prefill_seq_len
        mc  = self.model_config
        dsz = mc.dtype_size
        # MLP: gate + up projections live simultaneously for SiLU(gate) * up
        mlp_peak = B * S * 2 * mc.intermediate_size * dsz
        # Attention: CPU materialises full S×S attention scores per head;
        # GPU uses flash-attention (scores stay in SRAM, zero DRAM footprint)
        if device_type == "cpu":
            attn_peak = B * mc.num_attention_heads * S * S * dsz
        else:
            attn_peak = 0
        return max(mlp_peak, attn_peak)

    # ── Internal: helpers ─────────────────────────────────────────────────────

    def _flops_split(
        self, stage: StageSpec, workload: WorkloadSpec, for_prefill: bool
    ) -> Tuple[float, float]:
        """
        Returns (proj_flops, attn_flops) for the stage.

        Projections (Q/K/V/O + gate/up/down) scale O(S) for prefill, O(1) for
        decode.  Attention (QK^T dot products) scales O(S²) for prefill and
        O(context_len) for decode.  Keeping them separate lets each get its own
        roofline — projection GEMMs can flip compute-bound on GPU at large S
        while attention always remains memory-BW-bound (low arithmetic intensity).

        PartitionUnit.unit_type drives which formula is applied:
          "transformer_block" → full analytical GEMM + attention formula
          everything else (embedding, output) → stored flops from graph analysis
        """
        if self.model_config is None:
            total = sum(
                u.flops_prefill if for_prefill else u.flops_decode
                for u in stage.units
            )
            return float(total), 0.0

        mc  = self.model_config
        B   = workload.batch_size
        H   = mc.hidden_size
        I   = mc.intermediate_size
        kv  = mc.num_kv_heads * mc.head_dim

        proj_total = 0.0
        attn_total = 0.0
        u: PartitionUnit
        for u in stage.units:
            if u.unit_type == "transformer_block":
                if for_prefill:
                    S = workload.prefill_seq_len
                    proj_total += 2 * B * S * (H*H + H*kv + H*kv + H*H + H*I + H*I + I*H)
                    # Causal self-attention: O(S²) per head
                    attn_total += 2 * B * mc.num_attention_heads * S * S * mc.head_dim
                else:
                    ctx = workload.decode_context_len
                    proj_total += 2 * B * (H*H + H*kv + H*kv + H*H + H*I + H*I + I*H)
                    # One query token attends over ctx cached tokens
                    attn_total += 4 * B * mc.num_attention_heads * ctx * mc.head_dim
            else:
                # Embedding, norm, lm_head — not attention
                proj_total += float(
                    u.flops_prefill if for_prefill else u.flops_decode
                )
        return proj_total, attn_total

    def _effective_kv_bw(self, device: DeviceProfile, kv_bytes: int) -> float:
        """
        Effective bandwidth for KV cache reads on CPU, accounting for L3 cache.

        For short contexts the KV working set fits in L3; effective BW is the
        L3 bandwidth (typically 3–6× DRAM).  As context grows beyond l3_size,
        the fraction served from DRAM increases linearly.

        GPU devices and CPU devices without l3_bandwidth set use mem_bandwidth.
        """
        if (device.device_type != "cpu"
                or device.l3_bandwidth <= 0
                or device.l3_size_bytes <= 0
                or kv_bytes <= 0):
            return device.mem_bandwidth

        if kv_bytes <= device.l3_size_bytes:
            return device.l3_bandwidth
        # Fraction that fits in L3 vs DRAM (simplified linear blend)
        l3_frac = device.l3_size_bytes / kv_bytes
        return l3_frac * device.l3_bandwidth + (1.0 - l3_frac) * device.mem_bandwidth

    def _effective_decode_flops(self, device: DeviceProfile, batch_size: int) -> float:
        """
        Effective compute throughput for decode, scaling with batch size.

        At B=1, decode is GEMV with low SM utilisation → peak_flops_decode.
        As B grows, matmuls widen toward GEMM → peak_flops_prefill.
        Linear interpolation between B=1 and B=_GEMM_TRANSITION_BATCH.
        """
        if batch_size <= 1 or device.peak_flops_prefill <= device.peak_flops_decode:
            return device.peak_flops_decode
        alpha = min(1.0, (batch_size - 1) / (_GEMM_TRANSITION_BATCH - 1))
        return device.peak_flops_decode + alpha * (device.peak_flops_prefill - device.peak_flops_decode)

    # ── Internal: time estimates ──────────────────────────────────────────────

    def _decode_time(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        device: DeviceProfile,
    ) -> Tuple[float, float, float]:
        """
        Returns (compute_s_raw, memory_s_raw, base_s) for one decode step.

        Two separate rooflines are applied and summed:
          1. Projection roofline — bottleneck: weight bytes from DRAM
          2. Attention roofline  — bottleneck: KV bytes (DRAM or L3 on CPU)

        compute_s_raw and memory_s_raw are the aggregate un-efficiencied values
        stored in StageCost for introspection and validate() calibration.
        base_s applies efficiency_compute / efficiency_memory.
        """
        B   = workload.batch_size
        ctx = workload.decode_context_len

        param_bytes  = sum(u.param_bytes for u in stage.units)
        kv_per_layer = sum(u.kv_bytes_per_token for u in stage.units)
        qr           = workload.kv_quant_ratio
        kv_read      = kv_per_layer * B * ctx * qr
        kv_write     = kv_per_layer * B * 1 * qr
        kv_bytes     = kv_read + kv_write

        proj_flops, attn_flops = self._flops_split(stage, workload, for_prefill=False)

        eff_c    = device.efficiency_compute   # validated > 0 in DeviceProfile.__post_init__
        eff_m    = device.efficiency_memory    # validated > 0 in DeviceProfile.__post_init__
        peak     = self._effective_decode_flops(device, workload.batch_size)
        bw       = device.mem_bandwidth
        eff_kv_bw = self._effective_kv_bw(device, kv_bytes)

        # ── Projection roofline ───────────────────────────────────────────────
        proj_compute_s = proj_flops / (peak * eff_c) if peak > 0 else 0.0
        proj_memory_s  = param_bytes / (bw  * eff_m) if bw  > 0 else 0.0
        proj_s = max(proj_compute_s, proj_memory_s)

        # ── Attention roofline ────────────────────────────────────────────────
        attn_compute_s = attn_flops / (peak     * eff_c) if peak      > 0 else 0.0
        attn_memory_s  = kv_bytes   / (eff_kv_bw * eff_m) if eff_kv_bw > 0 else 0.0
        attn_s = max(attn_compute_s, attn_memory_s)

        base_s = proj_s + attn_s

        # Raw aggregate values for StageCost introspection (no efficiency applied)
        # Use raw peak_flops_decode (batch-invariant) for introspection; base_s uses the scaled peak.
        compute_s_raw = (proj_flops + attn_flops) / device.peak_flops_decode if device.peak_flops_decode > 0 else 0.0
        memory_s_raw  = (param_bytes + kv_bytes)  / bw  if bw  > 0 else 0.0

        return compute_s_raw, memory_s_raw, base_s

    def _prefill_time(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        device: DeviceProfile,
    ) -> Tuple[float, float, float]:
        """
        Returns (compute_s_raw, memory_s_raw, base_s) for a prefill pass.

        Two separate rooflines:
          1. Projection roofline — weight bytes + input hidden state (CPU only)
          2. Attention roofline  — KV write (+ activation traffic for CPU)

        GPU attention uses flash-attention style fused kernels: Q/K/V tiles stay
        in SRAM, so there is no DRAM traffic for intermediate activations.
        CPU attention does read/write the hidden state between layers.
        """
        B = workload.batch_size
        S = workload.prefill_seq_len

        param_bytes = sum(u.param_bytes for u in stage.units)
        kv_write    = sum(u.kv_bytes_per_token for u in stage.units) * B * S * workload.kv_quant_ratio

        if device.device_type == "cuda":
            # Flash attention: activations stay in SRAM, no DRAM round-trips
            act_traffic = 0
        elif self.model_config is None:
            act_traffic = sum(
                max(u.activation_bytes, 0)
                for u in stage.units
                if u.unit_type == "transformer_block"
            )
        else:
            H   = self.model_config.hidden_size
            dsz = self.model_config.dtype_size
            n_blocks = sum(1 for u in stage.units if u.unit_type == "transformer_block")
            # read input + write output of the hidden state per layer;
            # 0 if the stage contains no transformer blocks (e.g. embedding-only stage)
            act_traffic = 2 * B * S * H * dsz * n_blocks

        proj_flops, attn_flops = self._flops_split(stage, workload, for_prefill=True)

        eff_c = device.efficiency_compute   # validated > 0 in DeviceProfile.__post_init__
        eff_m = device.efficiency_memory    # validated > 0 in DeviceProfile.__post_init__
        peak  = device.peak_flops_prefill
        bw    = device.mem_bandwidth

        # ── Projection roofline ───────────────────────────────────────────────
        proj_compute_s = proj_flops / (peak * eff_c) if peak > 0 else 0.0
        proj_memory_s  = param_bytes / (bw   * eff_m) if bw   > 0 else 0.0
        proj_s = max(proj_compute_s, proj_memory_s)

        # ── Attention roofline ────────────────────────────────────────────────
        attn_mem_bytes = kv_write + act_traffic
        attn_compute_s = attn_flops     / (peak * eff_c) if peak > 0 else 0.0
        attn_memory_s  = attn_mem_bytes / (bw   * eff_m) if bw   > 0 else 0.0
        attn_s = max(attn_compute_s, attn_memory_s)

        base_s = proj_s + attn_s

        # Raw aggregate values for StageCost introspection (no efficiency applied)
        total_bytes   = param_bytes + act_traffic + kv_write
        compute_s_raw = (proj_flops + attn_flops) / peak if peak > 0 else 0.0
        memory_s_raw  = total_bytes                / bw  if bw  > 0 else 0.0

        return compute_s_raw, memory_s_raw, base_s

    def _boundary_comm(
        self,
        stage: StageSpec,
        next_stage: Optional[StageSpec],
        workload: WorkloadSpec,
    ) -> Tuple[float, float]:
        """
        PCIe transfer time for the activation tensor at the stage boundary.

        The only tensor crossing devices is the hidden state:
          prefill: B × S × H × dtype_size
          decode:  B × 1 × H × dtype_size

        Raises
        ------
        ValueError
            If model_config is None and adjacent stages are on different devices.
            The hidden-state size cannot be inferred without it.
        KeyError
            If no LinkProfile exists for the (src, dst) device pair.
        """
        if next_stage is None or stage.device_id == next_stage.device_id:
            return 0.0, 0.0

        link_key = (stage.device_id, next_stage.device_id)
        if link_key not in self.links:
            raise KeyError(
                f"No LinkProfile for {stage.device_id!r} → {next_stage.device_id!r}. "
                f"Register this link in the 'links' dict passed to HelmCostModel. "
                f"Available keys: {sorted(self.links.keys())}"
            )

        link    = self.links[link_key]
        bw      = link.bandwidth_bytes_per_s
        latency = link.latency_s

        B   = workload.batch_size
        S   = workload.prefill_seq_len

        if self.model_config is None:
            pre_bytes = max((max(u.activation_bytes, 0) for u in stage.units), default=0)
            dec_bytes = (
                max(1, pre_bytes // max(S, 1))
                if pre_bytes > 0
                else 0
            )
        else:
            H   = self.model_config.hidden_size
            dsz = self.model_config.dtype_size
            pre_bytes = B * S * H * dsz
            dec_bytes = B * 1 * H * dsz

        return (latency + pre_bytes / bw, latency + dec_bytes / bw)
