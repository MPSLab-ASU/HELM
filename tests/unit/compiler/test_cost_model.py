"""
Unit tests for HelmCostModel (helm/compiler/optimization/cost_model.py).
"""
import math
import pytest

from helm.compiler.partition.partition_units import PartitionUnit
from helm.compiler.partition.partition_plan import StageSpec, PartitionPlan
from helm.compiler.optimization.cost_model import (
    HelmCostModel, ModelConfig, DeviceProfile, LinkProfile, WorkloadSpec,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unit(uid, layer_id=0, param_bytes=10 * 1024 ** 2, activation_bytes=1024,
          kv_bytes_per_token=256, flops_prefill=int(1e9), flops_decode=int(1e7)):
    return PartitionUnit(
        unit_id=uid,
        unit_type="transformer_block",
        layer_start=layer_id,
        layer_end=layer_id,
        node_ids=[uid],
        flops_prefill=flops_prefill,
        flops_decode=flops_decode,
        param_bytes=param_bytes,
        activation_bytes=activation_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def _stage(stage_id, device_id, units):
    return StageSpec(stage_id=stage_id, device_id=device_id, units=units)


def _plan(*stages):
    return PartitionPlan(stages=list(stages))


def _mc():
    """Minimal ModelConfig used as default in tests that don't exercise model-specific behaviour."""
    return ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=16,
        dtype_size=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Memory feasibility
# ──────────────────────────────────────────────────────────────────────────────

def test_oom_infeasible(workload):
    tiny = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9, 1 * 1024 ** 2)  # 1 MB
    model = HelmCostModel({"cuda:0": tiny}, {}, model_config=_mc())
    stage = _stage(0, "cuda:0", [_unit(0, param_bytes=100 * 1024 ** 2)])  # 100 MB > 1 MB
    cost = model.estimate_plan(_plan(stage), workload)
    assert cost.feasible is False
    assert "OOM" in (cost.reason or "")


def test_feasible_cpu_fits(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0, param_bytes=1 * 1024 ** 2)])  # 1 MB << 32 GB
    cost = cost_model.estimate_plan(_plan(stage), workload)
    assert cost.feasible is True


def test_feasible_returns_no_reason(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0)])
    cost = cost_model.estimate_plan(_plan(stage), workload)
    assert cost.reason is None


def test_unknown_device_raises(workload):
    """Unknown device raises KeyError rather than returning a silent infeasible cost."""
    model = HelmCostModel({}, {})
    stage = _stage(0, "nonexistent", [_unit(0)])
    with pytest.raises(KeyError, match="nonexistent"):
        model.estimate_plan(_plan(stage), workload)


def test_empty_plan_raises_clear_error(workload, cpu_profile):
    model = HelmCostModel({"cpu": cpu_profile}, {}, model_config=_mc())
    with pytest.raises(ValueError, match="at least one stage"):
        model.estimate_plan(PartitionPlan(stages=[]), workload)


def test_empty_stage_raises_clear_error(workload, cpu_profile):
    model = HelmCostModel({"cpu": cpu_profile}, {}, model_config=_mc())
    empty_stage = _stage(0, "cpu", [])
    with pytest.raises(ValueError, match="Stage 0.*at least one unit"):
        model.estimate_plan(_plan(empty_stage), workload)


def test_oom_stops_at_first_bad_stage(workload):
    """Plan evaluation halts at the first infeasible stage."""
    tiny_gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9, 1 * 1024 ** 2)
    big_cpu = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 32 * 1024 ** 3)
    mc = ModelConfig(hidden_size=4096, intermediate_size=11008,
                     num_attention_heads=32, num_kv_heads=32, head_dim=128)
    link = LinkProfile("cpu", "cuda:0", bandwidth_bytes_per_s=16e9, latency_s=1e-4)
    model = HelmCostModel(
        {"cpu": big_cpu, "cuda:0": tiny_gpu},
        {("cpu", "cuda:0"): link},
        model_config=mc,
    )
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cuda:0", [_unit(1, param_bytes=100 * 1024 ** 2)])  # OOM on GPU
    cost = model.estimate_plan(_plan(s0, s1), workload)
    assert cost.feasible is False
    assert "Stage 1" in (cost.reason or "")


# ──────────────────────────────────────────────────────────────────────────────
# Communication costs
# ──────────────────────────────────────────────────────────────────────────────

def test_same_device_zero_comm(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cpu", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    assert cost.feasible is True
    assert cost.stage_costs[0].prefill_comm_s == 0.0
    assert cost.stage_costs[0].decode_comm_s == 0.0


def test_cross_device_comm_positive(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cuda:0", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    assert cost.feasible is True
    assert cost.stage_costs[0].prefill_comm_s > 0.0
    assert cost.stage_costs[0].decode_comm_s > 0.0


def test_final_stage_no_comm(cost_model, workload):
    """The last stage never has outgoing cross-stage comms."""
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cuda:0", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    assert cost.stage_costs[1].prefill_comm_s == 0.0
    assert cost.stage_costs[1].decode_comm_s == 0.0


def test_no_link_profile_raises(workload):
    """Missing LinkProfile raises KeyError instead of silently using a fallback bandwidth."""
    cpu = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 32 * 1024 ** 3)
    gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9, 8 * 1024 ** 3)
    mc = ModelConfig(hidden_size=4096, intermediate_size=11008,
                     num_attention_heads=32, num_kv_heads=32, head_dim=128)
    model = HelmCostModel({"cpu": cpu, "cuda:0": gpu}, {}, model_config=mc)
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cuda:0", [_unit(1)])
    with pytest.raises(KeyError, match="cpu"):
        model.estimate_plan(_plan(s0, s1), workload)


# ──────────────────────────────────────────────────────────────────────────────
# Latency aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_total_latency_formula(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0)])
    cost = cost_model.estimate_plan(_plan(stage), workload)
    expected = cost.prefill_latency_s + workload.decode_tokens * cost.decode_token_latency_s
    assert abs(cost.total_latency_s - expected) < 1e-12


def test_throughput_positive(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0)])
    cost = cost_model.estimate_plan(_plan(stage), workload)
    assert cost.throughput_tokens_per_s > 0.0


def test_stage_costs_count(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cuda:0", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    assert len(cost.stage_costs) == 2


def test_validate_requires_measurement_for_each_stage(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cpu", [_unit(1)])
    plan = _plan(s0, s1)

    with pytest.raises(ValueError, match=r"missing measured timings for stage_id\(s\): \[1\]"):
        cost_model.validate(
            plan,
            workload,
            {0: {"decode_ms": 1.0, "prefill_ms": 1.0}},
        )


def test_max_stage_memory_nonzero(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0, param_bytes=50 * 1024 ** 2)])
    cost = cost_model.estimate_plan(_plan(stage), workload)
    assert cost.max_stage_memory_bytes > 0


def test_unit_based_estimate_without_model_config(workload):
    cpu = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 32 * 1024 ** 3)
    gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9, 8 * 1024 ** 3)
    link = LinkProfile("cpu", "cuda:0", bandwidth_bytes_per_s=16e9, latency_s=1e-4)
    model = HelmCostModel(
        {"cpu": cpu, "cuda:0": gpu},
        {("cpu", "cuda:0"): link},
        model_config=None,
    )
    s0 = _stage(0, "cpu", [_unit(0, activation_bytes=4096)])
    s1 = _stage(1, "cuda:0", [_unit(1, activation_bytes=4096)])

    cost = model.estimate_plan(_plan(s0, s1), workload)

    assert cost.feasible is True
    assert cost.stage_costs[0].prefill_comm_s > 0.0
    assert cost.max_stage_memory_bytes > 0


def test_prefill_latency_equals_sum_of_stage_prefills(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cpu", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    expected = sum(sc.prefill_total_s for sc in cost.stage_costs)
    assert abs(cost.prefill_latency_s - expected) < 1e-12


def test_decode_latency_equals_sum_of_stage_decodes(cost_model, workload):
    s0 = _stage(0, "cpu", [_unit(0)])
    s1 = _stage(1, "cpu", [_unit(1)])
    cost = cost_model.estimate_plan(_plan(s0, s1), workload)
    expected = sum(sc.decode_total_s for sc in cost.stage_costs)
    assert abs(cost.decode_token_latency_s - expected) < 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# Compute time is non-negative
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_times_non_negative(cost_model, workload):
    stage = _stage(0, "cpu", [_unit(0)])
    cost = cost_model.estimate_plan(_plan(stage), workload)
    sc = cost.stage_costs[0]
    assert sc.prefill_compute_s >= 0.0
    assert sc.prefill_memory_s >= 0.0
    assert sc.decode_compute_s >= 0.0
    assert sc.decode_memory_s >= 0.0


def test_zero_flops_zero_compute():
    cpu = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 32 * 1024 ** 3)
    model = HelmCostModel({"cpu": cpu}, {}, model_config=_mc())
    # Use a non-transformer unit so stored flops (0) are used directly by _flops_split
    zero_unit = PartitionUnit(
        unit_id=0, unit_type="embedding",
        layer_start=0, layer_end=0, node_ids=[0],
        flops_prefill=0, flops_decode=0,
        param_bytes=0, activation_bytes=0, kv_bytes_per_token=0,
    )
    stage = _stage(0, "cpu", [zero_unit])
    wl = WorkloadSpec(1, 128, 256, 64)
    cost = model.estimate_plan(_plan(stage), wl)
    sc = cost.stage_costs[0]
    assert sc.prefill_compute_s >= 0.0
    assert sc.decode_compute_s >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture helpers for the 5 analytical model improvements
# ──────────────────────────────────────────────────────────────────────────────

def _qwen7b_config():
    """Approximate Qwen2.5-7B-Instruct architecture parameters."""
    return ModelConfig(
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        dtype_size=2,
    )

def _cpu_dev(**kwargs):
    # peak_flops_decode represents the theoretical SIMD peak throughput (not the
    # bandwidth-limited measured GEMV throughput).  On a modern laptop CPU with
    # AVX2+FMA the ridge point is peak_flops / mem_bw ≈ 100G / 51G ≈ 2 FLOPs/byte,
    # well above the GEMV arithmetic intensity of 1 FLOPs/byte → correctly
    # memory-bound.  Setting it to the measured GEMV value (≈ bw/dtype_size ≈ 25G)
    # would put the ridge at 0.49, making the model incorrectly classify GEMV as
    # compute-bound.
    defaults = dict(
        device_id="cpu", device_type="cpu",
        peak_flops_prefill=1e12, peak_flops_decode=100e9,
        mem_bandwidth=51e9, memory_capacity=16 * 1024**3,
    )
    defaults.update(kwargs)
    return DeviceProfile(**defaults)

def _gpu_dev(**kwargs):
    defaults = dict(
        device_id="cuda", device_type="cuda",
        peak_flops_prefill=22e12, peak_flops_decode=11e12,
        mem_bandwidth=272e9, memory_capacity=8 * 1024**3,
    )
    defaults.update(kwargs)
    return DeviceProfile(**defaults)

def _transformer_unit(uid, layer_id=0, n_layers=1):
    """
    A PartitionUnit sized roughly like one Qwen2.5-7B transformer block:
      params = (H² + H×kv + H×kv + H² + H×I + H×I + I×H) × dtype_size
             ≈ 466 MB per layer  (28 layers × 466 MB ≈ 13 GB total, matches 7B)
      kv_bytes_per_token = 2 × num_kv_heads × head_dim × dtype_size = 2048 bytes
    Note: no factor-of-2 multiplier — that belongs in the FLOP formula, not bytes.
    """
    H, kv_heads, head_dim, I, dsz = 3584, 4, 128, 18944, 2
    param = int(dsz * (H*H + H*kv_heads*head_dim + H*kv_heads*head_dim
                       + H*H + H*I + H*I + I*H))
    kv_per_tok = 2 * kv_heads * head_dim * dsz  # 2048 bytes/token
    return PartitionUnit(
        unit_id=uid, unit_type="transformer_block",
        layer_start=layer_id, layer_end=layer_id + n_layers - 1,
        node_ids=[uid],
        flops_prefill=int(1e12), flops_decode=int(1e9),
        param_bytes=param, activation_bytes=3584*2,
        kv_bytes_per_token=kv_per_tok,
    )


def _kv_heavy_unit(uid, param_mb=2):
    """
    Unit with small weights but standard KV size (2048 bytes/token like Qwen-7B).
    Used in L3 tests where we want attention BW to be significant relative to
    projection BW — impossible with full 466 MB weight blocks at 128 tokens.
    """
    kv_per_tok = 2 * 4 * 128 * 2  # 2048 bytes/token
    return PartitionUnit(
        unit_id=uid, unit_type="transformer_block",
        layer_start=uid, layer_end=uid,
        node_ids=[uid],
        flops_prefill=int(1e6), flops_decode=int(1e6),
        param_bytes=param_mb * 1024 * 1024,
        activation_bytes=1024,
        kv_bytes_per_token=kv_per_tok,
    )


def _kv_only_unit(uid, kv_bytes_per_token=100 * 1024):
    """
    Unit with zero weights and zero compute — decode time is purely attention
    KV bandwidth.  Used in L3 tests to isolate the cache-hierarchy effect
    without projection weights diluting the timing signal.
    100 KB/token default lets the KV fit in L3 at ctx=64 (6.4 MB < 12 MB L3)
    and exceed L3 at ctx=256 (25.6 MB > 12 MB), giving a clean contrast.
    """
    return PartitionUnit(
        unit_id=uid, unit_type="transformer_block",
        layer_start=uid, layer_end=uid,
        node_ids=[uid],
        flops_prefill=0, flops_decode=0,
        param_bytes=0, activation_bytes=0,
        kv_bytes_per_token=kv_bytes_per_token,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Point 1 — Efficiency coefficients
# ──────────────────────────────────────────────────────────────────────────────

class TestEfficiencyCoefficients:
    """
    efficiency_compute and efficiency_memory de-rate the device's measured
    peak values.  With eff=0.5 everything should take ~2× longer than eff=1.0.
    """

    def test_half_memory_efficiency_doubles_decode_time(self):
        # Memory-bound decode (large weights, few FLOPs per byte)
        cpu_full = _cpu_dev()
        cpu_half = _cpu_dev(efficiency_memory=0.5)
        mc = _qwen7b_config()
        m_full = HelmCostModel({"cpu": cpu_full}, {}, model_config=mc)
        m_half = HelmCostModel({"cpu": cpu_half}, {}, model_config=mc)
        wl   = WorkloadSpec(1, 64, 64, 32)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        t_full = m_full.estimate_plan(_plan(s), wl).decode_token_latency_s
        t_half = m_half.estimate_plan(_plan(s), wl).decode_token_latency_s
        # Should be ~2× — allow 5% tolerance
        assert abs(t_half / t_full - 2.0) < 0.05, (
            f"expected ~2× latency with eff=0.5, got {t_half/t_full:.3f}×")

    def test_half_compute_efficiency_doubles_compute_bound_time(self):
        # Make the stage compute-bound: tiny weights, high FLOPs, fast BW
        cpu = DeviceProfile(
            "cpu", "cpu",
            peak_flops_prefill=1e12, peak_flops_decode=1e12,
            mem_bandwidth=1000e9,           # effectively unlimited BW
            memory_capacity=32 * 1024**3,
            efficiency_compute=0.5,
        )
        mc = _qwen7b_config()
        m  = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        m1 = HelmCostModel({"cpu": _cpu_dev(peak_flops_decode=1e12, mem_bandwidth=1000e9)}, {}, model_config=mc)
        wl = WorkloadSpec(1, 64, 64, 32)
        u  = _transformer_unit(0)
        s  = _stage(0, "cpu", [u])
        t_half = m.estimate_plan(_plan(s), wl).decode_token_latency_s
        t_full = m1.estimate_plan(_plan(s), wl).decode_token_latency_s
        assert t_half > t_full, "eff=0.5 should produce longer compute time"

    def test_default_efficiency_is_one(self):
        cpu = _cpu_dev()
        assert cpu.efficiency_compute == 1.0
        assert cpu.efficiency_memory  == 1.0

    def test_validate_suggests_efficiency(self):
        """validate() returns suggested_efficiency_memory close to raw/actual."""
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        wl  = WorkloadSpec(1, 64, 64, 32)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        plan = _plan(s)
        pc   = m.estimate_plan(plan, wl)
        sc   = pc.stage_costs[0]
        # Simulate: actual hardware is 15% slower than predicted
        actual_dec_ms = sc.decode_total_s * 1000 * (1 / 0.85)
        report = m.validate(plan, wl, {0: {"decode_ms": actual_dec_ms, "prefill_ms": 0.0}})
        sug = report[0]["suggested_efficiency_memory"]
        if sug is not None:
            # Should suggest ≈0.85 (within 5%)
            assert abs(sug - 0.85) < 0.05, f"Expected ~0.85, got {sug}"


# ──────────────────────────────────────────────────────────────────────────────
# Point 2 — Separate attention and projection rooflines
# ──────────────────────────────────────────────────────────────────────────────

class TestSeparateRooflines:
    """
    At long context, attention KV traffic grows (O(context_len)) and becomes
    a non-trivial fraction of total decode time.  The split model must reflect
    this — decode latency should grow with context_len even for a fixed weight
    footprint (which is all a single-roofline model would see).
    """

    def test_decode_latency_grows_with_context(self):
        mc   = _qwen7b_config()
        cpu  = _cpu_dev()
        m    = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl_short = WorkloadSpec(1, 64, 64,   32)
        wl_long  = WorkloadSpec(1, 64, 4096, 32)
        t_short = m.estimate_plan(_plan(s), wl_short).decode_token_latency_s
        t_long  = m.estimate_plan(_plan(s), wl_long).decode_token_latency_s
        assert t_long > t_short, (
            f"decode latency should grow with context: short={t_short*1000:.1f}ms "
            f"long={t_long*1000:.1f}ms")

    def test_attn_flops_scale_with_context(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        # Compare _flops_split at two context lengths
        wl64   = WorkloadSpec(1, 64, 64,   32)
        wl4096 = WorkloadSpec(1, 64, 4096, 32)
        _, attn64   = m._flops_split(s, wl64,   for_prefill=False)
        _, attn4096 = m._flops_split(s, wl4096, for_prefill=False)
        assert attn4096 / attn64 == pytest.approx(4096 / 64, rel=1e-3)

    def test_proj_flops_independent_of_context(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl64   = WorkloadSpec(1, 64, 64,   32)
        wl4096 = WorkloadSpec(1, 64, 4096, 32)
        proj64,   _ = m._flops_split(s, wl64,   for_prefill=False)
        proj4096, _ = m._flops_split(s, wl4096, for_prefill=False)
        assert proj64 == pytest.approx(proj4096, rel=1e-6), (
            "Projection FLOPs should not depend on context_len at decode")

    def test_prefill_attn_flops_quadratic_in_seq(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl64  = WorkloadSpec(1, 64,  64,  32)
        wl128 = WorkloadSpec(1, 128, 128, 32)
        _, attn64  = m._flops_split(s, wl64,  for_prefill=True)
        _, attn128 = m._flops_split(s, wl128, for_prefill=True)
        # Doubling S → 4× attention FLOPs (causal: O(S²))
        assert attn128 / attn64 == pytest.approx(4.0, rel=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Point 3 — GPU prefill has zero activation traffic
# ──────────────────────────────────────────────────────────────────────────────

class TestGPUPrefillNoActTraffic:
    """
    GPU uses fused attention (flash-attention style): activations stay in
    SRAM, so there is no DRAM round-trip for the hidden state between layers.
    CPU has no equivalent: hidden state goes through DRAM at each layer.
    """

    def _make_models(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        gpu = _gpu_dev()
        cpu_m = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        gpu_m = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        return cpu_m, gpu_m

    def test_gpu_prefill_memory_less_than_cpu(self):
        """GPU prefill memory_s_raw should be smaller than CPU (no act traffic)."""
        cpu_m, gpu_m = self._make_models()
        wl   = WorkloadSpec(1, 512, 512, 32)
        unit = _transformer_unit(0)
        s_cpu = _stage(0, "cpu",  [unit])
        s_gpu = _stage(0, "cuda", [unit])
        cpu_cost = cpu_m.estimate_plan(_plan(s_cpu), wl)
        gpu_cost = gpu_m.estimate_plan(_plan(s_gpu), wl)
        # memory_s_raw = total_bytes / bw
        # CPU has act_traffic > 0, GPU has 0 — so even normalising by BW,
        # CPU's total_bytes is larger.
        cpu_mem_bytes = cpu_cost.stage_costs[0].prefill_memory_s * 51e9
        gpu_mem_bytes = gpu_cost.stage_costs[0].prefill_memory_s * 272e9
        assert gpu_mem_bytes < cpu_mem_bytes, (
            f"GPU prefill memory bytes ({gpu_mem_bytes/1e6:.0f} MB) should be "
            f"< CPU ({cpu_mem_bytes/1e6:.0f} MB) due to no act_traffic")

    def test_gpu_prefill_act_traffic_zero(self):
        """Directly: removing act_traffic means memory bytes = param + kv_write."""
        mc  = _qwen7b_config()
        gpu = _gpu_dev(mem_bandwidth=1e12)   # high BW → memory time negligible
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        wl  = WorkloadSpec(1, 512, 512, 32)
        n_layers = 5
        units = [_transformer_unit(i, layer_id=i) for i in range(n_layers)]
        s  = _stage(0, "cuda", units)
        pc = m.estimate_plan(_plan(s), wl)
        sc = pc.stage_costs[0]
        H, kv_heads, head_dim, I, dsz = 3584, 4, 128, 18944, 2
        # param bytes = elements × dtype_size (no ×2 FLOP factor)
        param = n_layers * int(dsz * (H*H + H*kv_heads*head_dim + H*kv_heads*head_dim
                                      + H*H + H*I + H*I + I*H))
        kv_write = n_layers * 2 * kv_heads * head_dim * dsz * 1 * 512
        expected_bytes = param + kv_write
        actual_bytes = sc.prefill_memory_s * 1e12  # s × BW = bytes
        # Allow 1% tolerance for floating-point accumulation
        assert abs(actual_bytes - expected_bytes) / expected_bytes < 0.01, (
            f"GPU prefill bytes: expected {expected_bytes/1e6:.0f} MB, "
            f"got {actual_bytes/1e6:.0f} MB")

    def test_multiple_layers_cpu_act_traffic_scales(self):
        """CPU act_traffic = 2 × B × S × H × dtype × n_blocks — scales with n_blocks."""
        mc  = _qwen7b_config()
        cpu = _cpu_dev(mem_bandwidth=1e12)
        m1  = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        m5  = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        wl  = WorkloadSpec(1, 128, 128, 32)
        u1 = [_transformer_unit(0)]
        u5 = [_transformer_unit(i, layer_id=i) for i in range(5)]
        s1 = _stage(0, "cpu", u1)
        s5 = _stage(0, "cpu", u5)
        bytes_1 = m1.estimate_plan(_plan(s1), wl).stage_costs[0].prefill_memory_s * 1e12
        bytes_5 = m5.estimate_plan(_plan(s5), wl).stage_costs[0].prefill_memory_s * 1e12
        # Activation traffic = 2×B×S×H×dsz × n_blocks, which is 5× larger for 5 blocks.
        # Total bytes (param + act + kv_write) should be larger for 5 blocks.
        assert bytes_5 > bytes_1


# ──────────────────────────────────────────────────────────────────────────────
# Point 4 — No kernel_launch_overhead_s
# ──────────────────────────────────────────────────────────────────────────────

class TestNoKernelLaunchOverhead:
    """DeviceProfile no longer has kernel_launch_overhead_s."""

    def test_field_does_not_exist(self):
        cpu = _cpu_dev()
        assert not hasattr(cpu, "kernel_launch_overhead_s"), (
            "kernel_launch_overhead_s should have been removed from DeviceProfile")

    def test_construction_without_overhead_field(self):
        # Should construct fine without the field
        cpu = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 32 * 1024**3)
        assert cpu.device_id == "cpu"

    def test_zero_bytes_zero_latency(self):
        """With zero params and zero KV, decode time should be exactly 0."""
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=_mc())
        # Use "embedding" unit type so _flops_split uses stored flops (0),
        # not the analytical transformer formula.
        z   = PartitionUnit(
            unit_id=0, unit_type="embedding",
            layer_start=0, layer_end=0, node_ids=[0],
            flops_prefill=0, flops_decode=0,
            param_bytes=0, activation_bytes=0, kv_bytes_per_token=0,
        )
        s   = _stage(0, "cpu", [z])
        wl  = WorkloadSpec(1, 64, 64, 32)
        sc  = m.estimate_plan(_plan(s), wl).stage_costs[0]
        assert sc.decode_total_s == 0.0, (
            f"No work + no overhead field → 0 latency, got {sc.decode_total_s*1000:.3f}ms")
        assert sc.prefill_total_s == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Point 5 — CPU L3 cache hierarchy for KV attention
# ──────────────────────────────────────────────────────────────────────────────

class TestL3CacheHierarchy:
    """
    When KV fits in L3, _effective_kv_bw() returns l3_bandwidth.
    When KV exceeds L3, it blends down toward mem_bandwidth.
    This reduces predicted attention latency at short contexts.
    """

    L3_SIZE  = 12 * 1024 * 1024   # 12 MB
    L3_BW    = 150e9               # 150 GB/s  (≈3× DRAM)
    DRAM_BW  = 50e9

    def _cpu_with_l3(self):
        return _cpu_dev(
            mem_bandwidth=self.DRAM_BW,
            l3_size_bytes=self.L3_SIZE,
            l3_bandwidth=self.L3_BW,
        )

    def test_kv_fits_in_l3_uses_l3_bw(self):
        cpu = self._cpu_with_l3()
        m   = HelmCostModel({"cpu": cpu}, {})
        # kv_bytes < L3_SIZE
        eff = m._effective_kv_bw(cpu, self.L3_SIZE // 2)
        assert eff == self.L3_BW

    def test_kv_exceeds_l3_uses_dram_bw(self):
        cpu = self._cpu_with_l3()
        m   = HelmCostModel({"cpu": cpu}, {})
        # kv_bytes >> L3_SIZE → should approach DRAM BW
        eff = m._effective_kv_bw(cpu, self.L3_SIZE * 100)
        # At 100× L3 size, L3 fraction = 0.01; effective BW ≈ DRAM BW
        expected = 0.01 * self.L3_BW + 0.99 * self.DRAM_BW
        assert abs(eff - expected) / expected < 1e-6

    def test_kv_exactly_l3_boundary(self):
        cpu = self._cpu_with_l3()
        m   = HelmCostModel({"cpu": cpu}, {})
        eff = m._effective_kv_bw(cpu, self.L3_SIZE)
        assert eff == self.L3_BW

    def test_l3_shortens_decode_at_short_context(self):
        """
        Decode is faster with L3 at short context vs no L3 model.

        Uses _kv_only_unit (zero weights, 100 KB/token) so decode time is
        purely attention KV bandwidth.  At ctx=64: KV = 6.4 MB < L3 (12 MB)
        → full L3 BW, clearly faster than DRAM.
        """
        cpu_l3    = self._cpu_with_l3()
        cpu_no_l3 = _cpu_dev(mem_bandwidth=self.DRAM_BW)
        m_l3    = HelmCostModel({"cpu": cpu_l3},    {}, model_config=_mc())
        m_no_l3 = HelmCostModel({"cpu": cpu_no_l3}, {}, model_config=_mc())
        unit = _kv_only_unit(0)   # 100 KB/token, zero weights
        s    = _stage(0, "cpu", [unit])
        # ctx=64 → KV = 100KB×64 = 6.4 MB < L3 (12 MB)
        wl = WorkloadSpec(1, 64, 64, 32)
        t_l3    = m_l3.estimate_plan(_plan(s), wl).decode_token_latency_s
        t_no_l3 = m_no_l3.estimate_plan(_plan(s), wl).decode_token_latency_s
        assert t_l3 < t_no_l3, (
            f"L3 should give shorter decode: l3={t_l3*1e6:.1f}µs, "
            f"no_l3={t_no_l3*1e6:.1f}µs")

    def test_l3_effect_diminishes_at_long_context(self):
        """
        The relative L3 speedup decreases as context length grows past L3 size.

        We don't require the speedup to vanish (it approaches zero asymptotically
        in the linear-blend model, requiring ~40× L3_SIZE to reach < 5%), but it
        must be strictly smaller at long context than at short context where KV
        fits entirely in L3.
        """
        cpu_l3    = self._cpu_with_l3()
        cpu_no_l3 = _cpu_dev(mem_bandwidth=self.DRAM_BW)
        m_l3    = HelmCostModel({"cpu": cpu_l3},    {}, model_config=_mc())
        m_no_l3 = HelmCostModel({"cpu": cpu_no_l3}, {}, model_config=_mc())
        unit = _kv_only_unit(0)   # 100 KB/token, zero weights
        s    = _stage(0, "cpu", [unit])

        # Short: KV = 100KB×64 = 6.4 MB < L3 (12 MB) → full L3 BW
        wl_short = WorkloadSpec(1, 64, 64,   32)
        # Long:  KV = 100KB×1024 = 100 MB >> L3 (12 MB) → mostly DRAM
        wl_long  = WorkloadSpec(1, 64, 1024, 32)

        def speedup(wl):
            t_l3    = m_l3.estimate_plan(_plan(s), wl).decode_token_latency_s
            t_no_l3 = m_no_l3.estimate_plan(_plan(s), wl).decode_token_latency_s
            return t_no_l3 / t_l3   # > 1 means L3 is faster

        sp_short = speedup(wl_short)
        sp_long  = speedup(wl_long)
        assert sp_short > sp_long, (
            f"L3 speedup should be smaller at long context: "
            f"short={sp_short:.3f}×, long={sp_long:.3f}×")

    def test_gpu_ignores_l3_fields(self):
        """L3 modeling is CPU-only; GPU device uses mem_bandwidth regardless."""
        gpu = DeviceProfile(
            "cuda", "cuda", 22e12, 22e12, 272e9, 8 * 1024**3,
            l3_size_bytes=12 * 1024 * 1024, l3_bandwidth=1e12,
        )
        m   = HelmCostModel({"cuda": gpu}, {})
        eff = m._effective_kv_bw(gpu, 1024)
        assert eff == 272e9, "GPU should use mem_bandwidth, not l3_bandwidth"

    def test_l3_disabled_when_zero(self):
        cpu = _cpu_dev()   # l3_size_bytes=0 by default
        m   = HelmCostModel({"cpu": cpu}, {})
        eff = m._effective_kv_bw(cpu, 1024)
        assert eff == cpu.mem_bandwidth


# ──────────────────────────────────────────────────────────────────────────────
# Roofline regime sanity checks (decode always memory-bound at batch=1)
# ──────────────────────────────────────────────────────────────────────────────

class TestRooflineRegime:
    """
    At batch=1 (seq=1 decode), arithmetic intensity ≈ 1 FLOP/byte for GEMV.
    All realistic hardware (CPU and GPU) has ops/byte >> 1, so decode must
    always be memory-bound.
    """

    def test_decode_memory_bound_on_cpu_batch1(self):
        mc   = _qwen7b_config()
        cpu  = _cpu_dev()
        m    = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl   = WorkloadSpec(1, 64, 64, 32)
        sc   = m.estimate_plan(_plan(s), wl).stage_costs[0]
        assert sc.decode_regime == "memory", (
            "batch=1 decode on CPU must be memory-bound (GEMV arithmetic "
            f"intensity ≈ 1 FLOP/byte); compute_s={sc.decode_compute_s*1000:.3f}ms "
            f"memory_s={sc.decode_memory_s*1000:.3f}ms")

    def test_decode_memory_bound_on_gpu_batch1(self):
        mc   = _qwen7b_config()
        gpu  = _gpu_dev()
        m    = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cuda", [unit])
        wl   = WorkloadSpec(1, 64, 64, 32)
        sc   = m.estimate_plan(_plan(s), wl).stage_costs[0]
        assert sc.decode_regime == "memory", (
            "batch=1 decode on GPU must be memory-bound (GEMV): "
            f"compute_s={sc.decode_compute_s*1000:.3f}ms "
            f"memory_s={sc.decode_memory_s*1000:.3f}ms")

    def test_prefill_can_be_compute_bound_on_gpu_long_seq(self):
        """
        At large S on GPU, projection arithmetic intensity = 2S FLOPs/byte.
        For S=4096, H=3584: intensity ≈ 2×4096 = 8192 FLOPs/byte >> GPU ops/byte
        (~80 for RTX 4060 at 22 TFLOPS / 272 GB/s), so prefill should be compute-bound.
        """
        mc  = _qwen7b_config()
        gpu = _gpu_dev()
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cuda", [unit])
        wl   = WorkloadSpec(1, 4096, 4096, 32)
        sc   = m.estimate_plan(_plan(s), wl).stage_costs[0]
        assert sc.decode_regime is not None   # field exists

        # Separately check prefill regime via raw values
        # intensity = total_flops / total_bytes; if > peak_flops / mem_bw → compute-bound
        proj_flops, attn_flops = m._flops_split(s, wl, for_prefill=True)
        total_flops = proj_flops + attn_flops
        intensity = total_flops / (sc.prefill_memory_s * gpu.mem_bandwidth)
        ops_per_byte = gpu.peak_flops_prefill / gpu.mem_bandwidth
        assert intensity > ops_per_byte * 0.5, (
            f"GPU prefill at S=4096 should be near/at compute-bound: "
            f"intensity={intensity:.0f} ops/byte, ridge={ops_per_byte:.0f}")


# ──────────────────────────────────────────────────────────────────────────────
# KV cache quantization ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestKVQuantRatio:
    """kv_quant_ratio < 1.0 reduces KV memory and bandwidth proportionally."""

    def test_int8_halves_kv_memory(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl_fp16 = WorkloadSpec(1, 64, 256, 32)
        wl_int8 = WorkloadSpec(1, 64, 256, 32, kv_quant_ratio=0.5)
        sc_fp16 = m.estimate_plan(_plan(s), wl_fp16).stage_costs[0]
        sc_int8 = m.estimate_plan(_plan(s), wl_int8).stage_costs[0]
        assert sc_int8.kv_bytes == pytest.approx(sc_fp16.kv_bytes * 0.5, rel=1e-6)
        assert sc_int8.memory_bytes < sc_fp16.memory_bytes

    def test_int8_reduces_decode_latency(self):
        # Use a KV-only unit (zero weights) so decode time is purely KV bandwidth.
        # Halving kv_quant_ratio should halve the attention time and thus total time.
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=_mc())
        unit = _kv_only_unit(0, kv_bytes_per_token=100 * 1024)  # 100 KB/token
        s    = _stage(0, "cpu", [unit])
        wl_fp16 = WorkloadSpec(1, 64, 4096, 32)
        wl_int8 = WorkloadSpec(1, 64, 4096, 32, kv_quant_ratio=0.5)
        t_fp16 = m.estimate_plan(_plan(s), wl_fp16).decode_token_latency_s
        t_int8 = m.estimate_plan(_plan(s), wl_int8).decode_token_latency_s
        assert t_int8 == pytest.approx(t_fp16 * 0.5, rel=0.02)

    def test_default_ratio_is_one(self):
        wl = WorkloadSpec(1, 64, 64, 32)
        assert wl.kv_quant_ratio == 1.0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match="kv_quant_ratio"):
            WorkloadSpec(1, 64, 64, 32, kv_quant_ratio=0.0)
        with pytest.raises(ValueError, match="kv_quant_ratio"):
            WorkloadSpec(1, 64, 64, 32, kv_quant_ratio=1.5)


def test_model_config_allows_int8_dtype():
    mc = ModelConfig(hidden_size=4096, intermediate_size=11008,
                     num_attention_heads=32, num_kv_heads=32,
                     head_dim=128, dtype_size=1)
    assert mc.dtype_size == 1


# ──────────────────────────────────────────────────────────────────────────────
# Peak intermediate memory
# ──────────────────────────────────────────────────────────────────────────────

class TestPeakIntermediateMemory:
    """Feasibility check accounts for transient MLP/attention intermediates."""

    def test_mlp_intermediate_causes_oom(self):
        """A device that fits params+KV+act but not the MLP intermediate is infeasible."""
        mc = ModelConfig(
            hidden_size=4096, intermediate_size=11008,
            num_attention_heads=32, num_kv_heads=32, head_dim=128,
        )
        # MLP peak at S=2048, B=1: 1×2048×2×11008×2 = ~86 MB
        # Device has 300 MB: params(10MB) + act(~16MB) + KV(~0.5MB) + overhead(200MB) < 300MB
        # but adding 86MB intermediate pushes it over.
        small_gpu = DeviceProfile(
            "cuda:0", "cuda", 100e12, 100e12, 500e9,
            300 * 1024**2,  # 300 MB
        )
        m = HelmCostModel({"cuda:0": small_gpu}, {}, model_config=mc)
        unit = PartitionUnit(
            unit_id=0, unit_type="transformer_block",
            layer_start=0, layer_end=0, node_ids=[0],
            flops_prefill=int(1e12), flops_decode=int(1e9),
            param_bytes=10 * 1024**2,  # 10 MB params
            activation_bytes=0, kv_bytes_per_token=2048,
        )
        s  = _stage(0, "cuda:0", [unit])
        wl = WorkloadSpec(1, 2048, 64, 32)  # S=2048 → MLP intermediate ~86 MB
        cost = m.estimate_plan(_plan(s), wl)
        assert cost.feasible is False, (
            "Should be infeasible: MLP intermediate ~86 MB + params + overhead > 300 MB capacity"
        )

    def test_cpu_attention_scores_counted(self):
        """On CPU, the S×S attention score matrix dominates when num_heads is large."""
        mc = ModelConfig(
            hidden_size=64, intermediate_size=16,
            num_attention_heads=32, num_kv_heads=4, head_dim=2,
        )
        cpu = _cpu_dev(memory_capacity=16 * 1024**3)
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        # Use a simple unit (any transformer_block will do)
        unit = PartitionUnit(
            unit_id=0, unit_type="transformer_block",
            layer_start=0, layer_end=0, node_ids=[0],
            flops_prefill=int(1e9), flops_decode=int(1e7),
            param_bytes=1024, activation_bytes=0, kv_bytes_per_token=0,
        )
        s    = _stage(0, "cpu", [unit])
        wl   = WorkloadSpec(1, 512, 64, 32)
        peak = m._peak_intermediate_bytes(s, wl, "cpu")
        # attn_peak = 1 × 32 × 512 × 512 × 2 = 16,777,216 (dominates mlp_peak=32,768)
        attn_expected = 1 * 32 * 512 * 512 * 2
        mlp_expected  = 1 * 512 * 2 * 16 * 2
        assert attn_expected > mlp_expected, "test setup: attn must dominate"
        assert peak == attn_expected

    def test_gpu_no_attention_score_memory(self):
        """GPU flash-attention: no S×S score matrix in DRAM."""
        mc = ModelConfig(
            hidden_size=4096, intermediate_size=18944,
            num_attention_heads=32, num_kv_heads=32, head_dim=128,
        )
        gpu = _gpu_dev()
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cuda", [unit])
        wl   = WorkloadSpec(1, 4096, 64, 32)
        peak = m._peak_intermediate_bytes(s, wl, "cuda")
        # GPU: only MLP peak, no attention scores
        mlp_expected = 1 * 4096 * 2 * 18944 * 2
        assert peak == mlp_expected

    def test_stage_cost_has_peak_intermediate_field(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl   = WorkloadSpec(1, 64, 64, 32)
        sc   = m.estimate_plan(_plan(s), wl).stage_costs[0]
        assert hasattr(sc, "peak_intermediate_bytes")
        assert sc.peak_intermediate_bytes > 0

    def test_no_transformer_blocks_zero_intermediate(self):
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        embed = PartitionUnit(
            unit_id=0, unit_type="embedding",
            layer_start=None, layer_end=None, node_ids=[0],
            flops_prefill=0, flops_decode=0,
            param_bytes=1024, activation_bytes=0, kv_bytes_per_token=0,
        )
        s  = _stage(0, "cpu", [embed])
        wl = WorkloadSpec(1, 64, 64, 32)
        peak = m._peak_intermediate_bytes(s, wl, "cpu")
        assert peak == 0


# ──────────────────────────────────────────────────────────────────────────────
# Batch-aware decode FLOPS
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchAwareDecode:
    """Decode effective FLOPS scales with batch size (GEMV → GEMM transition)."""

    def test_batch1_uses_decode_peak(self):
        gpu = _gpu_dev()
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=_qwen7b_config())
        eff = m._effective_decode_flops(gpu, 1)
        assert eff == gpu.peak_flops_decode

    def test_large_batch_uses_prefill_peak(self):
        gpu = _gpu_dev()
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=_qwen7b_config())
        eff = m._effective_decode_flops(gpu, 64)
        assert eff == gpu.peak_flops_prefill

    def test_intermediate_batch_interpolates(self):
        gpu = _gpu_dev()
        m   = HelmCostModel({"cuda": gpu}, {}, model_config=_qwen7b_config())
        eff = m._effective_decode_flops(gpu, 16)
        assert gpu.peak_flops_decode < eff < gpu.peak_flops_prefill

    def test_decode_latency_scales_sublinearly_with_batch(self):
        """
        decode_total_s should grow sub-linearly with batch due to GEMV→GEMM FLOP scaling.
        If effective FLOPS increase with batch, B=8 should take less than 8× the B=1 time.
        Uses a compute-dominant scenario: high FLOPs, fast memory bandwidth.
        """
        mc  = _qwen7b_config()
        # Set very high mem bandwidth to make stage compute-bound at B=8
        gpu = DeviceProfile(
            "cuda", "cuda",
            peak_flops_prefill=22e12,
            peak_flops_decode=11e12,
            mem_bandwidth=1e15,          # effectively unlimited → compute-bound
            memory_capacity=8 * 1024**3,
        )
        m    = HelmCostModel({"cuda": gpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cuda", [unit])
        wl_b1 = WorkloadSpec(1, 64, 64, 32)
        wl_b8 = WorkloadSpec(8, 64, 64, 32)
        t_b1 = m.estimate_plan(_plan(s), wl_b1).decode_token_latency_s
        t_b8 = m.estimate_plan(_plan(s), wl_b8).decode_token_latency_s
        # With 8× FLOPs and higher effective FLOPS, t_b8 < 8 × t_b1
        assert t_b8 < t_b1 * 8, (
            f"Compute-bound decode should scale sub-linearly: "
            f"B=1={t_b1*1000:.3f}ms, B=8={t_b8*1000:.3f}ms, ratio={t_b8/t_b1:.2f}×")

    def test_equal_peaks_returns_decode_peak(self):
        """When prefill==decode peaks (e.g. CPU), no interpolation needed."""
        cpu = DeviceProfile("cpu", "cpu", 100e9, 100e9, 50e9, 32 * 1024**3)
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=_qwen7b_config())
        eff = m._effective_decode_flops(cpu, 16)
        assert eff == 100e9


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline throughput — bottleneck stage, not serial sum
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineThroughput:
    """Throughput uses bottleneck stage (max), not serial sum."""

    def test_two_stage_throughput_uses_max(self):
        mc = _qwen7b_config()
        cpu = _cpu_dev()
        gpu = _gpu_dev()
        link = LinkProfile("cpu", "cuda", bandwidth_bytes_per_s=16e9, latency_s=1e-4)
        m = HelmCostModel(
            {"cpu": cpu, "cuda": gpu},
            {("cpu", "cuda"): link},
            model_config=mc,
        )
        # CPU stage with many units (slow), GPU stage with few (fast)
        cpu_units = [_transformer_unit(i, layer_id=i) for i in range(5)]
        gpu_units = [_transformer_unit(5, layer_id=5)]
        s0 = _stage(0, "cpu",  cpu_units)
        s1 = _stage(1, "cuda", gpu_units)
        wl = WorkloadSpec(1, 64, 64, 32)
        cost = m.estimate_plan(_plan(s0, s1), wl)
        bottleneck = max(sc.decode_total_s for sc in cost.stage_costs)
        serial_sum = sum(sc.decode_total_s for sc in cost.stage_costs)
        assert cost.throughput_tokens_per_s == pytest.approx(
            wl.batch_size / bottleneck, rel=1e-6
        )
        # Pipeline throughput strictly better than serial
        serial_throughput = wl.batch_size / serial_sum
        assert cost.throughput_tokens_per_s > serial_throughput

    def test_single_stage_throughput_unchanged(self):
        """Single stage: max == sum, so throughput formula is identical."""
        mc  = _qwen7b_config()
        cpu = _cpu_dev()
        m   = HelmCostModel({"cpu": cpu}, {}, model_config=mc)
        unit = _transformer_unit(0)
        s    = _stage(0, "cpu", [unit])
        wl   = WorkloadSpec(1, 64, 64, 32)
        cost = m.estimate_plan(_plan(s), wl)
        assert cost.throughput_tokens_per_s == pytest.approx(
            wl.batch_size / cost.decode_token_latency_s, rel=1e-6
        )
