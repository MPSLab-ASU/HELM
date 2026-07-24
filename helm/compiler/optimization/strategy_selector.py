# helm/compiler/optimization/strategy_selector.py

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

from helm.compiler.partition.partition_plan import PartitionPlan, StageSpec
from helm.compiler.partition.partition_units import PartitionUnit
from helm.compiler.optimization.cost_model import HelmCostModel, PlanCost, WorkloadSpec

_VALID_OBJECTIVES = ("decode_latency", "prefill_latency", "total_latency", "throughput")


@dataclass
class StrategySelectorConfig:
    objective: str = "decode_latency"   # "prefill_latency" | "total_latency" | "throughput"
    max_stages: int = 2                 # only 1 and 2 are currently implemented
    allow_cpu: bool = True
    allow_multi_gpu: bool = False
    kv_offload: bool = False  # when True, KV cache lives in CPU RAM — excluded from GPU memory budget
    kv_reserve_tokens: int = 0  # reserve GPU headroom for this many KV tokens when kv_offload=True

    def __post_init__(self) -> None:
        if self.objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"StrategySelectorConfig.objective must be one of {_VALID_OBJECTIVES}, "
                f"got {self.objective!r}"
            )
        if self.max_stages > 2:
            logger.warning(
                "StrategySelectorConfig.max_stages=%d is set but only 1- and 2-stage plans "
                "are currently implemented; plans with more than 2 stages will not be explored.",
                self.max_stages,
            )
        if self.kv_offload and self.kv_reserve_tokens <= 0:
            raise ValueError(
                "StrategySelectorConfig.kv_reserve_tokens must be > 0 when kv_offload=True. "
                "Set it to the number of GPU-resident KV tokens kept before eviction "
                "(e.g. prefill_seq_len for a single-request warm cache)."
            )


class StrategySelector:
    def __init__(
        self,
        cost_model: HelmCostModel,
        devices: List[str],
        config: StrategySelectorConfig,
    ):
        self.cost_model = cost_model
        self.devices = devices
        self.config = config

    def select(
        self,
        units: List[PartitionUnit],
        workload: WorkloadSpec,
    ) -> Tuple[PartitionPlan, PlanCost]:
        """
        Find the partition that minimises the objective function.

        Evaluates ALL feasible plans rather than returning the first feasible
        one.  The greedy "first feasible = most GPU = best latency" assumption
        does not hold in general: the cost model may find that an intermediate
        split (more CPU layers than minimum) is cheaper because the GPU stage
        becomes memory-bound before the CPU stage, making the boundary comm cost
        dominate at certain split points.

        Candidate plans evaluated in order of decreasing prior probability:
          1. All-GPU (1 stage) — best case, evaluated first
          2. All CPU→GPU 2-stage splits (k=1..N-1) — every feasible split scored
          3. All-CPU (1 stage) — last resort

        The winning plan minimises `_score(plan, cost)` for the configured
        objective.  All feasible candidates are retained; infeasible candidates
        are excluded because their `PlanCost.reason` already records the exact
        memory constraint that failed.
        """
        if not units:
            raise ValueError("StrategySelector.select requires at least one partition unit")

        cuda_devices = [d for d in self.devices if "cuda" in d]
        if len(cuda_devices) > 1:
            if self.config.allow_multi_gpu:
                logger.warning(
                    "StrategySelectorConfig.allow_multi_gpu=True, but multi-GPU partition "
                    "candidates are not implemented yet; using %s only.",
                    cuda_devices[0],
                )
            cuda_devices = cuda_devices[:1]
        cpu_devices  = [d for d in self.devices if "cpu"  in d]

        candidates: List[tuple] = []   # (plan, PlanCost)

        # --- 1-stage all-GPU ---
        if cuda_devices:
            plan = self._build_one_stage_plan(units, cuda_devices[0])
            cost = self.cost_model.estimate_plan(plan, workload,
                                                  kv_offload=self.config.kv_offload,
                                                  kv_reserve_tokens=self.config.kv_reserve_tokens)
            if cost.feasible:
                candidates.append((plan, cost))

        # --- 2-stage CPU→GPU: evaluate every split ---
        if (self.config.max_stages >= 2
                and self.config.allow_cpu
                and cpu_devices and cuda_devices
                and len(units) > 1):
            for k in range(1, len(units)):
                plan = self._build_two_stage_plan(
                    units, k, cpu_devices[0], cuda_devices[0])
                cost = self.cost_model.estimate_plan(plan, workload,
                                                      kv_offload=self.config.kv_offload,
                                                      kv_reserve_tokens=self.config.kv_reserve_tokens)
                if cost.feasible:
                    candidates.append((plan, cost))

        # --- 1-stage all-CPU ---
        if self.config.allow_cpu and cpu_devices:
            plan = self._build_one_stage_plan(units, cpu_devices[0])
            cost = self.cost_model.estimate_plan(plan, workload,
                                                  kv_offload=self.config.kv_offload,
                                                  kv_reserve_tokens=self.config.kv_reserve_tokens)
            if cost.feasible:
                candidates.append((plan, cost))

        if not candidates:
            raise RuntimeError("No feasible partition plan found")

        best_plan, best_cost = min(candidates, key=lambda pc: self._score(pc[1]))
        self._log_selection(candidates, best_plan, best_cost)
        return best_plan, best_cost

    def _score(self, cost) -> float:
        """
        Scalar objective to minimise.  Lower is better.

        decode_latency  — minimise per-token decode time (seconds)
        prefill_latency — minimise time-to-first-token (seconds)
        total_latency   — minimise total wall time for the workload
        throughput      — maximise tokens/s (negate for min)
        """
        obj = self.config.objective
        if obj == "decode_latency":
            return cost.decode_token_latency_s
        if obj == "prefill_latency":
            return cost.prefill_latency_s
        if obj == "total_latency":
            return cost.total_latency_s
        if obj == "throughput":
            tps = cost.throughput_tokens_per_s
            return -tps if tps > 0 else float("inf")
        # Should never reach here — __post_init__ validates objective at construction.
        raise RuntimeError(
            f"unsupported StrategySelector objective {obj!r}; "
            f"expected one of {_VALID_OBJECTIVES}"
        )

    def _log_selection(self, candidates, best_plan, best_cost):
        """Log a compact summary of all evaluated plans."""
        obj = self.config.objective
        logger.info("[HELM] StrategySelector objective='%s' — %d feasible plan(s):", obj, len(candidates))
        for plan, cost in candidates:
            marker = " <- best" if plan is best_plan else ""
            stage_descs = ", ".join(
                f"stage{s.stage_id}@{s.device_id}({len(s.units)}u)" for s in plan.stages
            )
            decode_ms = cost.decode_token_latency_s * 1000
            prefill_ms = cost.prefill_latency_s * 1000
            mem_mb = cost.max_stage_memory_bytes / 1e6
            logger.info("  [%s] decode=%.1fms prefill=%.0fms mem=%.0fMB%s",
                        stage_descs, decode_ms, prefill_ms, mem_mb, marker)

    def _build_one_stage_plan(self, units: List[PartitionUnit], device: str) -> PartitionPlan:
        layer_start = next((u.layer_start for u in units if u.layer_start is not None), 0)
        layer_end = next((u.layer_end for u in reversed(units) if u.layer_end is not None), -1)

        stage = StageSpec(
            stage_id=0,
            device_id=device,
            units=units,
            layer_start=layer_start,
            layer_end=layer_end
        )
        return PartitionPlan(stages=[stage])

    def _build_two_stage_plan(self, units: List[PartitionUnit], split_idx: int, dev0: str, dev1: str) -> PartitionPlan:
        units0 = units[:split_idx]
        layer_start0 = next((u.layer_start for u in units0 if u.layer_start is not None), 0)
        layer_end0 = next((u.layer_end for u in reversed(units0) if u.layer_end is not None), -1)

        stage0 = StageSpec(
            stage_id=0,
            device_id=dev0,
            units=units0,
            layer_start=layer_start0,
            layer_end=layer_end0
        )

        units1 = units[split_idx:]
        layer_start1 = next((u.layer_start for u in units1 if u.layer_start is not None), 0)
        layer_end1 = next((u.layer_end for u in reversed(units1) if u.layer_end is not None), -1)

        stage1 = StageSpec(
            stage_id=1,
            device_id=dev1,
            units=units1,
            layer_start=layer_start1,
            layer_end=layer_end1
        )
        return PartitionPlan(stages=[stage0, stage1])
