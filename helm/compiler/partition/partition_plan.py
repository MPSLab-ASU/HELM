from dataclasses import dataclass, field
from typing import List
from .partition_units import PartitionUnit

@dataclass
class StageSpec:
    stage_id: int
    device_id: str
    units: List[PartitionUnit] = field(default_factory=list)
    layer_start: int = 0
    layer_end: int = 0

@dataclass
class PartitionPlan:
    stages: List[StageSpec] = field(default_factory=list)

@dataclass
class ExecutionSchedule:
    mode: str
    num_stages: int
    microbatches: int

