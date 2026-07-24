import torch
from dataclasses import dataclass

@dataclass
class Stage:
    stage_id: int
    device: str
    module: torch.fx.GraphModule
