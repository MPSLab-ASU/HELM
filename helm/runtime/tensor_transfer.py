import torch
from typing import Any

def move_tensor(tensor: Any, device: str) -> Any:
    """
    Moves a tensor to the specified device if necessary.
    In the future, this layer will intercept async PCIe/NVLink transfers.
    """
    if isinstance(tensor, tuple):
        return tuple(move_tensor(t, device) for t in tensor)
    if isinstance(tensor, list):
        return [move_tensor(t, device) for t in tensor]
    if isinstance(tensor, dict):
        return {k: move_tensor(v, device) for k, v in tensor.items()}
        
    if not isinstance(tensor, torch.Tensor):
        return tensor

    return tensor.to(device)
