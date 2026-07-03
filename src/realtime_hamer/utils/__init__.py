from typing import Any

import torch


def recursive_to(x: Any, target: torch.device):
    """Recursively move tensors in a nested structure to ``target``."""
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(target)
    if isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    return x
