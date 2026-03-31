from __future__ import annotations
from typing import Any, Dict
import torch.optim as optim


def build_optimizer(cfg: Dict[str, Any], model):
    optim_cfg = cfg.get("optimizer", {})
    class_name = optim_cfg.get("class_name", "")
    optim_kwargs = optim_cfg.get("kwargs", {})

    if not class_name:
        raise ValueError("Missing cfg['optimizer']['class_name'].")

    # Get optimizer class from torch.optim
    cls = getattr(optim, class_name, None)
    if cls is None:
        raise ValueError(f"Unknown optimizer class: {class_name}")

    # Instantiate optimizer with model parameters
    optimizer = cls(model.parameters(), **optim_kwargs)

    return optimizer