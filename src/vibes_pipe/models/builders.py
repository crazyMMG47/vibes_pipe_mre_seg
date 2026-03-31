# Build models from config
# create model from the YAML config

from __future__ import annotations
from typing import Any, Dict

from src.vibes_pipe.models.prob_unet import *


def build_model(cfg: Dict[str, Any]):
    """
    Dispatcher to choose which class, and forwards all params. 
    
    The class decides what to do with them.
    """
    model_cfg = cfg.get("model", {})
    class_name = model_cfg.get("class_name")
    model_kwargs = model_cfg.get("kwargs", {})

    if not class_name:
        raise ValueError("Missing cfg['model']['class_name'].")

    # creates the model class instance 
    cls = globals().get(class_name)
    if cls is None:
        raise ValueError(f"Unknown model class: {class_name}")

    # builder passes kwargs into model's instances initiated above
    return cls(**model_kwargs)