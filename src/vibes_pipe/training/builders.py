from __future__ import annotations
from typing import Any, Dict

# import all other builder functions
from src.vibes_pipe.models.builders import build_model
from src.vibes_pipe.losses.builders import build_loss
from src.vibes_pipe.optim.builders import build_optimizer
from src.vibes_pipe.training.engine import TrainEngine


def build_engine(cfg: Dict[str, Any]):
    engine_cfg = cfg.get("trainer", {})

    device = engine_cfg.get("device", "cpu")
    num_epochs = engine_cfg.get("num_epochs", 100)
    grad_clip = engine_cfg.get("grad_clip", 1.0)
    fast_val = engine_cfg.get("fast_val", True)
    log_every = engine_cfg.get("log_every", 1)
    save_path = engine_cfg.get("save_path", "best_model.pt")
    history_save_path = engine_cfg.get("history_save_path", "train_history.json")

    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)

    engine = TrainEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        grad_clip=grad_clip,
        fast_val=fast_val,
        log_every=log_every,
        save_path=save_path,
        history_save_path=history_save_path,
    )

    return engine