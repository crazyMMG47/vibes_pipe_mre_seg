from __future__ import annotations
from typing import Any, Dict

from src.vibes_pipe.losses.recon_combo_loss import FocalDiceComboLoss, ProbUNetLoss


def build_loss(cfg: Dict[str, Any]):
    loss_cfg = cfg.get("loss", {})
    class_name = loss_cfg.get("class_name", "")
    kwargs = loss_cfg.get("kwargs", {})

    if not class_name:
        raise ValueError("Missing cfg['loss']['class_name'].")

    if class_name == "ProbUNetLoss":
        recon_cfg = kwargs.pop("recon_loss", {})
        recon_name = recon_cfg.get("class_name", "FocalDiceComboLoss")
        recon_kwargs = recon_cfg.get("kwargs", {})

        if recon_name == "FocalDiceComboLoss":
            recon_loss = FocalDiceComboLoss(**recon_kwargs)
        else:
            raise ValueError(f"Unknown recon loss: {recon_name}")

        return ProbUNetLoss(recon_loss=recon_loss, **kwargs)

    raise ValueError(f"Unknown loss class: {class_name}")