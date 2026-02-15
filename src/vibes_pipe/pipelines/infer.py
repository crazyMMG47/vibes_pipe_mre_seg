from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def run_infer(cfg: Dict[str, Any]) -> None:
    """
    Orchestrates inference end-to-end.
    Contract: writes manifest.json and other artifacts to cfg['io']['out_dir'].
    """
    io = cfg.get("io", {})
    out_dir = Path(io.get("out_dir", "runs/infer")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: wire to your existing functions.
    # Example sketch:
    # device = resolve_device(cfg.get("run", {}).get("device", "auto"))
    # model = load_model(io["checkpoint"], device=device, cfg=cfg)
    # preds = run_inference(model, inputs=..., cfg=cfg)
    # export_predictions(preds, out_dir, cfg)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "config": cfg,
        "artifacts": {
            "manifest": str(out_dir / "manifest.json"),
            "predictions_mat": str(out_dir / "predictions.mat"),
            "summary_csv": str(out_dir / "summary.csv"),
            "figures_dir": str(out_dir / "figures"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
