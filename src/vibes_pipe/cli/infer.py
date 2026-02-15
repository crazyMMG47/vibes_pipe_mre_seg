"""
Inference Pipeline CLI Module

This script serves as the entry point for running the inference pipeline. It implements 
a flexible configuration system that allows for:

1. Base Configuration: Loading default settings from a YAML file via --config.
2. Dynamic Overrides: Modifying any nested setting using dot-notation (e.g., --set model.device=cuda) 
   without changing the original YAML file.
3. Dry-Run Mode: Validating the final resolved configuration by printing it to the terminal 
   without executing the actual inference logic.

Typical usage:
    python -m vibes_pipe.cli.infer --config configs/infer.yaml --set io.batch_size=32
    
    Multiple overrides:
    python -m vibes_pipe.cli.infer \
    --config configs/base.yaml \
    --set model.batch_size=128 \
    --set model.device=cuda:0
    
More detailed explaination of the configuration system is provided in the code comments.
-m: module-run, runs the entire "vibes_pipe" as a package. 
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from vibes_pipe.pipelines.infer import run_infer  # you'll create this (or adapt)


# ---------------------------
# helpers: config + overrides
# ---------------------------

# 1. Load YAML default config
def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping/dict, got: {type(cfg)}")
    return cfg


def _parse_value(raw: str) -> Any:
    # lets you pass numbers/bools/lists/dicts via --set key=value
    # e.g. --set infer.window_size=[96,96,48]
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur: Dict[str, Any] = d
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set must be key=value, got: {item}")
        k, v = item.split("=", 1)
        _deep_set(cfg, k.strip(), _parse_value(v.strip()))
    return cfg


# ---------------------------
# cli
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="package-infer", description="Run inference pipeline")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config: dot.path=value (repeatable). Example: --set run.device=cuda",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print final resolved config, but do not run inference.",
    )
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    cfg = _apply_overrides(cfg, args.set)

    # optional: store config path (useful in manifest)
    cfg.setdefault("run", {})
    cfg["run"]["config_path"] = str(cfg_path)

    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    # sanity: ensure output dir exists if present
    out_dir = cfg.get("io", {}).get("out_dir")
    if out_dir:
        Path(out_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    run_infer(cfg)


if __name__ == "__main__":
    main()
