#!/usr/bin/env python3
"""
pipeline_cli.py

A pipeline-style CLI with subcommands:
  1) prep  -> read pairs.json, copy .mat files into a stable workspace, write manifest.json
  2) train -> read config and run initial training (stub here; hook into your trainer)

Usage examples:
  python pipeline_cli.py prep --pairs_json pairs.json --out_dir ./workspace_root
  python pipeline_cli.py prep --pairs_json pairs.json --out_dir ./workspace_root --overwrite
  python pipeline_cli.py train --config configs/config.yaml --workspace_root ./workspace_root
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from data import read_json, write_json_atomic, build_workspace_from_pairs
 

# ----------------------------
# Commands
# ----------------------------

def cmd_prep(args: argparse.Namespace) -> int:
    """
    Prepare a reproducible workspace snapshot:
    - reads pairs spec JSON
    - validates + copies .mat files into out_dir/{train,val,test}/<id>/
    - writes out_dir/manifest.json
    """
    pairs_json_path = Path(args.pairs_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    # Load pairs spec
    pairs_data = read_json(pairs_json_path)
    if isinstance(pairs_data, dict) and "pairs" in pairs_data:
        pairs_data = pairs_data["pairs"]
    if not isinstance(pairs_data, list):
        raise ValueError("pairs.json must contain a top-level list of pair items.")

    # Build workspace + manifest
    manifest: Dict[str, Any] = build_workspace_from_pairs(
        pairs_spec=pairs_data,
        workspace_root=out_dir,
        overwrite=bool(args.overwrite),
        compute_hash=not bool(args.no_hash),
    )

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(f"[prep] Workspace created at: {out_dir.resolve()}")
    print(f"[prep] Manifest written to:  {manifest_path.resolve()}")
    print(f"[prep] Num pairs: {len(manifest.get('pairs', []))}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """
    Training entrypoint (stub):
    - loads config
    - points trainer to workspace_root (and/or manifest.json)
    """
    config_path = Path(args.config).expanduser()
    workspace_root = Path(args.workspace_root).expanduser()
    manifest_path = workspace_root / "manifest.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not workspace_root.exists():
        raise FileNotFoundError(f"Workspace root not found: {workspace_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found (run prep first): {manifest_path}")

    # cfg = load_config(config_path)  # <- plug in your config loader
    # train_from_workspace(cfg, workspace_root=workspace_root, manifest_path=manifest_path)

    print("[train] Stub: connect this to your training code.")
    print(f"[train] config:        {config_path.resolve()}")
    print(f"[train] workspace:     {workspace_root.resolve()}")
    print(f"[train] manifest.json: {manifest_path.resolve()}")
    return 0


# ----------------------------
# Argparse
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline_cli.py",
        description="Pipeline CLI: prep workspace + train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- prep ----
    p_prep = subparsers.add_parser(
        "prep",
        help="Prepare training workspace by copying .mat files and writing manifest.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_prep.add_argument("--pairs_json", required=True, help="Path to pairs spec JSON file")
    p_prep.add_argument("--out_dir", required=True, help="Workspace root output directory")
    p_prep.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in workspace",
    )
    p_prep.add_argument(
        "--no_hash",
        action="store_true",
        help="Skip SHA256 hashing (faster, less integrity checking)",
    )
    p_prep.set_defaults(func=cmd_prep)

    # ---- train ----
    p_train = subparsers.add_parser(
        "train",
        help="Run initial training using a config and a prepared workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_train.add_argument("--config", required=True, help="Path to training config (e.g., YAML)")
    p_train.add_argument(
        "--workspace_root",
        required=True,
        help="Workspace root produced by `prep` (contains manifest.json)",
    )
    p_train.set_defaults(func=cmd_train)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
