from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict

from vibes_pipe.data.pairs import create_and_save_pairs, preprocess_and_save_as_dict

def run_prepare_data(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Creates pairs_mapping.json and train/val/test dataset.pkl.
    Returns paths to produced artifacts.
    """
    # ---- read config ----
    io = cfg["io"]
    artifacts = cfg["artifacts"]
    prep = cfg["prepare"]

    out_root = Path(artifacts["root"]).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pairs_file = Path(artifacts.get("pairs_file", out_root / "pairs_mapping.json")).expanduser().resolve()
    preprocessed_dir = Path(artifacts.get("preprocessed_dir", out_root)).expanduser().resolve()

    force = bool(cfg.get("run", {}).get("force", False))

    # ---- stage 1: pairs mapping ----
    if force or not pairs_file.exists():
        create_and_save_pairs(
            t2_dir=io["t2_dir"],
            mask_dir=io["mask_dir"],
            train_txt=io["splits"]["train_txt"],
            val_txt=io["splits"]["val_txt"],
            test_txt=io["splits"]["test_txt"],
            noise_dir=io.get("noise_dir", None),
            output_file=str(pairs_file),
        )

    # ---- stage 2: preprocess ----
    # datasets are: preprocessed_dir/train|val|test/dataset.pkl
    train_pkl = preprocessed_dir / "train" / "dataset.pkl"
    val_pkl = preprocessed_dir / "val" / "dataset.pkl"
    test_pkl = preprocessed_dir / "test" / "dataset.pkl"

    if force or (not train_pkl.exists() or not val_pkl.exists() or not test_pkl.exists()):
        preprocess_and_save_as_dict(
            pairs_file=str(pairs_file),
            output_dir=str(preprocessed_dir),
            target_spacing=tuple(prep["target_spacing"]),
            target_size=tuple(prep["target_size"]),
            is_2d=bool(prep.get("is_2d", False)),
        )

    # ---- manifest ----
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "prepare_data",
        "config": cfg,
        "artifacts": {
            "pairs_file": str(pairs_file),
            "train_dataset": str(train_pkl),
            "val_dataset": str(val_pkl),
            "test_dataset": str(test_pkl),
        },
    }
    (out_root / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest["artifacts"]
