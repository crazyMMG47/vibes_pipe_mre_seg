"""
Data loader to prepare train, val, test data loaders. Make the data ready for the next step (training). 
We are going to call this func in the train_driver. 
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from torch.utils.data import DataLoader, Dataset

import torch

from .dataset import ManifestDataset, manifest_collate
from .transforms import Preprocessor

def build_loaders(
    manifest_path: str | Path,
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:

    mpath = Path(manifest_path).expanduser().resolve()

    # -------------------------
    # Read config sections 
    # ------------------------
    # This step is essential 
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    dl_cfg = cfg.get("dataloader", {}) if isinstance(cfg, dict) else {}

    label_mode = dataset_cfg.get("label_mode", "gt")
    pseudo_dir = dataset_cfg.get("pseudo_dir", None)

    batch_size = int(dl_cfg.get("batch_size", 1))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", True))

    pp = Preprocessor(cfg=cfg)

    def _make(split: str, shuffle: bool) -> Optional[DataLoader]:
        ds = ManifestDataset(
            manifest=mpath,
            split=split,
            preprocessor=pp,
            label_mode=label_mode,
            pseudo_dir=pseudo_dir,
            return_dict=True,
        )
        if len(ds) == 0:
            return None

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=manifest_collate
        )

    train_loader = _make("train", shuffle=True)
    val_loader = _make("val", shuffle=False)
    test_loader = _make("test", shuffle=False)

    if train_loader is None:
        raise ValueError("No training samples found for split='train'.")

    return train_loader, val_loader, test_loader
