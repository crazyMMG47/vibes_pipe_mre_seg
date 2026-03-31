from __future__ import annotations
from typing import Any, Dict
from pathlib import Path

from torch.utils.data import DataLoader

from src.vibes_pipe.data.dataset import ManifestDataset
from src.vibes_pipe.data.transforms import Preprocessor
from src.vibes_pipe.augmentation.builders import build_train_augmenter


def build_data(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    preprocess_cfg = cfg.get("preprocess", {})
    augment_cfg = cfg.get("augmentation", {})
    loader_cfg = cfg.get("dataloader", {})

    manifest_path = data_cfg.get("manifest_path", "")
    if not manifest_path:
        raise ValueError("Missing cfg['data']['manifest_path'].")

    preprocessor = Preprocessor(preprocess_cfg)
    train_augmenter = build_train_augmenter(augment_cfg) if augment_cfg else None

    train_ds = ManifestDataset(
        manifest=Path(manifest_path),
        split="train",
        preprocessor=preprocessor,
        augmenter=train_augmenter,
    )

    val_ds = ManifestDataset(
        manifest=Path(manifest_path),
        split="val",
        preprocessor=preprocessor,
        augmenter=None,
    )

    test_ds = ManifestDataset(
        manifest=Path(manifest_path),
        split="test",
        preprocessor=preprocessor,
        augmenter=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=loader_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=loader_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", True),
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }