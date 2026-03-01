"""
Bridge between manifest.json and the model training loop. 
The ManifestDataset can:
- read manifest.json
- filter samples by split 
- load file paths 
- call Preprocessor.process_pair 
- return tensors (ready for training)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import Preprocessor

JsonDict = Dict[str, Any]


@dataclass
class Sample:
    """One resolved sample from the manifest."""
    pair_id: str
    split: str
    x_path: Path
    gt_path: Path
    nli_path: Optional[Path] = None
    eligible_preds_path: Optional[Path] = None
    meta: Optional[JsonDict] = None


def load_manifest(manifest_path: str | Path) -> Tuple[Path, List[JsonDict]]:
    p = Path(manifest_path).expanduser().resolve()
    d = json.loads(p.read_text())
    ws_root = Path(d["workspace_root"]).expanduser().resolve()
    pairs = d["pairs"]
    return ws_root, pairs


def _resolve_samples(ws_root: Path, pairs: List[JsonDict], split: str) -> List[Sample]:
    out: List[Sample] = []
    for pair in pairs:
        if pair["split"] != split:
            continue

        files = pair["files"]
        x = ws_root / files["X"]["dst"]
        gt = ws_root / files["GT"]["dst"]

        nli = None
        if files.get("NLI_output") is not None:
            nli = ws_root / files["NLI_output"]["dst"]

        ep = None
        if files.get("eligible_preds") is not None:
            ep = ws_root / files["eligible_preds"]["dst"]

        out.append(
            Sample(
                pair_id=pair["id"],
                split=pair["split"],
                x_path=x,
                gt_path=gt,
                nli_path=nli,
                eligible_preds_path=ep,
                meta=pair.get("meta", {}),
            )
        )
    return out


class ManifestDataset(Dataset):
    """
    Manifest-backed dataset:
      - uses manifest.json to locate files
      - runs deterministic preprocessing (resize/normalize/etc.)
      - optionally applies a transform/augmentation callable
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        preprocessor: Preprocessor,
        transform: Optional[Callable[[np.ndarray, np.ndarray, JsonDict], Tuple[np.ndarray, np.ndarray, JsonDict]]] = None,
        return_paths: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.preprocessor = preprocessor
        self.transform = transform
        self.return_paths = return_paths

        ws_root, pairs = load_manifest(self.manifest_path)
        self.ws_root = ws_root
        self.samples = _resolve_samples(ws_root, pairs, split)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split={split!r} in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        # deterministic steps: CLAHE/resample/resize/normalize + label binarize
        img, lbl, orig_meta = self.preprocessor.process_pair(str(s.x_path), str(s.gt_path))

        meta: JsonDict = {}
        meta.update(s.meta or {})
        meta["orig_meta"] = orig_meta
        meta["id"] = s.pair_id
        meta["split"] = s.split

        # optional random augmentation (train only)
        if self.transform is not None:
            img, lbl, meta = self.transform(img, lbl, meta)

        # to torch tensors (C, H, W, D) or (1, H, W, D)
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        lbl_t = torch.from_numpy(lbl).float().unsqueeze(0)

        out: Dict[str, Any] = {"image": img_t, "label": lbl_t, "meta": meta}
        if self.return_paths:

            out["paths"] = {"X": str(s.x_path), "GT": str(s.gt_path)}
        return out
