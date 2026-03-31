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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
import torch 

import numpy as np
from torch.utils.data import Dataset

from .io_mat import load_mat_dict, find_primary_array
from .transforms import Preprocessor 

JsonDict = Dict[str, Any]
LabelMode = Literal["gt", "pseudo", "prefer_pseudo"]


def _abs_from_manifest(workspace_root: Path, dst_rel: str) -> Path:
    return (workspace_root / Path(dst_rel)).resolve()


def _load_mat_array(mat_path: Path) -> np.ndarray:
    md = load_mat_dict(mat_path)
    arr = find_primary_array(md, mat_path=str(mat_path))
    return arr

def manifest_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([torch.from_numpy(b["image"]).float() for b in batch], 0)
    labels = torch.stack([torch.from_numpy(b["label"]).float() for b in batch], 0)

    out = {
        "image": images,
        "label": labels,
        "id": [b["id"] for b in batch],
        "split": [b["split"] for b in batch],
        "scanner_type": [b.get("scanner_type") for b in batch],
        "meta": [b["meta"] for b in batch],
        "paths": [b["paths"] for b in batch],
    }

    if all("noise" in b for b in batch):
        out["noise"] = torch.stack(
            [torch.from_numpy(b["noise"]).float() for b in batch], 0
        )

    return out

@dataclass
class SamplePaths:
    id: str
    split: str
    x_mat: Path
    gt_mat: Path
    x_nii: Optional[Path] = None
    nli_mat: Optional[Path] = None
    eligible_preds_mat: Optional[Path] = None
    # future: pseudo label path (may not exist at start)
    pseudo_mat: Optional[Path] = None
    noise_mat: Optional [Path] = None
    scanner_type: Optional[str] = None
    noise_mat: Optional[Path] = None


class ManifestDataset(Dataset):
    """
    Manifest-backed dataset.

    Goals:
      - Train now with GT masks
      - Later: support pseudo-label loop without breaking the interface
      - Keep NLI outputs available for scoring / filtering / curriculum

    Returns dict samples by default (MONAI-friendly).
    """

    def __init__(
        self,
        manifest: JsonDict | str | Path,
        *,
        split: str,
        workspace_root: str | Path | None = None,
        preprocessor: Optional[Preprocessor] = None,
        label_mode: LabelMode = "gt",
        pseudo_dir: str | Path | None = None,
        pseudo_suffix: str = "pseudo.mat",
        return_dict: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        augmenter: Optional[Callable[..., tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = None,
    ) -> None:
        """
        Args:
            manifest: dict or path to manifest.json
            split: train/val/test
            workspace_root: override if needed; otherwise read from manifest["workspace_root"]
            preprocessor: your numpy preprocessor (optional). If None, returns raw arrays.
            label_mode:
                - "gt": always use GT.mat
                - "pseudo": require pseudo mask exists
                - "prefer_pseudo": use pseudo if exists else GT
            pseudo_dir:
                Directory to look for pseudo masks (workspace-relative or absolute).
                Default behavior if provided: <pseudo_dir>/<split>/<id>/<pseudo_suffix>
            return_dict: if True returns dict with keys "image","label",... (MONAI-friendly)
            transform: optional callable applied after preprocessing (can be MONAI Compose later)
        """
        self.split = split
        self.preprocessor = preprocessor
        self.label_mode = label_mode
        self.return_dict = return_dict
        self.transform = transform
        self.augmenter = augmenter
        
        if isinstance(manifest, (str, Path)):
            import json

            mpath = Path(manifest).expanduser().resolve()
            self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        else:
            self.manifest = manifest

        ws_root = Path(workspace_root).expanduser().resolve() if workspace_root else Path(self.manifest["workspace_root"]).resolve()
        self.workspace_root = ws_root

        self.pseudo_dir = None if pseudo_dir is None else Path(pseudo_dir).expanduser().resolve()
        self.pseudo_suffix = pseudo_suffix

        self.samples: List[SamplePaths] = self._index_manifest()

    def _index_manifest(self) -> List[SamplePaths]:
        pairs = self.manifest.get("pairs", [])
        out: List[SamplePaths] = []

        for p in pairs:
            
            scanner_type = p.get("scanner_type")
            if scanner_type is not None:
                scanner_type = str(scanner_type).upper().strip()
                
            if p.get("split") != self.split:
                continue
            files = p.get("files", {})
            pid = str(p.get("id"))

            x_mat = _abs_from_manifest(self.workspace_root, files["t2stack"]["dst"])
            gt_mat = _abs_from_manifest(self.workspace_root, files["GT(human)"]["dst"])

            x_nii = None
            if files.get("t2stack_nii") is not None:
                x_nii = _abs_from_manifest(self.workspace_root, files["t2stack_nii"]["dst"])

            nli = None
            if files.get("NLI_output") is not None:
                nli = _abs_from_manifest(self.workspace_root, files["NLI_output"]["dst"])

            ep = None
            if files.get("eligible_preds") is not None:
                ep = _abs_from_manifest(self.workspace_root, files["eligible_preds"]["dst"])

            pseudo = None
            if self.pseudo_dir is not None:
                # default pseudo path convention
                pseudo = (self.pseudo_dir / self.split / pid / self.pseudo_suffix).resolve()
                if not pseudo.exists():
                    pseudo = None
            
            noise = None
            if files.get("subject_noise") is not None:
                noise = _abs_from_manifest(self.workspace_root, files["subject_noise"]["dst"])
                
            out.append(
                SamplePaths(
                    id=pid,
                    split=self.split,
                    x_mat=x_mat,
                    gt_mat=gt_mat,
                    x_nii=x_nii,
                    nli_mat=nli,
                    eligible_preds_mat=ep,
                    pseudo_mat=pseudo,
                    scanner_type=scanner_type,
                    noise_mat=noise
                )
            )
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _select_label_path(self, s: SamplePaths) -> Path:
        if self.label_mode == "gt":
            return s.gt_mat
        if self.label_mode == "pseudo":
            if s.pseudo_mat is None:
                raise FileNotFoundError(f"label_mode='pseudo' but pseudo mask missing for id={s.id}")
            return s.pseudo_mat
        if self.label_mode == "prefer_pseudo":
            return s.pseudo_mat if s.pseudo_mat is not None else s.gt_mat
        raise ValueError(f"Unknown label_mode: {self.label_mode}")


    def __getitem__(self, idx: int) -> Dict[str, Any] | Tuple[np.ndarray, np.ndarray]:
        s = self.samples[idx]
        label_path = self._select_label_path(s)

        # -------- load image + label --------
        if self.preprocessor is not None:
            image, label, meta = self.preprocessor.process_pair(
                str(s.x_mat),
                str(label_path),
                image_nii_path=str(s.x_nii) if s.x_nii else None,
            )
        else:
            image = _load_mat_array(s.x_mat).astype(np.float32)
            label = _load_mat_array(label_path).astype(np.float32)
            meta = {}

        # -------- load optional subject-specific noise --------
        noise = None
        if getattr(s, "noise_mat", None) is not None:
            noise = _load_mat_array(s.noise_mat).astype(np.float32)

        # -------- apply augmentation / noise injection --------
        if self.augmenter is not None:
            image, label, noise = self.augmenter(
                image=image,
                label=label,
                subject_id=s.id,
                noise_field=noise,
                is_2d=False,
            )

        # item: Dict[str, Any] = {
        #     "id": s.id,
        #     "split": s.split,
        #     "scanner_type": s.scanner_type,
        #     "image": image,
        #     "label": label,
        #     "meta": meta,
        #     "paths": {
        #         "x_mat": str(s.x_mat),
        #         "gt_mat": str(s.gt_mat),
        #         "x_nii": str(s.x_nii) if s.x_nii else None,
        #         "nli_mat": str(s.nli_mat) if s.nli_mat else None,
        #         "eligible_preds_mat": str(s.eligible_preds_mat) if s.eligible_preds_mat else None,
        #         "pseudo_mat": str(s.pseudo_mat) if s.pseudo_mat else None,
        #         "noise_mat": str(s.noise_mat) if getattr(s, "noise_mat", None) else None,
        #         "label_used": str(label_path),
        #     },
        # }
        
        image = torch.as_tensor(image, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.float32)

        if image.ndim == 3:
            image = image.unsqueeze(0)   # [C, H, W, D] or [C, D, H, W] depending on your convention
        if label.ndim == 3:
            label = label.unsqueeze(0)
        
        item = {
        "id": s.id,
        "scanner_type": s.scanner_type,
        "image": image,
        "label": label,
    }

        if noise is not None:
            item["noise"] = noise  # optional

        if self.transform is not None:
            item = self.transform(item)

        if self.return_dict:
            return item

    # return item["image"], item["label"]

    #     if noise is not None:
    #         item["noise"] = noise

    #     # optional MONAI Compose (later)
    #     if self.transform is not None:
    #         item = self.transform(item)

    #     if self.return_dict:
    #         return item

        return item["image"], item["label"]