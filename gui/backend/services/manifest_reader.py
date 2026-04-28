from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.io as sio


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_subject_manifest_entry(manifest_path: Path, subject_id: str) -> dict | None:
    manifest = _load_manifest(manifest_path)
    for pair in manifest.get("pairs", []):
        if str(pair.get("id")) == str(subject_id):
            return pair
    return None


def _resolve(workspace_root: Path, dst_rel: str) -> Path:
    return (workspace_root / Path(dst_rel)).resolve()


def get_raw_image_path(manifest_path: Path, workspace_root: Path, subject_id: str) -> Path | None:
    entry = get_subject_manifest_entry(manifest_path, subject_id)
    if entry is None:
        return None
    files = entry.get("files", {})
    t2 = files.get("t2stack")
    if t2 is None:
        return None
    return _resolve(workspace_root, t2["dst"])


def get_gt_path(manifest_path: Path, workspace_root: Path, subject_id: str) -> Path | None:
    entry = get_subject_manifest_entry(manifest_path, subject_id)
    if entry is None:
        return None
    files = entry.get("files", {})
    gt = files.get("GT(human)")
    if gt is None:
        return None
    return _resolve(workspace_root, gt["dst"])


def get_stiffness_path(manifest_path: Path, workspace_root: Path, subject_id: str) -> Path | None:
    entry = get_subject_manifest_entry(manifest_path, subject_id)
    if entry is None:
        return None
    files = entry.get("files", {})
    nli = files.get("NLI_output")
    if nli is None:
        return None
    return _resolve(workspace_root, nli["dst"])


def write_pseudo_gt(workspace_root: Path, split: str, subject_id: str, mask_array: np.ndarray) -> Path:
    out_dir = workspace_root / split / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pseudo_mask.mat"
    sio.savemat(str(out_path), {"data": mask_array})
    return out_path
