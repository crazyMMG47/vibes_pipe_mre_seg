from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.io as sio


def list_subjects(output_dir: Path) -> list[dict]:
    results = []
    if not output_dir.exists():
        return results
    for sub_dir in sorted(output_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        meta_path = sub_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            results.append({
                "id": meta.get("id", sub_dir.name),
                "scanner_type": meta.get("scanner_type", "UNKNOWN"),
                "dice": meta.get("dice"),
                "n_mc_samples": meta.get("n_mc_samples", 0),
                "mean_entropy": meta.get("mean_entropy"),
                "ged": meta.get("ged"),
                "mean_std": meta.get("mean_std"),
                "saved_at": meta.get("saved_at", ""),
                "stiffness_available": False,  # enriched by router
            })
        except Exception:
            continue
    return results


def get_subject(output_dir: Path, subject_id: str) -> dict | None:
    sub_dir = output_dir / subject_id
    meta_path = sub_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        meta.setdefault("id", subject_id)
        meta.setdefault("stiffness_available", False)
        return meta
    except Exception:
        return None


def load_npy_volume(output_dir: Path, subject_id: str, filename: str) -> np.ndarray | None:
    path = output_dir / subject_id / filename
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        return None


def load_mat_volume(mat_path: Path) -> np.ndarray | None:
    """Load a .mat file and return the primary array."""
    if not mat_path.exists():
        return None
    try:
        # Try scipy loadmat first (v5)
        data = sio.loadmat(str(mat_path))
        arrays = [v for k, v in data.items()
                  if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim >= 3]
        if not arrays:
            return None
        # pick the largest array
        arr = max(arrays, key=lambda a: a.size)
        return arr.astype(np.float32)
    except Exception:
        try:
            import h5py
            with h5py.File(str(mat_path), "r") as f:
                keys = [k for k in f.keys() if not k.startswith("#")]
                if not keys:
                    return None
                arr = np.array(f[keys[0]])
                # h5py transposes axes vs scipy; undo it
                arr = arr.T
                return arr.astype(np.float32)
        except Exception:
            return None
