"""
Savenger hunts 

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.io import loadmat


def load_mat_dict(path: str | Path) -> Dict[str, Any]:
    """
    A wrapper for loading the file.
    """
    p = Path(path).expanduser().resolve() # ensure the path is formatted correctly 
    return loadmat(p)


def find_primary_array(mat_obj: Dict[str, Any], *, mat_path: str | Path) -> np.ndarray:
    """
    Guess the variable name in the file. 
    """
    # Priority search 
    preferred = ("image", "data", "mask", "t2stack", "pred")
    for key in preferred:
        val = mat_obj.get(key)
        # ensures the dimension of the arrow is at least 2 (safety check)
        if isinstance(val, np.ndarray) and val.ndim >= 2 and np.issubdtype(val.dtype, np.number):
            return val

    candidates: list[np.ndarray] = []
    for key, val in mat_obj.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 2 and np.issubdtype(val.dtype, np.number):
            candidates.append(val)

    if not candidates:
        raise ValueError(f"No numeric array with ndim>=2 found in MAT file: {Path(mat_path)}")
    return max(candidates, key=lambda x: x.size)


def extract_spacing(mat_obj: Dict[str, Any]) -> Optional[list[float]]:
    """
    Find the voxel size of the data. 
    """
    for key in ("spacing", "voxel_spacing", "pixdim", "resolution"):
        raw = mat_obj.get(key)
        if isinstance(raw, np.ndarray):
            flat = raw.reshape(-1)
            if flat.size >= 3:
                return [float(flat[0]), float(flat[1]), float(flat[2])]
    return None


def extract_geometry(path: str | Path) -> Dict[str, Any]:
    
    """
    manager function that coordinates all above functions. 
    """
    p = Path(path).expanduser().resolve()
    mat_obj = load_mat_dict(p)
    arr = find_primary_array(mat_obj, mat_path=p)
    return {
        "orig_shape": [int(v) for v in arr.shape],
        "orig_spacing": extract_spacing(mat_obj),
        "dtype": str(arr.dtype),
    }