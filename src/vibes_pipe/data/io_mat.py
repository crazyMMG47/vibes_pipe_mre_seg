"""
Savenger hunts 

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import nibabel as nib
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


def extract_spacing(nii_path: str | Path | None = None):
    """
    Find the voxel size of the data (nifti format).
    
    Why we need to extract spacing? 
    - our subjects come from different scanners so their T2 space could be different. Mask spacing must follow T2 spacing (so we only need to extract spacing once.)
    """
    if nii_path is None:
        return None
    p = Path(nii_path).expanduser().resolve()
    if not p.exists():
        return None

    nii = nib.load(str(p))
    zooms = nii.header.get_zooms()[:3]
    spacing = [float(zooms[0]), float(zooms[1]), float(zooms[2])]
    if all(np.isfinite(s) and s > 0 for s in spacing):
        return spacing
    return None
            

def extract_geometry(mat_path: str | Path, *, nii_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Geometry for a MAT volume:
      - shape/dtype from MAT array
      - spacing from NIfTI (if provided)
    """
    mp = Path(mat_path).expanduser().resolve()
    mat_obj = load_mat_dict(mp)
    arr = find_primary_array(mat_obj, mat_path=mp)

    return {
        "orig_shape": [int(v) for v in arr.shape],
        "orig_spacing": extract_spacing(nii_path),
        "dtype": str(arr.dtype),
    }
    
    
def infer_companion_nii(mat_path: str | Path) -> Optional[Path]:
    p = Path(mat_path).expanduser().resolve()
    cand = p.with_suffix(".nii")
    if cand.exists():
        return cand
    cand_gz = p.with_suffix(".nii.gz")
    if cand_gz.exists():
        return cand_gz
    return None