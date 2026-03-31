import os
import gc
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat



def load_mat_array(mat_path: Path, preferred_keys: Tuple[str, ...]) -> np.ndarray:
    """
    Load array from MATLAB file, handling both v5 and v7.3 (HDF5) formats.
    
    Args:
        mat_path: Path to .mat file
        preferred_keys: Tuple of variable names to look for (in order)
    
    Returns:
        Numpy array from the first matching key
    """
    # Detect file format
    try:
        with open(mat_path, "rb") as f:
            is_hdf5 = f.read(8).startswith(b"\x89HDF")
    except Exception:
        is_hdf5 = False
    
    # Try scipy first for v5 format
    if not is_hdf5:
        try:
            data = loadmat(str(mat_path), simplify_cells=True)
            for key in preferred_keys:
                if key in data:
                    return np.asarray(data[key])
            raise KeyError(f"None of {preferred_keys} found in {mat_path.name}")
        except Exception as scipy_error:
            last_error = scipy_error
    else:
        last_error = None
    
    # Try HDF5 format
    try:
        with h5py.File(str(mat_path), "r") as f:
            # Direct match
            for key in preferred_keys:
                if key in f:
                    return np.transpose(np.array(f[key]))
            
            # Fuzzy match
            for key in preferred_keys:
                match = next((k for k in f.keys() if key.lower() in k.lower()), None)
                if match:
                    return np.transpose(np.array(f[match]))
            
            raise KeyError(f"None of {preferred_keys} found in HDF5 datasets")
    except Exception as hdf5_error:
        if last_error:
            raise RuntimeError(
                f"Failed to read {mat_path.name}:\n"
                f"  scipy: {last_error}\n"
                f"  HDF5: {hdf5_error}"
            )
        raise