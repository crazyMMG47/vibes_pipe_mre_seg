import numpy as np
from typing import Tuple


def compute_ge_noise(magimg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute noise profiles from GE magimg data.
    
    Args:
        magimg: Magnitude image stack (typically 4D or 5D)
    
    Returns:
        t2stack, noise, noise_scaled, t2noise
    """
    if magimg.ndim < 3:
        raise ValueError(f"magimg must have ≥3 dimensions, got {magimg.ndim}")
    
    print(f"GE data shape: {magimg.shape}")
    
    # Average over non-spatial dimensions
    reduce_axes = tuple(range(3, magimg.ndim))
    t2stack = magimg.mean(axis=reduce_axes) if reduce_axes else magimg
    
    # Select reference frame based on dimensionality
    if magimg.ndim >= 5:
        ref_frame = magimg[:, :, :, 0, 3]
        print("Using reference: [:, :, :, 0, 3] (5D data)")
    elif magimg.ndim == 4:
        ref_frame = magimg[:, :, :, 0]
        print("Using reference: [:, :, :, 0] (4D data)")
    else:
        ref_frame = magimg
        print("Using reference: entire 3D volume")
    
    # Compute noise
    noise = t2stack - np.abs(ref_frame)
    noise_scaled = noise * 1000.0
    t2noise = t2stack + noise
    
    return t2stack, noise, noise_scaled, t2noise
