"""
A very straight forward image processing module. 
I will apply the CLAHE normalization to it, together with resampling and resizing.

"""
import numpy as np
import nibabel as nib
from scipy import ndimage
import cv2
from typing import Tuple, Optional


class ImageProcessor:
    """
    Core image processing operations. 
    Moved from old Preprocessor class to be more modular.
    """
    def __init__(self, target_spacing=(1.5, 1.5, 1.5), target_size=(128, 128, 64)):
        self.target_spacing = target_spacing
        self.target_size = target_size

    def apply_clahe_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Applies 3D CLAHE slice-by-slice (Z-axis).
        
        CLAHE is a very robust contrast normalization method. 
        It operates on 2D slices, so we apply it across the depth (z-axis).
        """
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Robust min-max scaling for CLAHE
        p_low, p_high = np.percentile(volume, (1, 99)) # ignore extreme outliers 
        vol_norm = np.clip((volume - p_low) / (p_high - p_low + 1e-8), 0, 1)
        # Quantization to 8-bit (required by OpenCV CLAHE)
        vol_8bit = (vol_norm * 255).astype(np.uint8)
        # Apply CLAHE slice-by-slice
        slices = [clahe_obj.apply(vol_8bit[:, :, i]) for i in range(vol_8bit.shape[2])]
        vol_clahe = np.stack(slices, axis=2).astype(np.float32)
        
        # Restore approximate original scale
        return (vol_clahe / 255.0) * (p_high - p_low) + p_low

    def resample(self, volume: np.ndarray, orig_spacing: Tuple, order: int = 1) -> np.ndarray:
        zoom_factors = [o/t for o, t in zip(orig_spacing, self.target_spacing)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)

    def resize(self, volume: np.ndarray, order: int = 1) -> np.ndarray:
        zoom_factors = [t/c for t, c in zip(self.target_size, volume.shape)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)