"""
A preprocessor class handling clahe, resampling, resizing, and normalization.

It accepts a config dict and is able to perform the image preprocessing tailored the user config. 
"""


from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .io_mat import extract_geometry, find_primary_array, load_mat_dict, infer_companion_nii


class Preprocessor:
    """
    3D preprocessing for MRE volumes: CLAHE -> resample -> resize -> normalize.
    Configurable via a dict (e.g., loaded from YAML).
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        # cfg is a python dict, loaded from yaml file
        # fallback default value is set in case no yaml is provided
        cfg = cfg or {}
        p = cfg.get("preprocess", cfg)  # allow passing either full yaml or preprocess sub-dict
        
        self.target_spacing = tuple(p.get("target_spacing", (1.5, 1.5, 1.5)))
        ts = p.get("target_size", (128, 128, 64))
        self.target_size = None if ts is None else tuple(ts)

        # ---- sub-configs with safe defaults ----
        self.clahe_cfg = {
            "enabled": True,
            "clip_limit": 2.0,
            "tile_grid_size": (8, 8),
            **p.get("clahe", {}),
        }

        self.resample_cfg = {
            "enabled": True,
            "order_image": 1,
            "order_label": 0,
            "prefilter": False,
            **p.get("resample", {}),
        }

        self.norm_cfg = {
            "enabled": True,
            "mask_positive_only": True,
            "eps": 1e-8,
            **p.get("normalize", {}),
        }

        self.label_cfg = {
            "binarize_threshold": 0.5,
            "dtype": "float32",
            **p.get("label", {}),
        }

    def load_mat_volume(
        self,
        mat_path: str,
        *,
        nii_path: str | None = None,
    ) -> tuple[np.ndarray, Optional[tuple[float, float, float]], Dict[str, Any]]:
        mat_obj = load_mat_dict(mat_path)
        volume = find_primary_array(mat_obj, mat_path=mat_path).astype(np.float32)

        geo = extract_geometry(mat_path, nii_path=nii_path)
        spacing = geo.get("orig_spacing")
        spacing_tuple = tuple(spacing) if spacing is not None else None
        return volume, spacing_tuple, geo

    def resample(
        self,
        volume: np.ndarray,
        original_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        order: int = 1,
        prefilter: bool = False,
    ) -> np.ndarray:
        zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=prefilter)

    def resize(
        self,
        volume: np.ndarray,
        target_size: tuple[int, int, int],
        order: int = 1,
        prefilter: bool = False,
    ) -> np.ndarray:
        zoom_factors = [target / current for target, current in zip(target_size, volume.shape)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=prefilter)

    def apply_clahe(self, volume: np.ndarray) -> np.ndarray:
        if not self.clahe_cfg.get("enabled", True):
            return volume

        clip_limit = float(self.clahe_cfg.get("clip_limit", 2.0))
        tgs = self.clahe_cfg.get("tile_grid_size", (8, 8))
        tile_grid_size = (int(tgs[0]), int(tgs[1]))

        pct = self.clahe_cfg.get("percentile_clip", (1, 99))
        p_low, p_high = float(pct[0]), float(pct[1])
        eps = float(self.clahe_cfg.get("eps", 1e-8))

        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        lo, hi = np.percentile(volume, (p_low, p_high))
        volume_clipped = np.clip(volume, lo, hi)
        volume_norm_8bit = (255 * (volume_clipped - lo) / (hi - lo + eps)).astype(np.uint8)

        slices_clahe = [clahe_obj.apply(volume_norm_8bit[:, :, i]) for i in range(volume_norm_8bit.shape[2])]
        volume_clahe_8bit = np.stack(slices_clahe, axis=2)

        return (volume_clahe_8bit.astype(np.float32) / 255.0) * (hi - lo) + lo

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        if not self.norm_cfg.get("enabled", True):
            return volume

        eps = float(self.norm_cfg.get("eps", 1e-8))
        if self.norm_cfg.get("mask_positive_only", True):
            valid_voxels = volume[volume > 0]
        else:
            valid_voxels = volume.reshape(-1)

        if valid_voxels.size == 0:
            return volume

        mean = float(np.mean(valid_voxels))
        std = float(np.std(valid_voxels))
        return (volume - mean) / (std + eps)

    from pathlib import Path

    def process_pair(self, image_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        # infer image NIfTI once
        image_nii = infer_companion_nii(image_path)

        image, img_spacing, img_geo = self.load_mat_volume(image_path, nii_path=str(image_nii) if image_nii else None)

        # label spacing should follow image spacing (even if label has no nifti)
        label, _, lbl_geo = self.load_mat_volume(label_path, nii_path=None)
        lbl_geo["orig_spacing"] = img_geo.get("orig_spacing")  # keep meta consistent
        lbl_spacing = img_spacing

        orig_meta = {
            "orig_image_shape": img_geo.get("orig_shape"),
            "orig_image_spacing": img_geo.get("orig_spacing"),
            "orig_label_shape": lbl_geo.get("orig_shape"),
            "orig_label_spacing": lbl_geo.get("orig_spacing"),
            "image_nii_path": str(image_nii) if image_nii else None,
        }

        # CLAHE
        image = self.apply_clahe(image)

        # resample (only if spacing exists)
        if self.resample_cfg.get("enabled", True):
            prefilter = bool(self.resample_cfg.get("prefilter", False))
            order_img = int(self.resample_cfg.get("order_image", 1))
            order_lbl = int(self.resample_cfg.get("order_label", 0))

            if img_spacing is not None:
                image = self.resample(image, img_spacing, self.target_spacing, order=order_img, prefilter=prefilter)
                label = self.resample(label, img_spacing, self.target_spacing, order=order_lbl, prefilter=prefilter)

        # ensure label matches image
        if label.shape != image.shape:
            label = self.resize(label, image.shape, order=0, prefilter=False)

        # final resize to target_size
        if self.target_size is not None:
            image = self.resize(image, self.target_size, order=1, prefilter=False)
            label = self.resize(label, self.target_size, order=0, prefilter=False)

        # normalize + binarize label
        image = self.normalize(image)
        thr = float(self.label_cfg.get("binarize_threshold", 0.5))
        label = (label > thr).astype(getattr(np, str(self.label_cfg.get("dtype", "float32"))))

        return image, label, orig_meta