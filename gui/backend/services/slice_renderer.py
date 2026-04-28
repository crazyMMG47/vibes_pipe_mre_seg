"""
Renders a 2D PNG from a 3D numpy volume.
Reuses get_slice from src/vibes_pipe/viz/slices.py for axis/index extraction.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make the repo root importable so we can reuse viz code
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from src.vibes_pipe.viz.slices import get_slice, SliceSpec
    _HAS_SLICE_LIB = True
except ImportError:
    _HAS_SLICE_LIB = False


def _squeeze_to_3d(vol: np.ndarray) -> np.ndarray:
    v = np.squeeze(vol)
    if v.ndim == 3:
        return v
    if v.ndim == 4:          # [C,H,W,D] → take first channel
        return v[0]
    if v.ndim == 2:          # already a slice somehow
        return v[:, :, np.newaxis]
    raise ValueError(f"Cannot reduce volume of shape {vol.shape} to 3D.")


def _extract_slice(vol3d: np.ndarray, axis: int, index: int | None) -> np.ndarray:
    if _HAS_SLICE_LIB:
        spec = SliceSpec(axis=axis, index=index)
        return get_slice(vol3d, spec)
    # fallback
    n = vol3d.shape[axis]
    idx = n // 2 if index is None else max(0, min(index, n - 1))
    if axis == 0:
        return vol3d[idx, :, :]
    elif axis == 1:
        return vol3d[:, idx, :]
    else:
        return vol3d[:, :, idx]


def _normalize(sl: np.ndarray) -> np.ndarray:
    lo = np.percentile(sl, 1)
    hi = np.percentile(sl, 99)
    norm = (sl.astype(np.float32) - lo) / (hi - lo + 1e-8)
    return np.clip(norm, 0.0, 1.0)


def _apply_colormap(arr_01: np.ndarray, colormap: str) -> np.ndarray:
    """Return uint8 RGB [H,W,3]."""
    if colormap == "gray":
        gray8 = (arr_01 * 255).astype(np.uint8)
        return np.stack([gray8, gray8, gray8], axis=-1)
    elif colormap == "hot":
        # simple hot colormap: black→red→yellow→white
        r = np.clip(arr_01 * 3.0, 0, 1)
        g = np.clip(arr_01 * 3.0 - 1.0, 0, 1)
        b = np.clip(arr_01 * 3.0 - 2.0, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)
    else:
        gray8 = (arr_01 * 255).astype(np.uint8)
        return np.stack([gray8, gray8, gray8], axis=-1)


def _burn_contour(rgb: np.ndarray, mask_2d: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    """Overlay white contour of mask boundary onto rgb [H,W,3]."""
    import scipy.ndimage as ndi
    binary = (mask_2d > 0.5).astype(np.uint8)
    eroded = ndi.binary_erosion(binary).astype(np.uint8)
    contour = binary - eroded
    out = rgb.copy()
    for c, val in enumerate(color):
        out[:, :, c] = np.where(contour.astype(bool), val, out[:, :, c])
    return out


def render_slice(
    volume: np.ndarray,
    axis: int = 2,
    index: int | None = None,
    overlay_mask: np.ndarray | None = None,
    threshold: float = 0.5,
    colormap: str = "gray",
) -> bytes:
    """
    Render one 2D slice of a 3D volume to PNG bytes.
    volume: any shape reducible to 3D (H,W,D convention).
    Returns raw PNG bytes.
    """
    vol3d = _squeeze_to_3d(volume)
    sl = _extract_slice(vol3d, axis, index)
    sl_norm = _normalize(sl)
    rgb = _apply_colormap(sl_norm, colormap)

    if overlay_mask is not None:
        try:
            mask3d = _squeeze_to_3d(overlay_mask)
            mask_sl = _extract_slice(mask3d, axis, index)
            rgb = _burn_contour(rgb, mask_sl)
        except Exception:
            pass  # overlay failure is non-fatal

    img = Image.fromarray(rgb, mode="RGB")

    # Upscale small images so browser doesn't render a tiny thumbnail
    min_dim = 256
    if img.width < min_dim or img.height < min_dim:
        scale = max(min_dim // max(img.width, img.height, 1), 2)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
