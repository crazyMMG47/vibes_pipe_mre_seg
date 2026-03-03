from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray]


@dataclass
class SliceSpec:
    axis: int = 2                 # axial 
    index: Optional[int] = None   # if None -> use middle slice
    rotate_k: int = 0             # rotate by 90*k for display convenience


def _to_numpy(x: Any) -> np.ndarray:
    # supports np arrays and torch tensors without importing torch
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_to_3d(vol: np.ndarray) -> np.ndarray:
    """
    Make volume 3D by squeezing common channel/batch dims.
    Accepts: [D,H,W], [1,D,H,W], [D,H,W,1], [1,1,D,H,W], etc.
    """
    v = np.asarray(vol)
    # remove singleton dims
    v = np.squeeze(v)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape={v.shape}")
    return v


def get_slice(vol3d: np.ndarray, spec: SliceSpec) -> np.ndarray:
    v = _squeeze_to_3d(vol3d)
    axis = int(spec.axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    idx = spec.index
    if idx is None:
        idx = v.shape[axis] // 2
    idx = int(idx)
    if not (0 <= idx < v.shape[axis]):
        raise IndexError(f"slice index {idx} out of bounds for axis {axis} with size {v.shape[axis]}")

    if axis == 0:
        sl = v[idx, :, :]
    elif axis == 1:
        sl = v[:, idx, :]
    else:
        sl = v[:, :, idx]

    if spec.rotate_k:
        sl = np.rot90(sl, k=int(spec.rotate_k))
    return sl


def plot_image_label_slice(
    image: Any,
    label: Any = None,
    *,
    spec: SliceSpec = SliceSpec(),
    title: Optional[str] = None,
    overlay: bool = True,
    alpha: float = 0.35,
    show_contour: bool = False,
    contour_level: float = 0.5,
    cmap_image: str = "gray",
) -> plt.Figure:
    """
    Plot one slice of image (and optional label) with optional overlay/contour.
    Returns a matplotlib Figure (GUI-friendly).
    """
    img = _to_numpy(image)
    img3d = _squeeze_to_3d(img)
    img_sl = get_slice(img3d, spec)

    lbl_sl = None
    if label is not None:
        lbl = _to_numpy(label)
        lbl3d = _squeeze_to_3d(lbl)
        lbl_sl = get_slice(lbl3d, spec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_sl, cmap=cmap_image)
    ax.axis("off")

    if title:
        ax.set_title(title)

    if lbl_sl is not None:
        if overlay:
            ax.imshow(lbl_sl, alpha=float(alpha))
        if show_contour:
            ax.contour(lbl_sl, levels=[float(contour_level)])

    return fig


def plot_triplet(
    image: Any,
    gt: Any,
    pred: Any = None,
    *,
    spec: SliceSpec = SliceSpec(),
    suptitle: Optional[str] = None,
    cmap_image: str = "gray",
) -> plt.Figure:
    """
    Side-by-side: image | GT | (optional) pred.
    GUI-friendly: returns Figure.
    """
    img3d = _squeeze_to_3d(_to_numpy(image))
    gt3d = _squeeze_to_3d(_to_numpy(gt))
    img_sl = get_slice(img3d, spec)
    gt_sl = get_slice(gt3d, spec)

    has_pred = pred is not None
    ncols = 3 if has_pred else 2

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(img_sl, cmap=cmap_image); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(gt_sl); axes[1].set_title("GT"); axes[1].axis("off")

    if has_pred:
        pr3d = _squeeze_to_3d(_to_numpy(pred))
        pr_sl = get_slice(pr3d, spec)
        axes[2].imshow(pr_sl); axes[2].set_title("Pred"); axes[2].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")