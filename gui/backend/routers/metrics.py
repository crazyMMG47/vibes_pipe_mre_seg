from fastapi import APIRouter, HTTPException
from ..config import get_config
from ..services import output_reader, manifest_reader
import numpy as np

router = APIRouter(prefix="/api/subjects", tags=["metrics"])


@router.get("/{subject_id}/metrics")
def get_metrics(subject_id: str):
    cfg = get_config()

    meta = output_reader.get_subject(cfg.output_dir, subject_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")

    samples = output_reader.load_npy_volume(cfg.output_dir, subject_id, "mc_samples.npy")
    if samples is None:
        raise HTTPException(status_code=404, detail="mc_samples.npy not found.")

    # samples shape: [K, C, H, W, D] or [K, H, W, D]
    gt_array = None
    if cfg.manifest_path and cfg.workspace_root:
        gt_path = manifest_reader.get_gt_path(cfg.manifest_path, cfg.workspace_root, subject_id)
        if gt_path and gt_path.exists():
            gt_array = output_reader.load_mat_volume(gt_path)

    n_samples = samples.shape[0]
    per_sample = []

    for k in range(n_samples):
        s = samples[k]
        if s.ndim == 4:
            s = s[0]  # take first channel → [H,W,D]

        dice = None
        if gt_array is not None:
            gt = gt_array
            if gt.ndim == 4:
                gt = gt[0]
            pred_bin = (s > 0.5).astype(np.float32)
            gt_bin   = (gt > 0.5).astype(np.float32)
            intersection = (pred_bin * gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum()
            dice = float((2.0 * intersection + 1e-6) / (union + 1e-6))

        entropy_vol = _entropy(s)
        per_sample.append({
            "sample_index": k,
            "dice": dice,
            "entropy": float(np.mean(entropy_vol)),
            "std": float(np.std(s)),
        })

    mean_dice = None
    valid_dice = [p["dice"] for p in per_sample if p["dice"] is not None]
    if valid_dice:
        mean_dice = float(np.mean(valid_dice))

    return {
        "mean_dice": mean_dice,
        "mean_entropy": meta.get("mean_entropy"),
        "ged": meta.get("ged"),
        "mean_std": meta.get("mean_std"),
        "per_sample": per_sample,
    }


def _entropy(prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
