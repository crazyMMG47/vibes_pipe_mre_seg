from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from ..config import get_config
from ..services import output_reader, manifest_reader, slice_renderer

router = APIRouter(prefix="/api/subjects", tags=["slices"])

VALID_VOLUMES = {"raw", "gt", "mean", "stiffness"}


@router.get("/{subject_id}/slice")
def get_slice(
    subject_id: str,
    volume: str = Query("mean"),
    axis: int = Query(2, ge=0, le=2),
    index: int = Query(-1),           # -1 = middle
    overlay: str = Query("none"),     # "gt" | "none"
    threshold: float = Query(0.5),
):
    cfg = get_config()

    # ── resolve volume array ────────────────────────────────────────────
    vol_array = None
    overlay_mask = None

    if volume == "raw":
        if cfg.manifest_path and cfg.workspace_root:
            path = manifest_reader.get_raw_image_path(cfg.manifest_path, cfg.workspace_root, subject_id)
            if path and path.exists():
                vol_array = output_reader.load_mat_volume(path)
        if vol_array is None:
            raise HTTPException(status_code=404, detail="Raw image not found (check MANIFEST_PATH).")

    elif volume == "gt":
        if cfg.manifest_path and cfg.workspace_root:
            path = manifest_reader.get_gt_path(cfg.manifest_path, cfg.workspace_root, subject_id)
            if path and path.exists():
                vol_array = output_reader.load_mat_volume(path)
        if vol_array is None:
            raise HTTPException(status_code=404, detail="GT mask not found (check MANIFEST_PATH).")

    elif volume == "mean":
        vol_array = output_reader.load_npy_volume(cfg.output_dir, subject_id, "prob_map.npy")
        if vol_array is None:
            raise HTTPException(status_code=404, detail="prob_map.npy not found for this subject.")

    elif volume.startswith("sample_"):
        try:
            k = int(volume.split("_", 1)[1])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid volume key: {volume}")
        samples = output_reader.load_npy_volume(cfg.output_dir, subject_id, "mc_samples.npy")
        if samples is None:
            raise HTTPException(status_code=404, detail="mc_samples.npy not found for this subject.")
        if samples.ndim < 4:
            raise HTTPException(status_code=500, detail="mc_samples.npy has unexpected shape.")
        # shape: [K, C, H, W, D] or [K, H, W, D]
        if k >= samples.shape[0]:
            raise HTTPException(status_code=404, detail=f"Sample index {k} out of range (n={samples.shape[0]}).")
        vol_array = samples[k]  # [C,H,W,D] or [H,W,D]

    elif volume == "stiffness":
        if not (cfg.manifest_path and cfg.workspace_root):
            raise HTTPException(status_code=404, detail="Stiffness unavailable (MANIFEST_PATH not set).")
        path = manifest_reader.get_stiffness_path(cfg.manifest_path, cfg.workspace_root, subject_id)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="NLI_output.mat not found for this subject.")
        vol_array = output_reader.load_mat_volume(path)
        if vol_array is None:
            raise HTTPException(status_code=404, detail="Could not read NLI_output.mat.")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown volume: '{volume}'. Use raw|gt|mean|sample_{{k}}|stiffness.")

    # ── optional GT overlay ─────────────────────────────────────────────
    if overlay == "gt" and cfg.manifest_path and cfg.workspace_root:
        gt_path = manifest_reader.get_gt_path(cfg.manifest_path, cfg.workspace_root, subject_id)
        if gt_path and gt_path.exists():
            overlay_mask = output_reader.load_mat_volume(gt_path)

    # ── render ──────────────────────────────────────────────────────────
    is_stiffness = (volume == "stiffness")
    png_bytes = slice_renderer.render_slice(
        vol_array,
        axis=axis,
        index=None if index < 0 else index,
        overlay_mask=overlay_mask,
        threshold=threshold,
        colormap="hot" if is_stiffness else "gray",
    )
    return Response(content=png_bytes, media_type="image/png",
                    headers={"Cache-Control": "max-age=300"})
