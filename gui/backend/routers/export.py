from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import get_config
from ..services import output_reader, manifest_reader

router = APIRouter(prefix="/api/subjects", tags=["export"])


class ExportRequest(BaseModel):
    sample_index: int


@router.post("/{subject_id}/set-pseudo-gt")
def set_pseudo_gt(subject_id: str, body: ExportRequest):
    cfg = get_config()

    if not cfg.workspace_root:
        raise HTTPException(status_code=503, detail="WORKSPACE_ROOT not configured — export disabled.")
    if not cfg.manifest_path:
        raise HTTPException(status_code=503, detail="MANIFEST_PATH not configured — cannot determine split.")

    samples = output_reader.load_npy_volume(cfg.output_dir, subject_id, "mc_samples.npy")
    if samples is None:
        raise HTTPException(status_code=404, detail="mc_samples.npy not found.")

    k = body.sample_index
    if k < 0 or k >= samples.shape[0]:
        raise HTTPException(status_code=400, detail=f"sample_index {k} out of range (n={samples.shape[0]}).")

    chosen = samples[k]  # [C,H,W,D] or [H,W,D]
    import numpy as np
    if chosen.ndim == 4:
        chosen = chosen[0]
    mask = (chosen > 0.5).astype("uint8")

    entry = manifest_reader.get_subject_manifest_entry(cfg.manifest_path, subject_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not in manifest.")
    split = entry.get("split", "train")

    written_path = manifest_reader.write_pseudo_gt(cfg.workspace_root, split, subject_id, mask)

    return {"written_path": str(written_path), "subject_id": subject_id}
