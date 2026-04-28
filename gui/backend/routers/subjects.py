from fastapi import APIRouter, HTTPException
from ..config import get_config
from ..services import output_reader, manifest_reader

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


@router.get("")
def list_subjects():
    cfg = get_config()
    subjects = output_reader.list_subjects(cfg.output_dir)
    for s in subjects:
        s["stiffness_available"] = False
        if cfg.manifest_path and cfg.workspace_root:
            try:
                p = manifest_reader.get_stiffness_path(cfg.manifest_path, cfg.workspace_root, s["id"])
                s["stiffness_available"] = p is not None and p.exists()
            except Exception:
                pass
    return subjects


@router.get("/{subject_id}")
def get_subject(subject_id: str):
    cfg = get_config()
    detail = output_reader.get_subject(cfg.output_dir, subject_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found in output dir.")
    detail["stiffness_available"] = False
    if cfg.manifest_path and cfg.workspace_root:
        try:
            p = manifest_reader.get_stiffness_path(cfg.manifest_path, cfg.workspace_root, subject_id)
            detail["stiffness_available"] = p is not None and p.exists()
        except Exception:
            pass
    return detail
