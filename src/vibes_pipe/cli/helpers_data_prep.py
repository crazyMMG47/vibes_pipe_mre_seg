"""
helpers_data_prep.py

Data preparation helpers for MRE segmentation training/retraining.

This module validates pair specs, copies required/optional .mat files into a
stable workspace layout, and writes a reproducible manifest JSON that references
workspace-relative destinations.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.io import loadmat  # type: ignore
except Exception:  # pragma: no cover
    loadmat = None  # type: ignore


JsonDict = Dict[str, Any]
ALLOWED_SPLITS = {"train", "val", "test"}
REQUIRED_KEYS = ("id", "X", "GT")
OPTIONAL_FILE_KEYS = ("eligible_preds", "NLI_output")


# -----------------------------
# JSON IO
# -----------------------------
def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=dst.parent,
    ) as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, dst)


# -----------------------------
# File utilities
# -----------------------------
def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_copy(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    if not src_p.exists():
        raise FileNotFoundError(f"Source file not found: {src_p}")
    if not src_p.is_file():
        raise FileNotFoundError(f"Source path is not a file: {src_p}")

    dst_p.parent.mkdir(parents=True, exist_ok=True)

    if dst_p.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists (overwrite=False): {dst_p}")

    with src_p.open("rb") as sf, dst_p.open("wb") as df:
        while True:
            chunk = sf.read(1024 * 1024)
            if not chunk:
                break
            df.write(chunk)

    return dst_p


def _iso8601_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------
# Pair spec validation
# -----------------------------
def _as_abs_file_path(raw_path: Any, *, field_name: str, pair_id: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"Invalid `{field_name}` for pair `{pair_id}`: expected a non-empty string path."
        )
    p = Path(raw_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing `{field_name}` file for pair `{pair_id}`: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"`{field_name}` is not a file for pair `{pair_id}`: {p}")
    return p


def _validate_and_normalize_pair(item: Any, index: int) -> JsonDict:
    if not isinstance(item, dict):
        raise ValueError(f"pairs_spec[{index}] must be a dict, got {type(item).__name__}.")

    missing = [k for k in REQUIRED_KEYS if k not in item]
    if missing:
        raise ValueError(f"pairs_spec[{index}] missing required keys: {missing}")

    pair_id = item["id"]
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError(f"pairs_spec[{index}]['id'] must be a non-empty string.")
    pair_id = pair_id.strip()

    split = item.get("split", "train")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Invalid split for pair `{pair_id}`: {split!r}. "
            f"Allowed values: {sorted(ALLOWED_SPLITS)}"
        )

    x_path = _as_abs_file_path(item["X"], field_name="X", pair_id=pair_id)
    gt_path = _as_abs_file_path(item["GT"], field_name="GT", pair_id=pair_id)

    optional_paths: Dict[str, Optional[Path]] = {}
    for key in OPTIONAL_FILE_KEYS:
        raw = item.get(key)
        if raw is None:
            optional_paths[key] = None
        else:
            optional_paths[key] = _as_abs_file_path(raw, field_name=key, pair_id=pair_id)

    meta = item.get("meta", {})
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError(f"`meta` for pair `{pair_id}` must be a dict if provided.")

    return {
        "id": pair_id,
        "split": split,
        "X": x_path,
        "GT": gt_path,
        "eligible_preds": optional_paths["eligible_preds"],
        "NLI_output": optional_paths["NLI_output"],
        "meta": meta,
    }


def validate_pairs_spec(pairs_spec: Sequence[JsonDict]) -> List[JsonDict]:
    if not isinstance(pairs_spec, Sequence) or isinstance(pairs_spec, (str, bytes)):
        raise ValueError("pairs_spec must be a list of dict items.")

    normalized: List[JsonDict] = []
    seen_ids = set()
    for idx, item in enumerate(pairs_spec):
        pair = _validate_and_normalize_pair(item, idx)
        pair_id = pair["id"]
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate pair id found: `{pair_id}`")
        seen_ids.add(pair_id)
        normalized.append(pair)
    return normalized


# -----------------------------
# MAT geometry extraction
# -----------------------------
@dataclass
class MatGeometry:
    orig_shape: Optional[List[int]] = None
    orig_spacing: Optional[List[float]] = None
    array_key: Optional[str] = None
    spacing_key: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> JsonDict:
        out: JsonDict = {
            "orig_shape": self.orig_shape,
            "orig_spacing": self.orig_spacing,
        }
        # keep these for debugging, but harmless if you ignore them downstream
        if self.array_key is not None:
            out["array_key"] = self.array_key
        if self.spacing_key is not None:
            out["spacing_key"] = self.spacing_key
        if self.error is not None:
            out["error"] = self.error
        return out


def _find_primary_mat_array(mat_obj: Dict[str, Any], *, mat_path: Path) -> Tuple[str, np.ndarray]:
    """
    Pick the primary numeric array from a .mat dict (key, array).
    Prefers common names; otherwise picks the largest numeric ndarray with ndim>=2.
    """
    preferred = ("X", "x", "image", "img", "volume", "data", "GT", "gt", "mask", "label")
    for key in preferred:
        val = mat_obj.get(key)
        if isinstance(val, np.ndarray) and val.ndim >= 2 and np.issubdtype(val.dtype, np.number):
            return key, val

    best_key: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    for key, val in mat_obj.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 2 and np.issubdtype(val.dtype, np.number):
            if best_arr is None or val.size > best_arr.size:
                best_key, best_arr = key, val

    if best_arr is None or best_key is None:
        raise ValueError(f"No numeric array with ndim>=2 found in MAT file: {mat_path}")

    return best_key, best_arr


def _coerce_spacing(raw: Any) -> Optional[List[float]]:
    """
    Try to coerce a spacing-like value into [sx, sy, sz].
    Accepts np.ndarray-like objects with >=3 elements.
    """
    if not isinstance(raw, np.ndarray):
        return None
    flat = raw.reshape(-1)
    if flat.size < 3:
        return None
    try:
        return [float(flat[0]), float(flat[1]), float(flat[2])]
    except Exception:
        return None


def _extract_mat_geometry(mat_path: Path) -> MatGeometry:
    """
    Extract original shape and optional spacing from a .mat file.
    Best-effort; on failure, returns MatGeometry(error=...).
    """
    g = MatGeometry()

    try:
        if mat_path.stat().st_size == 0:
            g.error = "empty_file"
            return g
    except Exception as e:
        g.error = f"stat_failed:{e}"
        return g

    if loadmat is None:
        g.error = "scipy_not_available"
        return g

    try:
        mat_obj = loadmat(str(mat_path))
    except Exception as e:
        g.error = f"loadmat_failed:{e}"
        return g

    try:
        key, arr = _find_primary_mat_array(mat_obj, mat_path=mat_path)
        g.array_key = key
        g.orig_shape = [int(v) for v in arr.shape]
    except Exception as e:
        g.error = f"no_primary_array:{e}"
        return g

    # spacing: try common keys
    for sk in (
        "spacing",
        "voxel_spacing",
        "voxelSpacing",
        "pixdim",
        "resolution",
        "spacing_mm",
    ):
        if sk in mat_obj:
            sp = _coerce_spacing(mat_obj.get(sk))
            if sp is not None:
                g.orig_spacing = sp
                g.spacing_key = sk
                break

    return g


# -----------------------------
# Workspace + manifest building
# -----------------------------
def _file_entry(
    src_path: Path,
    dst_rel: Path,
    workspace_root: Path,
    overwrite: bool,
    compute_hash: bool,
) -> JsonDict:
    dst_abs = workspace_root / dst_rel
    safe_copy(src_path, dst_abs, overwrite=overwrite)
    return {
        "src": str(src_path),
        "dst": dst_rel.as_posix(),
        "sha256": sha256_file(dst_abs) if compute_hash else None,
    }


def build_workspace_from_pairs(
    pairs_spec: Sequence[JsonDict],
    workspace_root: str | Path,
    overwrite: bool = False,
    compute_hash: bool = True,
) -> JsonDict:
    """
    Validate pair specs, copy files into workspace, and return manifest dict.

    Geometry metadata is recorded under:
        meta["geometry_preprocess"] = {
          orig_image_shape, orig_image_spacing,
          orig_label_shape, orig_label_spacing,
          (optional) orig_eligible_preds_shape, orig_eligible_preds_spacing,
          (optional) orig_nli_output_shape, orig_nli_output_spacing,
          errors: { ... }   # only if any extraction failed
        }

    Geometry is extracted from the COPIED workspace files to ensure the manifest
    reflects the workspace state (and not an external source that might change).
    """
    normalized = validate_pairs_spec(pairs_spec)
    ws_root = Path(workspace_root).expanduser().resolve()
    ws_root.mkdir(parents=True, exist_ok=True)

    manifest_pairs: List[JsonDict] = []

    def _abs_from_dst_rel(dst_rel_str: str) -> Path:
        return ws_root / Path(dst_rel_str)

    for pair in sorted(normalized, key=lambda p: p["id"]):
        pair_id = pair["id"]
        split = pair["split"]
        pair_dir = Path(split) / pair_id

        # copy required
        x_entry = _file_entry(
            src_path=pair["X"],
            dst_rel=pair_dir / "X.mat",
            workspace_root=ws_root,
            overwrite=overwrite,
            compute_hash=compute_hash,
        )
        gt_entry = _file_entry(
            src_path=pair["GT"],
            dst_rel=pair_dir / "GT.mat",
            workspace_root=ws_root,
            overwrite=overwrite,
            compute_hash=compute_hash,
        )

        files: JsonDict = {
            "X": x_entry,
            "GT": gt_entry,
            "eligible_preds": None,
            "NLI_output": None,
        }

        # copy optional
        eligible_entry: Optional[JsonDict] = None
        if pair["eligible_preds"] is not None:
            eligible_entry = _file_entry(
                src_path=pair["eligible_preds"],
                dst_rel=pair_dir / "eligible_preds.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["eligible_preds"] = eligible_entry

        nli_entry: Optional[JsonDict] = None
        if pair["NLI_output"] is not None:
            nli_entry = _file_entry(
                src_path=pair["NLI_output"],
                dst_rel=pair_dir / "NLI_output.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["NLI_output"] = nli_entry

        # ---- meta: geometry_preprocess (new) ----
        meta = dict(pair.get("meta", {}) or {})
        geo = dict(meta.get("geometry_preprocess", {}) or {})
        errors: Dict[str, Any] = {}

        x_geo = _extract_mat_geometry(_abs_from_dst_rel(x_entry["dst"]))
        gt_geo = _extract_mat_geometry(_abs_from_dst_rel(gt_entry["dst"]))

        geo["orig_image_shape"] = x_geo.orig_shape
        geo["orig_image_spacing"] = x_geo.orig_spacing
        geo["orig_label_shape"] = gt_geo.orig_shape
        geo["orig_label_spacing"] = gt_geo.orig_spacing

        if x_geo.error:
            errors["X"] = x_geo.to_dict()
        if gt_geo.error:
            errors["GT"] = gt_geo.to_dict()

        if eligible_entry is not None:
            ep_geo = _extract_mat_geometry(_abs_from_dst_rel(eligible_entry["dst"]))
            geo["orig_eligible_preds_shape"] = ep_geo.orig_shape
            geo["orig_eligible_preds_spacing"] = ep_geo.orig_spacing
            if ep_geo.error:
                errors["eligible_preds"] = ep_geo.to_dict()

        if nli_entry is not None:
            n_geo = _extract_mat_geometry(_abs_from_dst_rel(nli_entry["dst"]))
            geo["orig_nli_output_shape"] = n_geo.orig_shape
            geo["orig_nli_output_spacing"] = n_geo.orig_spacing
            if n_geo.error:
                errors["NLI_output"] = n_geo.to_dict()

        if errors:
            geo["errors"] = errors

        meta["geometry_preprocess"] = geo

        manifest_pairs.append(
            {
                "id": pair_id,
                "split": split,
                "files": files,
                "meta": meta,
            }
        )

    manifest: JsonDict = {
        "schema_version": "1.0",
        "created_utc": _iso8601_utc_now(),
        "workspace_root": str(ws_root),
        "splits": ["train", "val", "test"],
        "pairs": manifest_pairs,
    }
    return manifest


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    root_path = Path("~/Desktop/vibes_pipe/experiments/mre_data_prep_test").expanduser()
    pairs_json_path = root_path / "pairs.json"
    workspace = root_path / "workspace_root"
    manifest_path = workspace / "manifest.json"

    pairs_data = read_json(pairs_json_path)
    if not isinstance(pairs_data, list):
        raise ValueError(
            "Demo expects pairs.json to contain a top-level list of pair items "
            "(i.e., directly the `pairs_spec` list)."
        )

    manifest_data = build_workspace_from_pairs(
        pairs_spec=pairs_data,
        workspace_root=workspace,
        overwrite=False,
        compute_hash=True,
    )
    write_json_atomic(manifest_path, manifest_data)