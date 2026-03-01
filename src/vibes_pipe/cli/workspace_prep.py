#!/usr/bin/env python3
"""
workspace_prep.py

Freeze a dataset into a reproducible workspace layout + manifest.

Responsibilities:
  - Validate a pairs spec (pairs.json)
  - Copy required/optional files into a stable workspace structure:
        <workspace_root>/{train,val,test}/{id}/X.mat
        <workspace_root>/{train,val,test}/{id}/GT.mat
        <workspace_root>/{train,val,test}/{id}/NLI_output.mat           (optional)
        <workspace_root>/{train,val,test}/{id}/eligible_preds.mat       (optional)
        <workspace_root>/{train,val,test}/{id}/X.nii or X.nii.gz        (optional; for spacing)
  - Write a manifest.json describing the frozen snapshot (workspace-relative dst paths)
  - Record geometry metadata using vibes_pipe.data.io_mat (shape/dtype from .mat,
    spacing from X_nii if available; mask spacing follows image spacing)

Notes:
  - This module intentionally avoids duplicating MAT/NIfTI parsing logic.
    All geometry extraction is delegated to vibes_pipe.data.io_mat.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from vibes_pipe.data.io_mat import extract_geometry, infer_companion_nii

JsonDict = Dict[str, Any]

ALLOWED_SPLITS = {"train", "val", "test"}

# pairs.json required keys
REQUIRED_KEYS = ("id", "X", "GT")

# pairs.json optional keys (paths); X_nii enables spacing extraction from NIfTI
OPTIONAL_FILE_KEYS = ("eligible_preds", "NLI_output", "X_nii")


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


def _iso8601_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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

    # optional paths (validate if provided)
    optional_paths: Dict[str, Optional[Path]] = {}
    for key in OPTIONAL_FILE_KEYS:
        raw = item.get(key)
        if raw is None:
            optional_paths[key] = None
        else:
            optional_paths[key] = _as_abs_file_path(raw, field_name=key, pair_id=pair_id)

    # convenience: if X_nii not provided, try to infer from X.mat location
    if optional_paths.get("X_nii") is None:
        inferred = infer_companion_nii(x_path)
        if inferred is not None:
            optional_paths["X_nii"] = inferred

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
        "X_nii": optional_paths["X_nii"],
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
# Workspace + manifest building
# -----------------------------
def build_workspace_from_pairs(
    pairs_spec: Sequence[JsonDict],
    workspace_root: str | Path,
    overwrite: bool = False,
    compute_hash: bool = True,
) -> JsonDict:
    """
    Validate pair specs, copy files into workspace, and return a manifest dict.

    Geometry metadata is recorded under:
        meta["geometry_preprocess"] = {
          orig_image_shape, orig_image_spacing, orig_image_dtype,
          orig_label_shape, orig_label_spacing, orig_label_dtype,
          (optional) orig_eligible_preds_shape, orig_eligible_preds_dtype,
          (optional) orig_nli_output_shape, orig_nli_output_dtype,
          errors: {...}  # only if something missing/unreadable
        }

    Spacing policy:
      - Image spacing is extracted from X_nii (preferred/expected in your dataset).
      - Mask spacing follows image spacing (the mask lives in the same voxel grid).
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

        # ---- copy required ----
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
            "X_nii": None,
            "eligible_preds": None,
            "NLI_output": None,
        }

        # ---- copy optional: X_nii ----
        if pair.get("X_nii") is not None:
            src = pair["X_nii"]
            # preserve .nii vs .nii.gz
            suffix = "".join(src.suffixes)  # ".nii" or ".nii.gz"
            xnii_entry = _file_entry(
                src_path=src,
                dst_rel=pair_dir / f"X{suffix}",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["X_nii"] = xnii_entry

        # ---- copy optional: eligible_preds ----
        if pair.get("eligible_preds") is not None:
            eligible_entry = _file_entry(
                src_path=pair["eligible_preds"],
                dst_rel=pair_dir / "eligible_preds.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["eligible_preds"] = eligible_entry

        # ---- copy optional: NLI_output ----
        if pair.get("NLI_output") is not None:
            nli_entry = _file_entry(
                src_path=pair["NLI_output"],
                dst_rel=pair_dir / "NLI_output.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["NLI_output"] = nli_entry

        # ---- geometry meta (delegated to io_mat.py) ----
        meta = dict(pair.get("meta", {}) or {})
        geo = dict(meta.get("geometry_preprocess", {}) or {})
        errors: Dict[str, Any] = {}

        x_mat_abs = _abs_from_dst_rel(x_entry["dst"])
        gt_mat_abs = _abs_from_dst_rel(gt_entry["dst"])

        x_nii_abs: Optional[Path] = None
        if files.get("X_nii") is not None:
            x_nii_abs = _abs_from_dst_rel(files["X_nii"]["dst"])

        # shapes/dtypes from MAT; spacing from X_nii
        xg = extract_geometry(x_mat_abs, nii_path=x_nii_abs)
        gg = extract_geometry(gt_mat_abs, nii_path=None)

        geo["orig_image_shape"] = xg.get("orig_shape")
        geo["orig_image_spacing"] = xg.get("orig_spacing")
        geo["orig_image_dtype"] = xg.get("dtype")

        geo["orig_label_shape"] = gg.get("orig_shape")
        geo["orig_label_spacing"] = xg.get("orig_spacing")  # mask follows image spacing
        geo["orig_label_dtype"] = gg.get("dtype")

        # optional mats: record shape/dtype (spacing follows image spacing if you ever want it later)
        if files.get("eligible_preds") is not None:
            ep_abs = _abs_from_dst_rel(files["eligible_preds"]["dst"])
            epg = extract_geometry(ep_abs, nii_path=None)
            geo["orig_eligible_preds_shape"] = epg.get("orig_shape")
            geo["orig_eligible_preds_dtype"] = epg.get("dtype")

        if files.get("NLI_output") is not None:
            nli_abs = _abs_from_dst_rel(files["NLI_output"]["dst"])
            ng = extract_geometry(nli_abs, nii_path=None)
            geo["orig_nli_output_shape"] = ng.get("orig_shape")
            geo["orig_nli_output_dtype"] = ng.get("dtype")

        # error reporting for spacing (common pain point)
        if x_nii_abs is None:
            errors["X_nii"] = {"error": "missing_X_nii (spacing will be None; resample disabled)"}
        elif geo["orig_image_spacing"] is None:
            errors["X_nii"] = {"error": "failed_to_extract_spacing_from_X_nii"}

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
# Demo / Script entry
# -----------------------------
if __name__ == "__main__":
    root_path = Path("~/Desktop/vibes_pipe/experiments/mre_data_prep_test").expanduser()
    pairs_json_path = root_path / "pairs.json"
    workspace = root_path / "workspace_root"
    manifest_path = workspace / "manifest.json"

    pairs_data = read_json(pairs_json_path)
    if isinstance(pairs_data, dict) and "pairs" in pairs_data:
        pairs_data = pairs_data["pairs"]

    if not isinstance(pairs_data, list):
        raise ValueError(
            "Expected pairs.json to contain a top-level list of pair items "
            "(or a dict with key 'pairs' holding the list)."
        )

    manifest_data = build_workspace_from_pairs(
        pairs_spec=pairs_data,
        workspace_root=workspace,
        overwrite=False,
        compute_hash=True,
    )
    write_json_atomic(manifest_path, manifest_data)
    print(f"[workspace_prep] manifest: {manifest_path.resolve()}")
    print(f"[workspace_prep] pairs: {len(manifest_data.get('pairs', []))}")