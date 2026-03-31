#!/usr/bin/env python3
"""
workspace_prep.py

Freeze a dataset into a reproducible workspace layout + manifest, using your
pairs.json field names (no X/GT renaming).

Expected pairs.json schema (per item):
  Required:
    - id: str
    - split: "train"|"val"|"test" (optional; default "train")
    - t2stack: str (path to .mat)
    - GT(human): str (path to mask .mat)
    - t2stack_nii: str (path to .nii/.nii.gz)  # required by your choice
  Optional:
    - eligible_preds: None | str | list[str]   (paths to pred mats)
    - NLI_output: None | str                   (path to Mu mat)
    - meta: dict
    - scanner_type: None | "GE" | "SIEMENS"
    - OPT_NOISE = "noise_profile"    # None | str

Workspace layout:
  <workspace_root>/{train,val,test}/{id}/
      t2stack.mat
      GT(human).mat
      t2stack.nii or t2stack.nii.gz
      eligible_preds_00.mat, eligible_preds_01.mat, ... (optional)
      NLI_output.mat                                  (optional)

Manifest:
  - Stores workspace-relative dst paths + optional sha256
  - Records geometry metadata via vibes_pipe.data.io_mat.extract_geometry
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from vibes_pipe.data.io_mat import extract_geometry
from vibes_pipe.utils.json_io import iso8601_utc_now, read_json, write_json_atomic

JsonDict = Dict[str, Any]

ALLOWED_SPLITS = {"train", "val", "test"}

# pairs.json required keys (your schema)
REQ_ID = "id"
REQ_T2STACK = "t2stack"
REQ_GT = "GT(human)"
REQ_T2NII = "t2stack_nii"

# pairs.json optional keys (your schema)
OPT_PREDS = "eligible_preds"   # None | str | list[str]
OPT_NLI = "NLI_output"         # None | str
OPT_META = "meta"
OPT_SCANNER = "scanner_type"
OPT_NOISE = "noise_profile"    # none | str
# -----------------------------
# File utilities
# -----------------------------
def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: str | Path, dst: str | Path, *, overwrite: bool = False) -> Path:
    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    if not src_p.exists() or not src_p.is_file():
        raise FileNotFoundError(f"Source file not found: {src_p}")

    dst_p.parent.mkdir(parents=True, exist_ok=True)

    if dst_p.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists (overwrite=False): {dst_p}")

    with src_p.open("rb") as sf, dst_p.open("wb") as df:
        for chunk in iter(lambda: sf.read(1024 * 1024), b""):
            df.write(chunk)

    return dst_p


def _file_entry(
    *,
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
# Pair spec validation + normalization
# -----------------------------
def _as_abs_file_path(raw_path: Any, *, field_name: str, pair_id: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Invalid `{field_name}` for pair `{pair_id}`: expected a non-empty string path.")
    p = Path(raw_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing `{field_name}` file for pair `{pair_id}`: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"`{field_name}` is not a file for pair `{pair_id}`: {p}")
    return p


def _as_abs_file_list(raw: Any, *, field_name: str, pair_id: str) -> Optional[List[Path]]:
    """
    Accept:
      - None
      - "path"
      - ["path1", "path2", ...]
    Return:
      - None or list[Path]
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [_as_abs_file_path(raw, field_name=field_name, pair_id=pair_id)]
    if isinstance(raw, list):
        paths: List[Path] = []
        for i, x in enumerate(raw):
            if not isinstance(x, str) or not x.strip():
                raise ValueError(f"Invalid `{field_name}[{i}]` for pair `{pair_id}`: expected a non-empty string path.")
            paths.append(_as_abs_file_path(x, field_name=f"{field_name}[{i}]", pair_id=pair_id))
        return paths
    raise ValueError(f"Invalid `{field_name}` for pair `{pair_id}`: expected None, str, or list[str].")


def _validate_and_normalize_pair(item: Any, index: int) -> JsonDict:
    if not isinstance(item, dict):
        raise ValueError(f"pairs_spec[{index}] must be a dict, got {type(item).__name__}.")

    missing = [k for k in (REQ_ID, REQ_T2STACK, REQ_GT, REQ_T2NII) if k not in item]
    if missing:
        raise ValueError(f"pairs_spec[{index}] missing required keys: {missing}")

    pair_id = item[REQ_ID]
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError(f"pairs_spec[{index}]['id'] must be a non-empty string.")
    pair_id = pair_id.strip()

    split = item.get("split", "train")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Invalid split for pair `{pair_id}`: {split!r}. Allowed values: {sorted(ALLOWED_SPLITS)}"
        )

    t2stack_path = _as_abs_file_path(item[REQ_T2STACK], field_name=REQ_T2STACK, pair_id=pair_id)
    gt_path = _as_abs_file_path(item[REQ_GT], field_name=REQ_GT, pair_id=pair_id)
    t2nii_path = _as_abs_file_path(item[REQ_T2NII], field_name=REQ_T2NII, pair_id=pair_id)

    preds_paths = _as_abs_file_list(item.get(OPT_PREDS), field_name=OPT_PREDS, pair_id=pair_id)

    nli_raw = item.get(OPT_NLI)
    nli_path: Optional[Path]
    if nli_raw is None:
        nli_path = None
    else:
        nli_path = _as_abs_file_path(nli_raw, field_name=OPT_NLI, pair_id=pair_id)

    meta = item.get(OPT_META, {}) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"`meta` for pair `{pair_id}` must be a dict if provided.")

    scanner_type = item.get(OPT_SCANNER, None)
    if scanner_type is not None:
        if not isinstance(scanner_type, str) or not scanner_type.strip():
            raise ValueError(f"`scanner_type` for pair `{pair_id}` must be a non-empty string if provided.")
        scanner_type = scanner_type.upper().strip()
        if scanner_type not in {"GE", "SIEMENS"}:
            raise ValueError(
                f"Invalid scanner_type for pair `{pair_id}`: {scanner_type!r}. "
                "Allowed values: 'GE', 'SIEMENS'."
            )
    
    noise_raw = item.get(OPT_NOISE)
    noise_path: Optional[Path]
    if noise_raw is None:
        noise_path = None
    else:
        noise_path = _as_abs_file_path(noise_raw, field_name=OPT_NOISE, pair_id=pair_id)
        
    return {
    "id": pair_id,
    "split": split,
    REQ_T2STACK: t2stack_path,
    REQ_GT: gt_path,
    REQ_T2NII: t2nii_path,
    OPT_PREDS: preds_paths,
    OPT_NLI: nli_path,
    OPT_META: meta,
    OPT_SCANNER: scanner_type,
    OPT_NOISE: noise_path
}


def validate_pairs_spec(pairs_spec: Sequence[JsonDict]) -> List[JsonDict]:
    if not isinstance(pairs_spec, Sequence) or isinstance(pairs_spec, (str, bytes)):
        raise ValueError("pairs_spec must be a list of dict items.")

    normalized: List[JsonDict] = []
    seen_ids = set()

    for idx, item in enumerate(pairs_spec):
        pair = _validate_and_normalize_pair(item, idx)
        pid = pair["id"]
        if pid in seen_ids:
            raise ValueError(f"Duplicate pair id found: `{pid}`")
        seen_ids.add(pid)
        normalized.append(pair)

    return normalized


# -----------------------------
# Workspace + manifest building
# -----------------------------
def build_workspace_from_pairs(
    pairs_spec: Sequence[JsonDict],
    workspace_root: str | Path,
    *,
    overwrite: bool = False,
    compute_hash: bool = True,
) -> JsonDict:
    """
    Copy files into a frozen workspace and return a manifest dict.

    Geometry:
      - image spacing from t2stack_nii
      - label spacing follows image spacing
    """
    normalized = validate_pairs_spec(pairs_spec)

    ws_root = Path(workspace_root).expanduser().resolve()
    ws_root.mkdir(parents=True, exist_ok=True)

    manifest_pairs: List[JsonDict] = []

    def _abs_from_dst(dst_rel: str) -> Path:
        return ws_root / Path(dst_rel)

    for pair in sorted(normalized, key=lambda p: p["id"]):
        pair_id: str = pair["id"]
        split: str = pair["split"]
        pair_dir = Path(split) / pair_id

        # ---- copy required ----
        t2_entry = _file_entry(
            src_path=pair[REQ_T2STACK],
            dst_rel=pair_dir / "t2stack.mat",
            workspace_root=ws_root,
            overwrite=overwrite,
            compute_hash=compute_hash,
        )

        gt_entry = _file_entry(
            src_path=pair[REQ_GT],
            dst_rel=pair_dir / "GT(human).mat",
            workspace_root=ws_root,
            overwrite=overwrite,
            compute_hash=compute_hash,
        )

        # preserve .nii vs .nii.gz
        t2nii_src: Path = pair[REQ_T2NII]
        t2nii_suffix = "".join(t2nii_src.suffixes)  # ".nii" or ".nii.gz"
        t2nii_entry = _file_entry(
            src_path=t2nii_src,
            dst_rel=pair_dir / f"t2stack{t2nii_suffix}",
            workspace_root=ws_root,
            overwrite=overwrite,
            compute_hash=compute_hash,
        )

        files: JsonDict = {
            REQ_T2STACK: t2_entry,
            REQ_GT: gt_entry,
            REQ_T2NII: t2nii_entry,
            OPT_PREDS: None,
            OPT_NLI: None,
            OPT_NOISE: None
        }

        # ---- copy optional: eligible_preds (0..n) ----
        preds: Optional[List[Path]] = pair.get(OPT_PREDS)
        if preds:
            pred_entries: List[JsonDict] = []
            for j, src in enumerate(preds):
                pred_entries.append(
                    _file_entry(
                        src_path=src,
                        dst_rel=pair_dir / f"eligible_preds_{j:02d}.mat",
                        workspace_root=ws_root,
                        overwrite=overwrite,
                        compute_hash=compute_hash,
                    )
                )
            files[OPT_PREDS] = pred_entries

        # ---- copy optional: NLI_output ----
        nli_src: Optional[Path] = pair.get(OPT_NLI)
        if nli_src is not None:
            nli_entry = _file_entry(
                src_path=nli_src,
                dst_rel=pair_dir / "NLI_output.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files[OPT_NLI] = nli_entry

        # ---- copy optional: noise_profile ----
        noise_src: Optional[Path] = pair.get(OPT_NOISE)
        if noise_src is not None:
            noise_entry = _file_entry(
                src_path=noise_src,
                dst_rel=pair_dir / "noise_profile.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files[OPT_NOISE] = noise_entry
            
            
        # ---- geometry meta ----
        meta = dict(pair.get(OPT_META, {}) or {})
        geo = dict(meta.get("geometry_preprocess", {}) or {})
        errors: Dict[str, Any] = {}

        t2_mat_abs = _abs_from_dst(t2_entry["dst"])
        gt_mat_abs = _abs_from_dst(gt_entry["dst"])
        t2_nii_abs = _abs_from_dst(t2nii_entry["dst"])

        xg = extract_geometry(t2_mat_abs, nii_path=t2_nii_abs)   # expects spacing from nii
        gg = extract_geometry(gt_mat_abs, nii_path=None)

        geo["orig_t2stack_shape"] = xg.get("orig_shape")
        geo["orig_t2stack_spacing"] = xg.get("orig_spacing")
        geo["orig_t2stack_dtype"] = xg.get("dtype")

        geo["orig_GT(human)_shape"] = gg.get("orig_shape")
        geo["orig_GT(human)_spacing"] = xg.get("orig_spacing")  # mask follows image spacing
        geo["orig_GT(human)_dtype"] = gg.get("dtype")

        if geo["orig_t2stack_spacing"] is None:
            errors[REQ_T2NII] = {"error": "failed_to_extract_spacing_from_t2stack_nii"}

        # optional mats geometry
        if files.get(OPT_PREDS):
            preds_geo: List[Dict[str, Any]] = []
            for ent in files[OPT_PREDS]:
                ep_abs = _abs_from_dst(ent["dst"])
                epg = extract_geometry(ep_abs, nii_path=None)
                preds_geo.append(
                    {
                        "dst": ent["dst"],
                        "shape": epg.get("orig_shape"),
                        "dtype": epg.get("dtype"),
                    }
                )
            geo["eligible_preds"] = preds_geo

        if files.get(OPT_NLI) is not None:
            nli_abs = _abs_from_dst(files[OPT_NLI]["dst"])
            ng = extract_geometry(nli_abs, nii_path=None)
            geo["orig_NLI_output_shape"] = ng.get("orig_shape")
            geo["orig_NLI_output_dtype"] = ng.get("dtype")

        if files.get(OPT_NOISE) is not None:
            noise_abs = _abs_from_dst(files[OPT_NOISE]["dst"])
            zg = extract_geometry(noise_abs, nii_path=None)
            geo["orig_noise_profile_shape"] = zg.get("orig_shape")
            geo["orig_noise_profile_dtype"] = zg.get("dtype")
            
        if errors:
            geo["errors"] = errors

        meta["geometry_preprocess"] = geo

        manifest_pairs.append(
            {
                "id": pair_id,
                "split": split,
                "scanner_type": pair.get(OPT_SCANNER),
                "files": files,
                "meta": meta,
            }
        )

    return {
        "schema_version": "1.0",
        "created_utc": iso8601_utc_now(),
        "workspace_root": str(ws_root),
        "splits": ["train", "val", "test"],
        "pairs": manifest_pairs,
    }


# -----------------------------
# Script entry (optional demo)
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
        raise ValueError("Expected pairs.json to be a top-level list (or dict with key 'pairs').")

    manifest = build_workspace_from_pairs(
        pairs_spec=pairs_data,
        workspace_root=workspace,
        overwrite=False,
        compute_hash=True,
    )
    write_json_atomic(manifest_path, manifest)
    print(f"[workspace_prep] manifest: {manifest_path.resolve()}")
    print(f"[workspace_prep] pairs: {len(manifest.get('pairs', []))}")