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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


JsonDict = Dict[str, Any]
ALLOWED_SPLITS = {"train", "val", "test"}
REQUIRED_KEYS = ("id", "X", "GT")
OPTIONAL_FILE_KEYS = ("eligible_preds", "NLI_output")


def read_json(path: str | Path) -> Any:
    """
    Read and parse JSON from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """
    Atomically write JSON via temp file + os.replace.

    Args:
        path: Destination JSON path.
        obj: JSON-serializable object.
    """
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


def sha256_file(path: str | Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: File path.

    Returns:
        Hex-encoded SHA256 digest.
    """
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
    """
    Copy file from src to dst with parent creation and overwrite control.

    Args:
        src: Source file path.
        dst: Destination file path.
        overwrite: If False, raise if dst exists.

    Returns:
        Destination path.

    Raises:
        FileNotFoundError: If src does not exist.
        FileExistsError: If dst exists and overwrite=False.
    """
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
    """
    Validate and normalize pairs_spec.

    Rules:
    - required keys: id, X, GT
    - required files must exist
    - optional files must exist if not None
    - split defaults to "train" and must be in {"train","val","test"}

    Args:
        pairs_spec: List of pair specification dicts.

    Returns:
        Normalized pair dicts with absolute Path objects for file fields.
    """
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


def _iso8601_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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

    Also records original (pre-augmentation) array shapes/dtypes into `meta`
    by inspecting the copied .mat files:
      - meta["orig_shape_X"], meta["orig_dtype_X"]
      - meta["orig_shape_GT"], meta["orig_dtype_GT"]
      - (optionally) eligible_preds / NLI_output if present

    Notes:
      - If scipy is not available, shape fields will be None.
      - If a .mat is empty or unreadable, shape fields will be None.
    """
    from pathlib import Path

    # --- optional: MAT inspection helper (kept local to avoid global deps) ---
    def _mat_summary(mat_path: Path) -> dict:
        """
        Return {'shape': list|None, 'dtype': str|None, 'key': str|None, 'error': str|None}
        Best-effort: picks the first non-metadata variable that looks array-like.
        """
        out = {"shape": None, "dtype": None, "key": None, "error": None}

        # empty file => empty sha (common in your tests)
        try:
            if mat_path.stat().st_size == 0:
                out["error"] = "empty_file"
                return out
        except Exception as e:
            out["error"] = f"stat_failed:{e}"
            return out

        try:
            from scipy.io import loadmat  # type: ignore
        except Exception:
            out["error"] = "scipy_not_available"
            return out

        try:
            md = loadmat(str(mat_path))
            # filter out MATLAB metadata keys
            keys = [k for k in md.keys() if not k.startswith("__")]
            # choose first array-like value
            for k in keys:
                v = md.get(k)
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", None)
                if shape is not None:
                    out["key"] = k
                    out["shape"] = list(shape)
                    out["dtype"] = str(dtype) if dtype is not None else None
                    return out
            out["error"] = "no_array_found"
            return out
        except Exception as e:
            out["error"] = f"loadmat_failed:{e}"
            return out

    # --- main ---
    normalized = validate_pairs_spec(pairs_spec)
    ws_root = Path(workspace_root).expanduser().resolve()
    ws_root.mkdir(parents=True, exist_ok=True)

    manifest_pairs: List[JsonDict] = []
    for pair in sorted(normalized, key=lambda p: p["id"]):
        pair_id = pair["id"]
        split = pair["split"]
        pair_dir = Path(split) / pair_id

        # Copy files + build manifest entries
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

        eligible_entry = None
        if pair["eligible_preds"] is not None:
            eligible_entry = _file_entry(
                src_path=pair["eligible_preds"],
                dst_rel=pair_dir / "eligible_preds.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["eligible_preds"] = eligible_entry

        nli_entry = None
        if pair["NLI_output"] is not None:
            nli_entry = _file_entry(
                src_path=pair["NLI_output"],
                dst_rel=pair_dir / "NLI_output.mat",
                workspace_root=ws_root,
                overwrite=overwrite,
                compute_hash=compute_hash,
            )
            files["NLI_output"] = nli_entry

        # Record original sizes/dtypes into meta (best-effort; based on copied files)
        meta = dict(pair.get("meta", {}) or {})

        def _abs_from_dst_rel(dst_rel_str: str) -> Path:
            return ws_root / Path(dst_rel_str)

        x_sum = _mat_summary(_abs_from_dst_rel(x_entry["dst"]))
        gt_sum = _mat_summary(_abs_from_dst_rel(gt_entry["dst"]))

        meta["orig_shape_X"] = x_sum["shape"]
        meta["orig_dtype_X"] = x_sum["dtype"]
        meta["mat_key_X"] = x_sum["key"]
        if x_sum["error"]:
            meta["orig_shape_X_error"] = x_sum["error"]

        meta["orig_shape_GT"] = gt_sum["shape"]
        meta["orig_dtype_GT"] = gt_sum["dtype"]
        meta["mat_key_GT"] = gt_sum["key"]
        if gt_sum["error"]:
            meta["orig_shape_GT_error"] = gt_sum["error"]

        if eligible_entry is not None:
            ep_sum = _mat_summary(_abs_from_dst_rel(eligible_entry["dst"]))
            meta["orig_shape_eligible_preds"] = ep_sum["shape"]
            meta["orig_dtype_eligible_preds"] = ep_sum["dtype"]
            meta["mat_key_eligible_preds"] = ep_sum["key"]
            if ep_sum["error"]:
                meta["orig_shape_eligible_preds_error"] = ep_sum["error"]

        if nli_entry is not None:
            nli_sum = _mat_summary(_abs_from_dst_rel(nli_entry["dst"]))
            meta["orig_shape_NLI_output"] = nli_sum["shape"]
            meta["orig_dtype_NLI_output"] = nli_sum["dtype"]
            meta["mat_key_NLI_output"] = nli_sum["key"]
            if nli_sum["error"]:
                meta["orig_shape_NLI_output_error"] = nli_sum["error"]

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



if __name__ == "__main__":
    # demo reading pairs.json, building workspace, writing manifest.json
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
