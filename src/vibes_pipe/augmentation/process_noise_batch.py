"""
Batch extraction of noise profiles from a mixed-scanner manifest.

Creates:
    <workspace_root>/noise_profiles/GE/
    <workspace_root>/noise_profiles/SIEMENS/
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import scipy.io as sio

from .ge_noise import compute_ge_noise
from .siemens_noise import compute_siemens_noise
from ..utils.is_valid_mat_file import *
from ..utils.load_mat_arrary import *


def _find_candidate_files(
    folder: Path,
    mat_pattern: str,
    required_keys: Tuple[str, ...],
    recursive: bool = True,
) -> List[Path]:
    """Find candidate MAT files exactly like v1 style."""
    if recursive:
        candidates = [p for p in folder.rglob(mat_pattern) if is_valid_mat_file(p, required_keys)]
        if not candidates:
            candidates = [p for p in folder.rglob("*.mat") if is_valid_mat_file(p, required_keys)]
    else:
        candidates = [p for p in folder.glob(mat_pattern) if is_valid_mat_file(p, required_keys)]
        if not candidates:
            candidates = [p for p in folder.glob("*.mat") if is_valid_mat_file(p, required_keys)]

    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates


def _load_first_valid_candidate(
    candidates: List[Path],
    preferred_keys: Tuple[str, ...],
    verbose: bool = True,
):
    """Try loading candidates until one works."""
    arr = None
    used_path = None

    for candidate in candidates:
        try:
            arr = load_mat_array(candidate, preferred_keys)
            used_path = candidate
            if verbose:
                print(f"  Loaded: {candidate.name}")
            break
        except Exception as e:
            if verbose:
                print(f"  Skipped {candidate.name}: {e}")
            continue

    return arr, used_path


def _infer_subject_folder_from_pair(pair: Dict[str, Any]) -> Path:
    """
    Infer the original subject folder from manifest entry.
    Uses parent of the original t2stack source file.
    """
    src_path = Path(pair["files"]["t2stack"]["src"])
    return src_path.parent


def _process_ge_subject(
    pid: str,
    subject_folder: Path,
    output_dir: Path,
    mat_pattern: str = "*magimg*.mat",
    verbose: bool = True,
) -> Dict[str, Any]:
    result = {
        "status": "success",
        "error": None,
        "output_file": None,
        "scanner_type": "GE",
        "shapes": {},
        "source_file": None,
    }

    try:
        candidates = _find_candidate_files(
            folder=subject_folder,
            mat_pattern=mat_pattern,
            required_keys=("magimg",),
            recursive=True,
        )

        if not candidates:
            raise FileNotFoundError(f"No magimg.mat files found in {subject_folder}")

        magimg, used_path = _load_first_valid_candidate(
            candidates,
            preferred_keys=("magimg",),
            verbose=verbose,
        )

        if magimg is None or used_path is None:
            raise RuntimeError("Could not load any valid magimg array")

        result["source_file"] = str(used_path)
        result["shapes"]["magimg"] = magimg.shape

        t2stack, noise, noise_scaled, t2noise = compute_ge_noise(magimg)

        result["shapes"].update({
            "t2stack": t2stack.shape,
            "noise": noise.shape,
            "noise_scaled": noise_scaled.shape,
            "t2noise": t2noise.shape,
        })

        save_path = output_dir / f"{pid}_noise.mat"
        sio.savemat(
            str(save_path),
            {
                "subject": pid,
                "scanner_type": "GE",
                "source_file": str(used_path),
                "t2stack": t2stack,
                "noise": noise,
                "noise_scaled": noise_scaled,
                "t2noise": t2noise,
            },
            do_compression=True,
        )

        result["output_file"] = str(save_path)

        del magimg, t2stack, noise, noise_scaled, t2noise
        gc.collect()

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def _process_siemens_subject(
    pid: str,
    subject_folder: Path,
    output_dir: Path,
    mat_pattern: str = "*imgraw*.mat",
    preferred_keys: Tuple[str, ...] = ("imgraw",),
    ref_dim: int = -1,
    ref_idx: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    result = {
        "status": "success",
        "error": None,
        "output_file": None,
        "scanner_type": "SIEMENS",
        "shapes": {},
        "metadata": None,
        "source_file": None,
    }

    try:
        candidates = _find_candidate_files(
            folder=subject_folder,
            mat_pattern=mat_pattern,
            required_keys=("imgraw",),
            recursive=False,   # same as your v1 Siemens code
        )

        if not candidates:
            raise FileNotFoundError(f"No imgraw.mat files found in {subject_folder}")

        imgraw, used_path = _load_first_valid_candidate(
            candidates,
            preferred_keys=preferred_keys,
            verbose=verbose,
        )

        if imgraw is None or used_path is None:
            raise RuntimeError("Could not load any valid imgraw array")

        result["source_file"] = str(used_path)
        result["shapes"]["imgraw"] = imgraw.shape

        t2stack, noise, noise_scaled, t2noise, metadata = compute_siemens_noise(
            imgraw,
            ref_dim=ref_dim,
            ref_idx=ref_idx,
            show_reference=verbose,
        )

        result["shapes"].update({
            "t2stack": t2stack.shape,
            "noise": noise.shape,
            "noise_scaled": noise_scaled.shape,
            "t2noise": t2noise.shape,
        })
        result["metadata"] = metadata

        save_path = output_dir / f"{pid}_noise.mat"
        sio.savemat(
            str(save_path),
            {
                "subject": pid,
                "scanner_type": "SIEMENS",
                "source_file": str(used_path),
                "t2stack": t2stack,
                "noise": noise,
                "noise_scaled": noise_scaled,
                "t2noise": t2noise,
                "ref_dim": np.int32(metadata["ref_dim"]),
                "ref_idx": np.int32(metadata["ref_idx"]),
            },
            do_compression=True,
        )

        result["output_file"] = str(save_path)

        del imgraw, t2stack, noise, noise_scaled, t2noise
        gc.collect()

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def run_batch(manifest_path, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Process all subjects in a mixed-scanner manifest, but search each subject
    folder for the raw source MATs exactly like v1.
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    workspace_root = Path(manifest["workspace_root"])
    base_noise_dir = workspace_root / "noise_profiles"
    ge_dir = base_noise_dir / "GE"
    siemens_dir = base_noise_dir / "SIEMENS"

    ge_dir.mkdir(parents=True, exist_ok=True)
    siemens_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    pairs = manifest.get("pairs", [])
    total = len(pairs)
    start_time = time.time()

    print("Starting mixed-scanner batch processing...")

    for idx, pair in enumerate(pairs, 1):
        pid = str(pair["id"])
        scanner_type = str(pair.get("scanner_type", "")).upper().strip()

        print(f"\n[{idx}/{total}] Processing {pid} ({scanner_type})...")

        if scanner_type not in {"GE", "SIEMENS"}:
            results[pid] = {
                "status": "failed",
                "scanner_type": scanner_type,
                "error": f"Invalid scanner_type: {scanner_type!r}",
                "output_file": None,
            }
            print(f"  Skip {pid}: bad scanner_type={scanner_type!r}")
            continue

        try:
            noise_entry = pair["files"]["noise_profile"]
            if noise_entry is None:
                raise KeyError("noise_profile is null")
            src_path = Path(noise_entry["src"])
        except KeyError:
            results[pid] = {
                "status": "failed",
                "scanner_type": scanner_type,
                "error": "Missing pair['files']['noise_profile']['src']",
                "output_file": None,
            }
            print(f"  Skip {pid}: missing noise_profile source path in manifest")
            continue

        if not src_path.exists():
            results[pid] = {
                "status": "failed",
                "scanner_type": scanner_type,
                "error": f"File not found: {src_path}",
                "output_file": None,
            }
            print(f"  Skip {pid}: file not found at {src_path}")
            continue

        if scanner_type == "GE":
            result = _process_ge_subject(pid, src_path, ge_dir, verbose=verbose)
        else:
            result = _process_siemens_subject(pid, src_path, siemens_dir, verbose=verbose)

        results[pid] = result

        if result["status"] == "success":
            print(f"  Saved: {result['output_file']}")
        else:
            print(f"  Failed: {result['error']}")

    total_time = time.time() - start_time
    success_count = sum(1 for r in results.values() if r["status"] == "success")

    print(f"\n{'=' * 60}")
    print(f"Completed: {success_count}/{total} subjects")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    manifest_path = "/home/smooi/Desktop/vibes_pipe/experiments/toast_data/workspace_root/manifest.json"
    run_batch(manifest_path)